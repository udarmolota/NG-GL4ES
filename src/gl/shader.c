#include "shader.h"
#include <time.h>

#include "../glx/hardext.h"
#include "debug.h"
#include "fpe_shader.h"
#include "init.h"
#include "gl4es.h"
#include "glstate.h"
#include "loader.h"
#include "shaderconv.h"
#include "vgpu/shaderconv.h"
#include "glsl/glsl_for_es.h"
#include <ctype.h>
#include <string.h>

// #define DEBUG
#ifdef DEBUG
#define DBG(a) a
#else
#define DBG(a)
#endif

KHASH_MAP_IMPL_INT(shaderlist, shader_t*);

// ============================================================
// ZOMDROID FIX (invisible character root): PZ shaders redefine GLSL builtins
// (max/min/clamp) for ancient GLSL 1.20 compatibility. GLES3 forbids redefining
// builtins — Adreno fails at LINK ("'max' : can't redefine/overload built-in
// functions!") → model programs never link → character/zombies invisible.
// ES3 has all needed overloads natively, so strip PZ's prototypes AND definitions.
// Removed regions are blanked with spaces (newlines kept → stable line numbers).
static int zomdroid_is_type_word(const char* w, int len) {
    static const char* t[] = {"float", "int",   "uint",  "vec2",  "vec3",  "vec4", "ivec2",
                              "ivec3", "ivec4", "uvec2", "uvec3", "uvec4", "bool"};
    for (unsigned zi = 0; zi < sizeof(t) / sizeof(t[0]); zi++)
        if ((int)strlen(t[zi]) == len && !strncmp(w, t[zi], len)) return 1;
    return 0;
}
static void zomdroid_strip_builtin_redefs(char* src) {
    // STRIP strategy (v3): remove PZ's prototypes AND definitions of builtin
    // max/min/clamp. Their mixed-type call sites (the reason PZ wrote the overloads)
    // are legalized by injecting GL_EXT_shader_implicit_conversions into the converted
    // shader (see the SIMPLE path). Rename-per-shader (v2) failed: PZ links several
    // compilation units per stage, definitions live in a shared unit while calls sit
    // in others — per-shader renaming broke the cross-unit resolution
    // ("INTERNAL ERROR: Call to undefined function").
    static const char* names[] = {"max", "min", "clamp"};
    if (!src) return;
    for (unsigned bi = 0; bi < sizeof(names) / sizeof(names[0]); bi++) {
        const char* name = names[bi];
        size_t nlen = strlen(name);
        char* p = src;
        while ((p = strstr(p, name)) != NULL) {
            if ((p > src && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) || isalnum((unsigned char)p[nlen]) ||
                p[nlen] == '_') {
                p += nlen;
                continue;
            }
            char* q = p + nlen;
            while (*q == ' ' || *q == '\t') q++;
            if (*q != '(') {
                p += nlen;
                continue;
            }
            char* w = p;
            while (w > src && (w[-1] == ' ' || w[-1] == '\t')) w--;
            char* ws = w;
            while (ws > src && (isalnum((unsigned char)ws[-1]) || ws[-1] == '_')) ws--;
            if (ws == w || !zomdroid_is_type_word(ws, (int)(w - ws))) {
                p += nlen;
                continue;
            }
            int depth = 0;
            char* e = q;
            while (*e) {
                if (*e == '(') depth++;
                else if (*e == ')') {
                    depth--;
                    if (!depth) break;
                }
                e++;
            }
            if (*e != ')') break;
            e++;
            while (*e == ' ' || *e == '\t' || *e == '\n' || *e == '\r') e++;
            char* zend = NULL;
            if (*e == ';') zend = e + 1;
            else if (*e == '{') {
                int bd = 0;
                char* b = e;
                while (*b) {
                    if (*b == '{') bd++;
                    else if (*b == '}') {
                        bd--;
                        if (!bd) {
                            zend = b + 1;
                            break;
                        }
                    }
                    b++;
                }
            }
            if (!zend) {
                p += nlen;
                continue;
            }
            for (char* z = ws; z < zend; z++)
                if (*z != '\n') *z = ' ';
            p = zend;
        }
    }
}

// ZOMDROID FIX (vehicle body / blur / water shaders — Adreno link failures):
// Adreno's ESSL has NO implicit int->float conversions (and it ignores our
// GL_EXT_shader_implicit_conversions request), so two desktop-GLSL idioms in PZ
// shaders fail at link:
//   1) const float X = <int expression>;   -> "non-matching types for const initializer"
//   2) clamp(x, 0, 1) etc.                 -> "no matching overloaded function found"
// Fix both at source level: wrap const-float initializers in float(...), and turn bare
// integer literals inside builtin-call arguments into float literals (0 -> 0.0).
static char* zomdroid_floatify_const_float(char* src) {
    if (!src) return src;
    int hits = 0;
    for (const char* p = src; (p = strstr(p, "const float")) != NULL; p += 11)
        hits++;
    if (!hits) return src;
    char* out = (char*)malloc(strlen(src) + hits * 8 + 1);
    if (!out) return src;
    const char* in = src;
    char* o = out;
    while (*in) {
        if (!strncmp(in, "const float", 11) && !(isalnum((unsigned char)in[11]) || in[11] == '_') &&
            (in == src || !(isalnum((unsigned char)in[-1]) || in[-1] == '_'))) {
            const char* eq = in;
            while (*eq && *eq != '=' && *eq != ';' && *eq != '{')
                eq++;
            const char* semi = (*eq == '=') ? strchr(eq, ';') : NULL;
            if (semi) {
                size_t head = (size_t)(eq - in) + 1;
                memcpy(o, in, head);
                o += head;
                memcpy(o, " float(", 7);
                o += 7;
                memcpy(o, eq + 1, (size_t)(semi - eq - 1));
                o += semi - eq - 1;
                *o++ = ')';
                *o++ = ';';
                in = semi + 1;
                continue;
            }
        }
        *o++ = *in++;
    }
    *o = '\0';
    free(src);
    return out;
}

static char* zomdroid_floatify_builtin_args(char* src) {
    if (!src) return src;
    static const char* fns[] = {"clamp", "min", "max", "mix", "step", "smoothstep", "pow", "mod"};
    size_t slen = strlen(src);
    // mark every char that sits inside a builtin-call argument list
    unsigned char* mark = (unsigned char*)calloc(slen + 1, 1);
    if (!mark) return src;
    for (unsigned fi = 0; fi < sizeof(fns) / sizeof(fns[0]); fi++) {
        const char* name = fns[fi];
        size_t nlen = strlen(name);
        const char* p = src;
        while ((p = strstr(p, name)) != NULL) {
            if ((p > src && (isalnum((unsigned char)p[-1]) || p[-1] == '_' || p[-1] == '.')) ||
                isalnum((unsigned char)p[nlen]) || p[nlen] == '_') {
                p += nlen;
                continue;
            }
            const char* q = p + nlen;
            while (*q == ' ' || *q == '\t') q++;
            if (*q != '(') {
                p += nlen;
                continue;
            }
            int depth = 0;
            const char* e = q;
            while (*e) {
                if (*e == '(') depth++;
                else if (*e == ')') {
                    depth--;
                    if (!depth) break;
                }
                e++;
            }
            if (*e == ')')
                for (const char* z = q + 1; z < e; z++)
                    mark[z - src] = 1;
            p = q;
        }
    }
    // count integer literals inside marked spans (outside [] indexing)
    int hits = 0;
    for (size_t i = 0; i < slen; i++) {
        if (!mark[i] || !isdigit((unsigned char)src[i])) continue;
        if (i > 0 && (isalnum((unsigned char)src[i - 1]) || src[i - 1] == '_' || src[i - 1] == '.')) continue;
        // ZOMDROID FIX (Opus): exponent digits ("1.0e-10") are part of a FLOAT literal
        if (i >= 2 && (src[i - 1] == '-' || src[i - 1] == '+') && (src[i - 2] == 'e' || src[i - 2] == 'E')) continue;
        size_t j = i;
        while (j < slen && isdigit((unsigned char)src[j]))
            j++;
        if (j < slen && (src[j] == '.' || src[j] == 'x' || src[j] == 'X' || src[j] == 'e' || src[j] == 'E' ||
                         src[j] == 'u' || src[j] == 'U' || isalpha((unsigned char)src[j]) || src[j] == '_')) {
            i = j;
            continue;
        }
        // skip array indices: look back for unbalanced '[' within the marked span
        int bd = 0;
        for (size_t k = i; k > 0 && mark[k - 1]; k--) {
            if (src[k - 1] == ']') bd++;
            else if (src[k - 1] == '[') bd--;
        }
        if (bd < 0) {
            i = j;
            continue;
        }
        hits++;
        i = j;
    }
    if (!hits) {
        free(mark);
        return src;
    }
    char* out = (char*)malloc(slen + hits * 2 + 1);
    if (!out) {
        free(mark);
        return src;
    }
    char* o = out;
    for (size_t i = 0; i < slen;) {
        if (mark[i] && isdigit((unsigned char)src[i]) &&
            !(i > 0 && (isalnum((unsigned char)src[i - 1]) || src[i - 1] == '_' || src[i - 1] == '.')) &&
            !(i >= 2 && (src[i - 1] == '-' || src[i - 1] == '+') && (src[i - 2] == 'e' || src[i - 2] == 'E'))) {
            size_t j = i;
            while (j < slen && isdigit((unsigned char)src[j]))
                j++;
            int isint = !(j < slen && (src[j] == '.' || src[j] == 'x' || src[j] == 'X' || src[j] == 'e' ||
                                       src[j] == 'E' || src[j] == 'u' || src[j] == 'U' ||
                                       isalpha((unsigned char)src[j]) || src[j] == '_'));
            int bd = 0;
            for (size_t k = i; k > 0 && mark[k - 1]; k--) {
                if (src[k - 1] == ']') bd++;
                else if (src[k - 1] == '[') bd--;
            }
            memcpy(o, src + i, j - i);
            o += j - i;
            if (isint && bd >= 0) {
                *o++ = '.';
                *o++ = '0';
            }
            i = j;
            continue;
        }
        *o++ = src[i++];
    }
    *o = '\0';
    free(mark);
    free(src);
    return out;
}

// ZOMDROID PERF (conversion cache, floor 1 of the warm-cache plan, 2026-07-29):
// RC19 timing on a fresh save showed OUR string passes are the top stutter source:
// 302 shaders -> CONVERT 1803ms total, 26 big ones eating 1652ms (worst 306ms for a
// single 22KB chunk shader), vs driver COMPILE 638ms / LINK 545ms. The passes are
// deterministic, so cache their product on disk keyed by the ORIGINAL source:
//   files/ngg_convcache/cc_<fnv64>_<len>.bin =
//     [u32 olen][orig][u32 slen][mutated source][u32 clen][converted]
//     [u16 n]{u16,name u16,type u16,value}*n      (uniform-defaults table)
// The stored original is memcmp-verified on load — a false hit is impossible.
// First-ever encounter pays full price and writes; every later encounter (same run or
// months later) restores in ~0ms. LIBGL_NOCONVCACHE=1 disables. Stale entries are
// harmless (key = content); the dir can always be deleted.
#include <sys/stat.h>
int zccache_hits = 0, zccache_miss = 0;
static const char* ZCC_DIR = "/data/data/com.zomdroid/files/ngg_convcache";
static int zcc_enabled(void) {
    // RELEASE SAFETY (2026-07-29): a fresh SIGSEGV (si_addr=0xd, save-load, exactly one
    // cache hit in the run) implicates the hit path. Until that autopsy is done the
    // cache is OPT-IN: set LIBGL_CONVCACHE=1 to enable. Perf work must never gamble a
    // release.
    static int zst = -1;
    if (zst < 0) {
        const char* e = getenv("LIBGL_CONVCACHE");
        zst = (e && e[0] == '1') ? 1 : 0;
        if (zst) mkdir(ZCC_DIR, 0700);
    }
    return zst;
}
static unsigned long long zcc_fnv64(const char* s, size_t n) {
    unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (unsigned char)s[i];
        h *= 1099511628211ULL;
    }
    return h;
}
static void zcc_path(char* out, size_t cap, const char* orig, size_t olen) {
    snprintf(out, cap, "%s/cc_%016llx_%zu.bin", ZCC_DIR, zcc_fnv64(orig, olen), olen);
}
static int zcc_rd(FILE* f, void* p, size_t n) { return fread(p, 1, n, f) == n; }
static char* zcc_rdblk(FILE* f, unsigned* plen) {
    unsigned n;
    if (!zcc_rd(f, &n, 4) || n > 16u * 1024 * 1024) return NULL;
    char* b = (char*)malloc((size_t)n + 1);
    if (!b) return NULL;
    if (n && !zcc_rd(f, b, n)) {
        free(b);
        return NULL;
    }
    b[n] = '\0';
    if (plen) *plen = n;
    return b;
}
// Returns 1 on verified hit (fields of glshader replaced), 0 otherwise.
static int zomdroid_ccache_try(struct shader_s* glshader, const char* orig, size_t olen) {
    if (!zcc_enabled()) return 0;
    char path[512];
    zcc_path(path, sizeof(path), orig, olen);
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int ok = 0;
    char *sorig = NULL, *smut = NULL, *sconv = NULL;
    unsigned lo = 0;
    do {
        sorig = zcc_rdblk(f, &lo);
        if (!sorig || lo != olen || memcmp(sorig, orig, olen) != 0) break;
        smut = zcc_rdblk(f, NULL);
        sconv = zcc_rdblk(f, NULL);
        if (!smut || !sconv) break;
        unsigned short n;
        if (!zcc_rd(f, &n, 2) || n > MAX_UNIFORM_VARIABLE_NUMBER) break;
        int bad = 0;
        for (int i = 0; i < n && !bad; i++) {
            unsigned short l[3];
            char* dst[3] = { glshader->uniforms_declarations[i].variable, glshader->uniforms_declarations[i].type,
                             glshader->uniforms_declarations[i].initial_value };
            unsigned cap3[3] = { MAX_VARIABLE_LENGTH, MAX_UNIFORM_TYPE_LENGTH, MAX_INITIAL_VALUE_LENGTH };
            for (int k = 0; k < 3; k++) {
                if (!zcc_rd(f, &l[k], 2) || l[k] >= cap3[k] || (l[k] && !zcc_rd(f, dst[k], l[k]))) {
                    bad = 1;
                    break;
                }
                dst[k][l[k]] = '\0';
            }
        }
        if (bad) break;
        glshader->uniforms_declarations_count = n;
        free(glshader->source);
        glshader->source = smut;
        smut = NULL;
        free(glshader->converted);
        glshader->converted = sconv;
        sconv = NULL;
        ok = 1;
    } while (0);
    free(sorig);
    free(smut);
    if (!ok) free(sconv);
    fclose(f);
    if (ok) zccache_hits++;
    return ok;
}
static void zomdroid_ccache_store(struct shader_s* glshader, const char* orig, size_t olen) {
    if (!zcc_enabled() || !glshader->converted || !glshader->source) return;
    zccache_miss++;
    char path[512], tmp[520];
    zcc_path(path, sizeof(path), orig, olen);
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f) return;
    unsigned v;
    int ok = 1;
#define ZCC_WB(P, N)                                                                                                   \
    do {                                                                                                               \
        v = (unsigned)(N);                                                                                             \
        ok = ok && fwrite(&v, 1, 4, f) == 4 && (v == 0 || fwrite((P), 1, v, f) == v);                                  \
    } while (0)
    ZCC_WB(orig, olen);
    ZCC_WB(glshader->source, strlen(glshader->source));
    ZCC_WB(glshader->converted, strlen(glshader->converted));
#undef ZCC_WB
    unsigned short n = (unsigned short)glshader->uniforms_declarations_count;
    if (n > MAX_UNIFORM_VARIABLE_NUMBER) n = MAX_UNIFORM_VARIABLE_NUMBER;
    ok = ok && fwrite(&n, 1, 2, f) == 2;
    for (int i = 0; i < n && ok; i++) {
        const char* s3[3] = { glshader->uniforms_declarations[i].variable, glshader->uniforms_declarations[i].type,
                              glshader->uniforms_declarations[i].initial_value };
        for (int k = 0; k < 3 && ok; k++) {
            unsigned short l = (unsigned short)strlen(s3[k]);
            ok = fwrite(&l, 1, 2, f) == 2 && (l == 0 || fwrite(s3[k], 1, l, f) == l);
        }
    }
    fclose(f);
    if (ok) rename(tmp, path);
    else remove(tmp);
}

// ZOMDROID CONVERSION MEMO (2026-09-16): PZ hands the same source to glShaderSource
// several times in one session. The three ~22 KB fragment shaders are converted two or
// three times at 230-690 ms each, and while driving the repeats land mid-game (frames
// 505..9554 in field traces) -- visible freezes. Within one process the SIMPLE path's
// output depends only on the source and the shader type (library, driver and probe
// results cannot change under it), so a repeat restores everything that path writes into
// the shader: the mutated source, the converted text, the builtin needs and the
// uniform-defaults table. Memory only: a disk entry would outlive a library or driver
// update and keep serving the old converter's output. LIBGL_CONVMEMO=0 disables.
typedef struct zmemo_s {
    struct zmemo_s* next;
    GLenum type;
    size_t olen;
    char* orig;
    char* source;
    char* converted;
    shaderconv_need_t need;
    int ndecl;
    uniform_declaration_s* decl;
} zmemo_t;
#define ZMEMO_BUCKETS 256
#define ZMEMO_MAX_BYTES (32L * 1024 * 1024)
static zmemo_t* zmemo_tab[ZMEMO_BUCKETS];
int zmemo_hits = 0, zmemo_entries = 0;
long zmemo_bytes = 0;
static int zmemo_enabled(void) {
    static int zst = -1;
    if (zst < 0) {
        const char* e = getenv("LIBGL_CONVMEMO");
        zst = (e && e[0] == '0') ? 0 : 1;
    }
    return zst;
}
static zmemo_t* zmemo_find(GLenum type, const char* orig, size_t olen, unsigned long long h) {
    for (zmemo_t* m = zmemo_tab[h % ZMEMO_BUCKETS]; m; m = m->next)
        if (m->type == type && m->olen == olen && memcmp(m->orig, orig, olen) == 0) return m;
    return NULL;
}
// Returns 1 when a stored conversion was restored into glshader.
static int zomdroid_memo_try(struct shader_s* glshader, const char* orig, size_t olen) {
    if (!zmemo_enabled()) return 0;
    zmemo_t* m = zmemo_find(glshader->type, orig, olen, zcc_fnv64(orig, olen));
    if (!m) return 0;
    char* src = strdup(m->source);
    char* conv = strdup(m->converted);
    if (!src || !conv) {
        free(src);
        free(conv);
        return 0;
    }
    free(glshader->source);
    glshader->source = src;
    free(glshader->converted);
    glshader->converted = conv;
    glshader->need = m->need;
    if (m->ndecl) memcpy(glshader->uniforms_declarations, m->decl, (size_t)m->ndecl * sizeof(uniform_declaration_s));
    glshader->uniforms_declarations_count = m->ndecl;
    zmemo_hits++;
    return 1;
}
static void zomdroid_memo_store(struct shader_s* glshader, const char* orig, size_t olen) {
    if (!zmemo_enabled() || !glshader->converted || !glshader->source) return;
    unsigned long long h = zcc_fnv64(orig, olen);
    if (zmemo_find(glshader->type, orig, olen, h)) return;
    int n = glshader->uniforms_declarations_count;
    if (n < 0) n = 0;
    if (n > MAX_UNIFORM_VARIABLE_NUMBER) n = MAX_UNIFORM_VARIABLE_NUMBER;
    size_t slen = strlen(glshader->source), clen = strlen(glshader->converted);
    long cost = (long)(sizeof(zmemo_t) + olen + slen + clen + 3 + (size_t)n * sizeof(uniform_declaration_s));
    if (zmemo_bytes + cost > ZMEMO_MAX_BYTES) return; // a full memo just stops learning
    zmemo_t* m = (zmemo_t*)calloc(1, sizeof(zmemo_t));
    if (!m) return;
    m->orig = (char*)malloc(olen + 1);
    m->source = strdup(glshader->source);
    m->converted = strdup(glshader->converted);
    m->decl = n ? (uniform_declaration_s*)malloc((size_t)n * sizeof(uniform_declaration_s)) : NULL;
    if (!m->orig || !m->source || !m->converted || (n && !m->decl)) {
        free(m->orig);
        free(m->source);
        free(m->converted);
        free(m->decl);
        free(m);
        return;
    }
    memcpy(m->orig, orig, olen + 1);
    if (n) memcpy(m->decl, glshader->uniforms_declarations, (size_t)n * sizeof(uniform_declaration_s));
    m->ndecl = n;
    m->type = glshader->type;
    m->olen = olen;
    m->need = glshader->need;
    m->next = zmemo_tab[h % ZMEMO_BUCKETS];
    zmemo_tab[h % ZMEMO_BUCKETS] = m;
    zmemo_bytes += cost;
    zmemo_entries++;
}

// ZOMDROID FIX (Mali strictness, wave 4 — field report 2026-07-26): ivec-TYPED
// EXPRESSIONS in arithmetic with float vectors. PZ's shadow-blur does
//     min((pixel + ivec2(x, y)) * pixelStep, maxUV)
// where pixel is ivec2 and pixelStep is vec2 — legal only under
// GL_EXT_shader_implicit_conversions, which some Mali drivers flatly lack
// ("Extension not supported" -> S0001 ivec2*vec2 -> link failure -> broken shadows).
// Correctness must not depend on the extension: wrap the ivec-side operand in vecN().
// Waves 1-3 covered bare int LITERALS and int MACROS; this covers typed ivec values:
// declared ivecN identifiers, ivecN(...) constructor calls, and parenthesized groups
// with pure-int content — whenever the OTHER operand of + - * / is a declared vecN.

#define ZVEC_MAX_IDS 128
#define ZVEC_ID_LEN 64
typedef struct {
    char name[ZVEC_ID_LEN];
    int n; // component count 2..4
} zvec_id_t;

static int zomdroid_collect_vecn_ids(const char* src, const char* base /*"vec" or "ivec"*/, zvec_id_t* ids) {
    int n = 0;
    size_t blen = strlen(base);
    for (const char* p = src; (p = strstr(p, base)) != NULL; p += blen) {
        if (p > src && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) continue;
        char digit = p[blen];
        if (digit < '2' || digit > '4') continue;
        const char* q = p + blen + 1;
        if (isalnum((unsigned char)*q) || *q == '_') continue; // e.g. vec2ToX
        while (*q == ' ' || *q == '\t') q++;
        if (!(isalpha((unsigned char)*q) || *q == '_')) continue; // constructor call etc.
        const char* s = q;
        while (isalnum((unsigned char)*q) || *q == '_') q++;
        size_t len = (size_t)(q - s);
        const char* r = q;
        while (*r == ' ' || *r == '\t') r++;
        if (*r == '(') continue; // function returning vecN, not a variable
        if (len < ZVEC_ID_LEN && n < ZVEC_MAX_IDS) {
            memcpy(ids[n].name, s, len);
            ids[n].name[len] = '\0';
            ids[n].n = digit - '0';
            n++;
        }
    }
    return n;
}

static int zvec_lookup(const char* tok, size_t len, zvec_id_t* ids, int idn) {
    for (int i = 0; i < idn; i++)
        if (strlen(ids[i].name) == len && !strncmp(tok, ids[i].name, len)) return ids[i].n;
    return 0;
}

// Operand classification within one line. Returns span [*s..*e) and kind:
// 0 unknown, 1 known float-vec (out_n = its N), 2 ivec-ish (id / ivecN(..) / pure-int group).
static int zvec_classify_left(const char* line, int op_i, int* s, int* e, zvec_id_t* vecs, int vn, zvec_id_t* ivecs,
                              int in_, int* out_n) {
    int i = op_i;
    while (i > 0 && (line[i - 1] == ' ' || line[i - 1] == '\t')) i--;
    if (i == 0) return 0;
    *e = i;
    if (line[i - 1] == ')') { // balanced group, maybe a call
        int d = 0, j = i - 1;
        while (j >= 0) {
            if (line[j] == ')') d++;
            else if (line[j] == '(') {
                d--;
                if (!d) break;
            }
            j--;
        }
        if (j < 0) return 0;
        int g = j; // '(' index
        int idend = j;
        while (idend > 0 && (isalnum((unsigned char)line[idend - 1]) || line[idend - 1] == '_')) idend--;
        if (idend < j) { // call name present
            size_t il = (size_t)(j - idend);
            *s = idend;
            if (il > 4 && !strncmp(line + idend, "ivec", 4)) return 2; // ivecN(...) call
            if (zvec_lookup(line + idend, il, vecs, vn)) { *out_n = zvec_lookup(line + idend, il, vecs, vn); return 1; }
            return 0; // some other call — unknown type
        }
        *s = g;
        // group content: ivec-ish if it references ivec ids or ivec ctor and no float hints
        int has_ivec = 0, has_float = 0;
        for (int k = g + 1; k < i - 1; k++) {
            if (line[k] == '.' && isdigit((unsigned char)line[k + 1])) has_float = 1;
            if (!strncmp(line + k, "ivec", 4)) has_ivec = 1;
            if (isalpha((unsigned char)line[k]) || line[k] == '_') {
                int t = k;
                while (isalnum((unsigned char)line[k]) || line[k] == '_') k++;
                if (zvec_lookup(line + t, (size_t)(k - t), ivecs, in_)) has_ivec = 1;
                if (zvec_lookup(line + t, (size_t)(k - t), vecs, vn)) has_float = 1;
                k--;
            }
        }
        if (has_ivec && !has_float) return 2;
        return 0;
    }
    if (isalnum((unsigned char)line[i - 1]) || line[i - 1] == '_') {
        int j = i;
        while (j > 0 && (isalnum((unsigned char)line[j - 1]) || line[j - 1] == '_')) j--;
        if (j > 0 && line[j - 1] == '.') return 0; // swizzle/member — component, not vector
        *s = j;
        size_t il = (size_t)(i - j);
        int nn = zvec_lookup(line + j, il, vecs, vn);
        if (nn) { *out_n = nn; return 1; }
        if (zvec_lookup(line + j, il, ivecs, in_)) return 2;
        return 0;
    }
    return 0;
}

static int zvec_classify_right(const char* line, size_t len, int op_i, int* s, int* e, zvec_id_t* vecs, int vn,
                               zvec_id_t* ivecs, int in_, int* out_n) {
    int i = op_i + 1;
    while (i < (int)len && (line[i] == ' ' || line[i] == '\t')) i++;
    if (i >= (int)len) return 0;
    *s = i;
    if (isalpha((unsigned char)line[i]) || line[i] == '_') {
        int j = i;
        while (j < (int)len && (isalnum((unsigned char)line[j]) || line[j] == '_')) j++;
        int k = j;
        while (k < (int)len && (line[k] == ' ' || line[k] == '\t')) k++;
        if (k < (int)len && line[k] == '(') { // call
            int d = 0;
            int m = k;
            while (m < (int)len) {
                if (line[m] == '(') d++;
                else if (line[m] == ')') {
                    d--;
                    if (!d) break;
                }
                m++;
            }
            if (m >= (int)len) return 0;
            *e = m + 1;
            if (!strncmp(line + i, "ivec", 4)) return 2;
            int nn = zvec_lookup(line + i, (size_t)(j - i), vecs, vn);
            if (nn) { *out_n = nn; return 1; }
            return 0;
        }
        if (j < (int)len && line[j] == '.') return 0; // swizzled — component access
        *e = j;
        int nn = zvec_lookup(line + i, (size_t)(j - i), vecs, vn);
        if (nn) { *out_n = nn; return 1; }
        if (zvec_lookup(line + i, (size_t)(j - i), ivecs, in_)) return 2;
        return 0;
    }
    if (line[i] == '(') {
        int d = 0, m = i;
        while (m < (int)len) {
            if (line[m] == '(') d++;
            else if (line[m] == ')') {
                d--;
                if (!d) break;
            }
            m++;
        }
        if (m >= (int)len) return 0;
        *e = m + 1;
        int has_ivec = 0, has_float = 0;
        for (int k = i + 1; k < m; k++) {
            if (line[k] == '.' && isdigit((unsigned char)line[k + 1])) has_float = 1;
            if (!strncmp(line + k, "ivec", 4)) has_ivec = 1;
            if (isalpha((unsigned char)line[k]) || line[k] == '_') {
                int t = k;
                while (isalnum((unsigned char)line[k]) || line[k] == '_') k++;
                if (zvec_lookup(line + t, (size_t)(k - t), ivecs, in_)) has_ivec = 1;
                if (zvec_lookup(line + t, (size_t)(k - t), vecs, vn)) has_float = 1;
                k--;
            }
        }
        if (has_ivec && !has_float) return 2;
        return 0;
    }
    return 0;
}

// One in-place rewrite per call; returns 1 if a wrap was applied to this line buffer.
static int zvec_wrap_once(char* line, size_t* plen, size_t cap, zvec_id_t* vecs, int vn, zvec_id_t* ivecs, int in_) {
    size_t len = *plen;
    for (int i = 0; i < (int)len; i++) {
        char c = line[i];
        if (c == '/' && i + 1 < (int)len && (line[i + 1] == '/' || line[i + 1] == '*')) break; // comment
        if (c != '*' && c != '/' && c != '+' && c != '-') continue;
        if (c == '-') { // binary only: previous non-space char must be an operand ending
            int pb = i - 1;
            while (pb >= 0 && (line[pb] == ' ' || line[pb] == '\t')) pb--;
            if (pb < 0 || strchr("=,(+-*/<>&|?:", line[pb])) continue; // unary minus
        }
        int ls, le, rs, re, lu = 0, ru = 0;
        int lk = zvec_classify_left(line, i, &ls, &le, vecs, vn, ivecs, in_, &lu);
        int rk = zvec_classify_right(line, len, i, &rs, &re, vecs, vn, ivecs, in_, &ru);
        int wrap_s = -1, wrap_e = -1, nn = 0;
        if (lk == 1 && rk == 2) { wrap_s = rs; wrap_e = re; nn = lu; }
        else if (lk == 2 && rk == 1) { wrap_s = ls; wrap_e = le; nn = ru; }
        if (wrap_s < 0) continue;
        char head[8];
        int hl = snprintf(head, sizeof(head), "vec%d(", nn);
        if (len + hl + 1 + 1 > cap) return 0;
        memmove(line + wrap_e + hl + 1, line + wrap_e, len - wrap_e + 1);
        line[wrap_e + hl] = ')';
        memmove(line + wrap_s + hl, line + wrap_s, (size_t)(wrap_e - wrap_s));
        memcpy(line + wrap_s, head, hl);
        *plen = len + hl + 1;
        return 1;
    }
    return 0;
}

static char* zomdroid_vecify_ivec_arith(char* src) {
    if (!src || !strstr(src, "ivec")) return src;
    zvec_id_t* vecs = (zvec_id_t*)malloc(sizeof(zvec_id_t) * ZVEC_MAX_IDS);
    zvec_id_t* ivecs = (zvec_id_t*)malloc(sizeof(zvec_id_t) * ZVEC_MAX_IDS);
    if (!vecs || !ivecs) {
        free(vecs);
        free(ivecs);
        return src;
    }
    int vn = zomdroid_collect_vecn_ids(src, "vec", vecs);
    int in_ = zomdroid_collect_vecn_ids(src, "ivec", ivecs);
    if (!vn || !in_) {
        free(vecs);
        free(ivecs);
        return src;
    }
    size_t slen = strlen(src);
    char* out = (char*)malloc(slen * 2 + 64);
    if (!out) {
        free(vecs);
        free(ivecs);
        return src;
    }
    char lbuf[2048];
    char* o = out;
    const char* p = src;
    while (*p) {
        const char* nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) + 1 : strlen(p);
        int touchable = len < sizeof(lbuf) - 64 && !memchr(p, '#', len) && !memchr(p, '[', len);
        if (touchable && (strstr(p, "ivec") == NULL || (size_t)(strstr(p, "ivec") - p) >= len)) {
            // fast path: line has no ivec ctor; still may reference an ivec-declared id
            int any = 0;
            for (int i = 0; i < in_ && !any; i++)
                if (strstr(p, ivecs[i].name) && (size_t)(strstr(p, ivecs[i].name) - p) < len) any = 1;
            if (!any) touchable = 0;
        }
        if (touchable) {
            memcpy(lbuf, p, len);
            lbuf[len] = '\0';
            size_t ll = len;
            int guard = 0;
            while (guard++ < 8 && zvec_wrap_once(lbuf, &ll, sizeof(lbuf), vecs, vn, ivecs, in_)) {}
            memcpy(o, lbuf, ll);
            o += ll;
        } else {
            memcpy(o, p, len);
            o += len;
        }
        p += len;
    }
    *o = '\0';
    free(vecs);
    free(ivecs);
    free(src);
    return out;
}

// ZOMDROID FIX (Mali strictness, wave 2): Mali rejects EVERY implicit int<->float mix
// and ignores GL_EXT_shader_implicit_conversions (reported "not supported"). The RC2
// passes covered const-float initializers and builtin args; the 28 shaders failing on
// the Mali tester used two more shapes: mixed ARITHMETIC/COMPARISONS with float locals
// ("(clip*2)-1", "if (d > 0)", "1 - pow(...)") and a sampler named "texture" (a builtin
// function name in GLSL 3.x -> "Symbol 'texture' redeclared").

#define ZFLT_MAX_IDS 256
#define ZFLT_ID_LEN 64

// Collect identifiers declared float in this source ("float x", "uniform float x",
// "const highp float x, y;"). Function names ("float foo(") are skipped.
static int zomdroid_collect_float_ids(const char* src, char ids[ZFLT_MAX_IDS][ZFLT_ID_LEN]) {
    int n = 0;
    for (const char* p = src; (p = strstr(p, "float")) != NULL; p += 5) {
        if ((p > src && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) ||
            isalnum((unsigned char)p[5]) || p[5] == '_')
            continue;
        const char* q = p + 5;
        for (;;) {
            while (*q == ' ' || *q == '\t') q++;
            if (!(isalpha((unsigned char)*q) || *q == '_')) break;
            const char* s = q;
            while (isalnum((unsigned char)*q) || *q == '_') q++;
            size_t len = (size_t)(q - s);
            const char* r = q;
            while (*r == ' ' || *r == '\t') r++;
            if (*r == '(') break; // function declaration, not a variable
            if (len < ZFLT_ID_LEN && n < ZFLT_MAX_IDS) {
                memcpy(ids[n], s, len);
                ids[n][len] = '\0';
                n++;
            }
            // multi-declarations: "float a, b = 0, c;"
            while (*r && *r != ',' && *r != ';' && *r != ')' && *r != '\n') r++;
            if (*r != ',') break;
            q = r + 1;
        }
    }
    return n;
}

// Collect object-like macros with a bare integer body ("#define BLUR_RANGE 2"). Their
// uses inside float arithmetic trip Mali exactly like literal ints do, but the define
// itself must stay integer (loop bounds / array sizes) — so uses get float()-wrapped.
static int zomdroid_collect_int_macros(const char* src, char ids[ZFLT_MAX_IDS][ZFLT_ID_LEN]) {
    int n = 0;
    for (const char* p = src; (p = strstr(p, "#define")) != NULL; p += 7) {
        const char* q = p + 7;
        while (*q == ' ' || *q == '\t') q++;
        const char* s = q;
        while (isalnum((unsigned char)*q) || *q == '_') q++;
        size_t len = (size_t)(q - s);
        if (!len || len >= ZFLT_ID_LEN) continue;
        if (*q == '(') continue; // function-like macro
        while (*q == ' ' || *q == '\t') q++;
        const char* v = q;
        while (isdigit((unsigned char)*q)) q++;
        if (q == v) continue; // body doesn't start with a number
        while (*q == ' ' || *q == '\t' || *q == '\r') q++;
        if (*q != '\n' && *q != '\0') continue; // body is an expression / float — skip
        if (n < ZFLT_MAX_IDS) {
            memcpy(ids[n], s, len);
            ids[n][len] = '\0';
            n++;
        }
    }
    return n;
}

// ZOMDROID FIX (Mali/ANGLE wave 5): integer UNIFORMS used in float arithmetic. PZ's
// fade shader does "alpha * (BASE_OPACITY * opacityScale)" where opacityScale is
// "uniform int" — strict ES compilers reject 'const float' * 'uniform highp int'.
// The declaration must stay integer (the game sets it with glUniform1i), so uses get
// float()-wrapped exactly like int-valued macros; both lists share the wrap below.
static int zomdroid_collect_int_uniforms(const char* src, char ids[ZFLT_MAX_IDS][ZFLT_ID_LEN], int n) {
    for (const char* p = src; (p = strstr(p, "uniform")) != NULL; p += 7) {
        if (p > src && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) continue;
        const char* q = p + 7;
        for (;;) {
            while (*q == ' ' || *q == '\t')
                q++;
            if (!strncmp(q, "highp", 5) || !strncmp(q, "lowp", 4)) {
                q += (*q == 'h') ? 5 : 4;
                continue;
            }
            if (!strncmp(q, "mediump", 7)) {
                q += 7;
                continue;
            }
            break;
        }
        if (strncmp(q, "int", 3) || isalnum((unsigned char)q[3]) || q[3] == '_') continue;
        q += 3;
        while (*q == ' ' || *q == '\t')
            q++;
        const char* s = q;
        while (isalnum((unsigned char)*q) || *q == '_')
            q++;
        size_t len = (size_t)(q - s);
        while (*q == ' ' || *q == '\t')
            q++;
        if (*q == '[') continue; // arrays are indexed with ints — leave them alone
        if (len && len < ZFLT_ID_LEN && n < ZFLT_MAX_IDS) {
            memcpy(ids[n], s, len);
            ids[n][len] = '\0';
            n++;
        }
    }
    return n;
}

static int zomdroid_line_has_float_id(const char* line, size_t len, char ids[ZFLT_MAX_IDS][ZFLT_ID_LEN], int idn) {
    for (int i = 0; i < idn; i++) {
        size_t il = strlen(ids[i]);
        if (il == 0 || il > len) continue;
        for (const char* p = line; (p = strstr(p, ids[i])) != NULL && p < line + len; p += il) {
            if (p >= line + len) break;
            if ((p > line && (isalnum((unsigned char)p[-1]) || p[-1] == '_' || p[-1] == '.')) ||
                (isalnum((unsigned char)p[il]) || p[il] == '_'))
                continue;
            return 1;
        }
    }
    // vector component access (.a/.rgb/.xyz...) is float-typed in every PZ shader
    for (const char* p = line; p < line + len - 1; p++) {
        if (*p != '.') continue;
        if (p == line || !(isalnum((unsigned char)p[-1]) || p[-1] == '_' || p[-1] == ')')) continue;
        const char* q = p + 1;
        while (q < line + len && strchr("rgbaxyzwstpq", *q)) q++;
        if (q > p + 1 && (q >= line + len || !(isalnum((unsigned char)*q) || *q == '_' || *q == '.')))
            return 1;
    }
    return 0;
}

// Floatify bare int literals on lines that visibly work with float data.
// Guards: 'for(' headers, 'int'/'ivec'/'uvec' declarations, bit/mod operators,
// preprocessor lines, [] indexing, '//' comments.
static char* zomdroid_floatify_float_context_lines(char* src) {
    if (!src) return src;
    char(*ids)[ZFLT_ID_LEN] = (char(*)[ZFLT_ID_LEN])malloc(ZFLT_MAX_IDS * ZFLT_ID_LEN);
    char(*macs)[ZFLT_ID_LEN] = (char(*)[ZFLT_ID_LEN])malloc(ZFLT_MAX_IDS * ZFLT_ID_LEN);
    if (!ids || !macs) {
        free(ids);
        free(macs);
        return src;
    }
    int idn = zomdroid_collect_float_ids(src, ids);
    int macn = zomdroid_collect_int_macros(src, macs);
    // int uniforms share the float()-wrapping list with int macros (same problem)
    macn = zomdroid_collect_int_uniforms(src, macs, macn);
    size_t slen = strlen(src);
    char* out = (char*)malloc(slen * 3 + 64);
    if (!out) {
        free(ids);
        free(macs);
        return src;
    }
    char* o = out;
    const char* p = src;
    while (*p) {
        const char* nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) + 1 : strlen(p);
        int touch = 1;
        if (memchr(p, '#', len) || strncmp(p + strspn(p, " \t"), "for", 3) == 0)
            touch = 0;
        if (touch) {
            static const char* bad[] = {"int ", "int\t", "ivec", "uvec", "<<", ">>", " % ", " & ", " | "};
            for (unsigned b = 0; touch && b < sizeof(bad) / sizeof(bad[0]); b++) {
                const char* f = p;
                while ((f = strstr(f, bad[b])) != NULL && f < p + len) {
                    touch = 0;
                    break;
                }
            }
        }
        if (touch && !zomdroid_line_has_float_id(p, len, ids, idn))
            touch = 0;
        if (!touch) {
            memcpy(o, p, len);
            o += len;
            p += len;
            continue;
        }
        int bracket = 0, comment = 0;
        for (size_t i = 0; i < len;) {
            char c = p[i];
            if (!comment && c == '/' && i + 1 < len && p[i + 1] == '/') comment = 1;
            if (c == '[') bracket++;
            else if (c == ']' && bracket > 0) bracket--;
            // int-valued macro used in float context -> float(NAME) (definition untouched)
            if (!comment && !bracket && (isalpha((unsigned char)c) || c == '_') &&
                !(i > 0 && (isalnum((unsigned char)p[i - 1]) || p[i - 1] == '_' || p[i - 1] == '.'))) {
                size_t j = i;
                while (j < len && (isalnum((unsigned char)p[j]) || p[j] == '_'))
                    j++;
                size_t il = j - i;
                // builtins with MANDATORY int args: copy the whole call verbatim so the
                // LOD/coords never get floatified ("textureSize(tex, 0)" stays int).
                if ((il == 11 && !strncmp(p + i, "textureSize", 11)) ||
                    (il == 10 && !strncmp(p + i, "texelFetch", 10))) {
                    size_t k = j;
                    while (k < len && (p[k] == ' ' || p[k] == '\t')) k++;
                    if (k < len && p[k] == '(') {
                        int d2 = 0;
                        size_t e2 = k;
                        while (e2 < len) {
                            if (p[e2] == '(') d2++;
                            else if (p[e2] == ')') {
                                d2--;
                                if (!d2) break;
                            }
                            e2++;
                        }
                        if (e2 < len) {
                            memcpy(o, p + i, e2 - i + 1);
                            o += e2 - i + 1;
                            i = e2 + 1;
                            continue;
                        }
                    }
                }
                int mi = -1;
                if (macn)
                    for (int m = 0; m < macn; m++)
                        if (strlen(macs[m]) == il && !strncmp(p + i, macs[m], il)) {
                            mi = m;
                            break;
                        }
                if (mi >= 0) {
                    memcpy(o, "float(", 6);
                    o += 6;
                    memcpy(o, p + i, il);
                    o += il;
                    *o++ = ')';
                    i = j;
                    continue;
                }
                memcpy(o, p + i, il);
                o += il;
                i = j;
                continue;
            }
            if (!comment && !bracket && isdigit((unsigned char)c) &&
                !(i > 0 && (isalnum((unsigned char)p[i - 1]) || p[i - 1] == '_' || p[i - 1] == '.')) &&
                !(i >= 2 && (p[i - 1] == '-' || p[i - 1] == '+') && (p[i - 2] == 'e' || p[i - 2] == 'E'))) {
                size_t j = i;
                while (j < len && isdigit((unsigned char)p[j]))
                    j++;
                int isint = !(j < len && (p[j] == '.' || p[j] == 'x' || p[j] == 'X' || p[j] == 'e' || p[j] == 'E' ||
                                          p[j] == 'u' || p[j] == 'U' || isalpha((unsigned char)p[j]) || p[j] == '_'));
                memcpy(o, p + i, j - i);
                o += j - i;
                if (isint) {
                    *o++ = '.';
                    *o++ = '0';
                }
                i = j;
                continue;
            }
            *o++ = c;
            i++;
        }
        p += len;
    }
    *o = '\0';
    free(ids);
    free(macs);
    free(src);
    return out;
}

// ZOMDROID FIX (Mali/ANGLE wave 5, field report 2026-07-31): GLSL ES demands that
// global initializers be constant expressions. PZ's puddles/water shaders initialize
// globals straight from a uniform —
//     float muddyPuddles = PuddlesParams[1].x;
// — which desktop GLSL 1.20 accepts and every strict ES compiler rejects ("global
// variable initializers must be constant expressions"). Nine shaders died on exactly
// this on a Dimensity/ANGLE device. Split them: the declaration stays at file scope,
// the initializer moves to the top of main() in source order, so dependency chains
// between such globals keep working. Only declarations ABOVE main() are touched.
static int zomdroid_rhs_is_constant(const char* s, size_t len) {
    // constant = literals and type constructors only; any other identifier (uniform,
    // varying, another hoisted global) makes it a runtime value
    for (size_t i = 0; i < len; i++) {
        if (!(isalpha((unsigned char)s[i]) || s[i] == '_')) continue;
        if (i > 0 && (isalnum((unsigned char)s[i - 1]) || s[i - 1] == '_' || s[i - 1] == '.')) continue;
        size_t j = i;
        while (j < len && (isalnum((unsigned char)s[j]) || s[j] == '_'))
            j++;
        size_t il = j - i;
        // a float suffix / exponent is part of a number, not an identifier
        if (!(i > 0 && isdigit((unsigned char)s[i - 1]))) {
            // constructors plus the pure builtins GLSL itself folds at compile time —
            // "vec3 v = normalize(vec3(...))" is a legal constant initializer, no need
            // to move it (only its ARGUMENTS decide, and they are scanned here too)
            static const char* ctor[] = { "float",  "int",     "uint",  "bool",  "vec2",   "vec3",
                                          "vec4",   "ivec2",   "ivec3", "ivec4", "uvec2",  "uvec3",
                                          "uvec4",  "mat2",    "mat3",  "mat4",  "true",   "false",
                                          "normalize", "radians", "degrees", "sqrt", "inversesqrt",
                                          "abs",    "sign",    "floor", "ceil",  "fract",  "min",
                                          "max",    "clamp",   "mix",   "step",  "pow",    "exp",
                                          "log",    "exp2",    "log2",  "sin",   "cos",    "tan",
                                          "length", "dot",     "cross" };
            int isctor = 0;
            for (unsigned c = 0; c < sizeof(ctor) / sizeof(ctor[0]); c++)
                if (strlen(ctor[c]) == il && !strncmp(s + i, ctor[c], il)) {
                    isctor = 1;
                    break;
                }
            if (!isctor) return 0;
        }
        i = j - 1;
    }
    return 1;
}
// main() DEFINITION (PZ prototypes it too, and in the puddles shaders main is defined
// ABOVE the globals and even above the uniforms — so the initializers cannot simply be
// moved to the top of main; they go into an init function placed next to the
// declarations, which main calls through a prototype in the preamble).
static const char* zomdroid_find_main_brace(const char* src) {
    const char* p = src;
    while ((p = strstr(p, "void main")) != NULL) {
        const char* par = strchr(p, '(');
        const char* brace = strchr(p, '{');
        const char* semi = strchr(p, ';');
        if (par && brace && (!semi || semi > brace)) return brace;
        p += 9;
    }
    return NULL;
}
// end of the leading #version / #extension / precision-free preprocessor block: the
// first spot where a declaration may legally appear
static size_t zomdroid_preamble_end(const char* src) {
    size_t off = 0;
    const char* p = src;
    while (*p) {
        const char* nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) + 1 : strlen(p);
        const char* q = p + strspn(p, " \t\r");
        if (*q == '\0' || *q == '\n' || *q == '\r' || *q == '#' || (q[0] == '/' && q[1] == '/')) {
            off += len;
            p += len;
            continue;
        }
        break;
    }
    return off;
}
// ZOMDROID (Mali-G615, field 2026-08-19): a driver may ADVERTISE
// GL_EXT_shader_non_constant_global_initializers in GL_EXTENSIONS and then have its
// compiler reject exactly what that extension permits. On that device every vehicle
// program (SphereMap + PuddlesParams, puddle math is inlined into the car shaders)
// failed to link, because the gate below stood the hoist down on the strength of the
// advertisement alone. So ask the compiler instead of the extension string: build the
// smallest shader the extension exists for and see whether it survives. Once per process.
static int zomdroid_nonconstinit_honored(const char* src) {
    static int zverdict = -1;
    if (zverdict >= 0) return zverdict;
    // Probe in the same shading language as the shader that raised the question: the
    // rejection is version specific (PZ compiles "#version 320 es"), and a probe written
    // in another version would answer a question nobody asked. Without a version line
    // there is nothing to match, so keep the old behavior and do not cache a verdict.
    char ver[64];
    ver[0] = '\0';
    if (src) {
        const char* v = strstr(src, "#version");
        if (v && (v == src || v[-1] == '\n')) {
            const char* e = strchr(v, '\n');
            size_t n = e ? (size_t)(e - v) : strlen(v);
            if (n < sizeof(ver) - 2) {
                memcpy(ver, v, n);
                ver[n] = '\n';
                ver[n + 1] = '\0';
            }
        }
    }
    if (!ver[0]) return 1;
    LOAD_GLES2(glCreateShader);
    LOAD_GLES2(glShaderSource);
    LOAD_GLES2(glCompileShader);
    LOAD_GLES2(glGetShaderiv);
    LOAD_GLES2(glDeleteShader);
    if (!gles_glCreateShader || !gles_glShaderSource || !gles_glCompileShader || !gles_glGetShaderiv ||
        !gles_glDeleteShader)
        return 1;
    // a VERTEX shader: gl_Position exists in every ES shading language version, so the
    // probe needs no output declaration and stays valid from ESSL 1.00 through 3.20.
    char probe[512];
    snprintf(probe, sizeof(probe),
             "%s#extension GL_EXT_shader_non_constant_global_initializers : enable\n"
             "uniform float zprobe_u;\n"
             "float zprobe_g = zprobe_u;\n"
             "void main() { gl_Position = vec4(zprobe_g, 0.0, 0.0, 1.0); }\n",
             ver);
    GLuint zs = gles_glCreateShader(GL_VERTEX_SHADER);
    if (!zs) return 1;
    const char* zsrc = probe;
    GLint zlen = (GLint)strlen(probe);
    gles_glShaderSource(zs, 1, &zsrc, &zlen);
    gles_glCompileShader(zs);
    GLint zst = 0;
    gles_glGetShaderiv(zs, GL_COMPILE_STATUS, &zst);
    gles_glDeleteShader(zs);
    zverdict = zst ? 1 : 0;
    {
        extern void zomdroid_gltrace(const char* fmt, ...);
        zomdroid_gltrace("NONCONSTINIT probe: extension advertised, compiler %s",
                         zverdict ? "honors it (shaders left untouched)"
                                  : "REJECTS it -> hoisting initializers");
    }
    return zverdict;
}
static char* zomdroid_hoist_global_initializers(char* src) {
    if (!src) return src;
    // ZOMDROID FIX (black farming plots, field 2026-08-04): this rewrite exists ONLY
    // for drivers that lack GL_EXT_shader_non_constant_global_initializers (ANGLE).
    // Where the driver honors it (Adreno, most vendor Malis), the untouched shader is
    // exactly what every pre-wave5 build shipped — and rewriting anyway proved able to
    // shift the puddles blend math (wet farming plots went black with puddles ON).
    // Prefer the driver; hoist only when there is no other way to compile at all.
    if (hardext.nonconstinit && zomdroid_nonconstinit_honored(src)) return src;
    if (!zomdroid_find_main_brace(src)) return src;
    size_t slen = strlen(src);
    char* out = (char*)malloc(slen * 2 + 4096);
    char* hoist = (char*)malloc(slen + 1024);
    if (!out || !hoist) {
        free(out);
        free(hoist);
        return src;
    }
    char* o = out;
    char* h = hoist;
    int nh = 0, depth = 0, blockcomment = 0, ppdepth = 0;
    size_t last_decl_end = 0;
    const char* p = src;
    while (*p) {
        const char* nl = strchr(p, '\n');
        size_t len = nl ? (size_t)(nl - p) + 1 : strlen(p);
        int handled = 0;
        size_t consume = len; // grows past the line only for a HOISTED multi-line init
        // Conditional blocks are off limits (learned on PZ's skybox shader): the same
        // global is declared once per #ifdef branch, so hoisting would merge day and
        // night values into one function — and that function would land INSIDE the
        // branch, vanishing with it while the call in main() remained.
        {
            const char* d = p + strspn(p, " \t");
            if (*d == '#') {
                const char* w = d + 1 + strspn(d + 1, " \t");
                if (!strncmp(w, "if", 2)) ppdepth++;
                else if (!strncmp(w, "endif", 5) && ppdepth > 0) ppdepth--;
            }
        }
        if (!blockcomment && depth == 0 && ppdepth == 0) {
            const char* q = p + strspn(p, " \t");
            static const char* skipw[] = { "const",     "uniform", "attribute", "varying", "in",
                                           "out",       "layout",  "struct",    "precision", "flat",
                                           "invariant", "#",       "//",        "/*" };
            int skip = 0;
            for (unsigned s = 0; !skip && s < sizeof(skipw) / sizeof(skipw[0]); s++) {
                size_t wl = strlen(skipw[s]);
                if (!strncmp(q, skipw[s], wl) && !(isalnum((unsigned char)q[wl]) || q[wl] == '_')) skip = 1;
            }
            if (!skip) {
                const char* t = q;
                while (isalnum((unsigned char)*t) || *t == '_')
                    t++;
                int tlen = (int)(t - q);
                // "mat2/3/4" are valid here too but not in the shared type-word helper
                int istype = zomdroid_is_type_word(q, tlen) ||
                             (tlen == 4 && !strncmp(q, "mat", 3) && q[3] >= '2' && q[3] <= '4');
                if (istype) {
                    const char* n = t + strspn(t, " \t");
                    const char* ne = n;
                    while (isalnum((unsigned char)*ne) || *ne == '_')
                        ne++;
                    const char* e = ne + strspn(ne, " \t");
                    if (ne > n && *e == '=' && e[1] != '=') {
                        const char* rhs = e + 1 + strspn(e + 1, " \t");
                        const char* semi = (const char*)memchr(rhs, ';', len - (size_t)(rhs - p));
                        // Multi-line initializer (PZ's water shaders wrap them): scan
                        // ahead for the ';', refusing to cross a block or directive.
                        // Skipping these caused MIXED semantics — hoisted globals were
                        // still zero while a leftover global initializer read them.
                        // A hoist must be all-or-nothing to preserve evaluation order.
                        if (!semi && !memchr(rhs, '{', len - (size_t)(rhs - p))) {
                            const char* lim = rhs + 2048;
                            for (const char* c = p + len; *c && c < lim; c++) {
                                if (*c == '{' || *c == '}' || *c == '#') break;
                                if (*c == ';') {
                                    semi = c;
                                    break;
                                }
                            }
                        }
                        size_t mlspan = 0;
                        if (semi && semi >= p + len) { // initializer continues past this line
                            const char* nl2 = strchr(semi, '\n');
                            mlspan = nl2 ? (size_t)(nl2 - p) + 1 : strlen(p);
                        }
                        // single declarator only: a ',' outside parens means "float a=1, b;"
                        int multi = 0, par = 0;
                        for (const char* c = rhs; semi && c < semi; c++) {
                            if (*c == '(') par++;
                            else if (*c == ')') par--;
                            else if (*c == ',' && !par) multi = 1;
                        }
                        if (semi && !multi && nh < 64 &&
                            !zomdroid_rhs_is_constant(rhs, (size_t)(semi - rhs))) {
                            // declaration without the initializer
                            memcpy(o, q, (size_t)(ne - q));
                            o += ne - q;
                            *o++ = ';';
                            *o++ = '\n';
                            last_decl_end = (size_t)(o - out); // init function goes here
                            // assignment, evaluated later inside the init function
                            *h++ = ' ';
                            *h++ = ' ';
                            *h++ = ' ';
                            *h++ = ' ';
                            memcpy(h, n, (size_t)(semi - n) + 1);
                            h += semi - n + 1;
                            *h++ = '\n';
                            nh++;
                            handled = 1;
                            if (mlspan) consume = mlspan; // swallow the whole span
                        }
                    }
                }
            }
        }
        if (!handled) {
            memcpy(o, p, len);
            o += len;
        }
        // brace depth / comment state for everything we just consumed
        for (size_t i = 0; i < consume; i++) {
            if (blockcomment) {
                if (p[i] == '*' && i + 1 < consume && p[i + 1] == '/') {
                    blockcomment = 0;
                    i++;
                }
                continue;
            }
            if (p[i] == '/' && i + 1 < consume && p[i + 1] == '/') {
                // line comment: skip to that line's end, keep tracking the rest of span
                const char* cnl = (const char*)memchr(p + i, '\n', consume - i);
                if (!cnl) break;
                i = (size_t)(cnl - p);
                continue;
            }
            if (p[i] == '/' && i + 1 < consume && p[i + 1] == '*') {
                blockcomment = 1;
                i++;
                continue;
            }
            if (p[i] == '{') depth++;
            else if (p[i] == '}' && depth > 0) depth--;
        }
        p += consume;
    }
    *o = '\0';
    *h = '\0';
    if (!nh) {
        free(out);
        free(hoist);
        return src;
    }
    // Three splices, applied in offset order (main may sit ABOVE the declarations, so
    // the call can precede the definition — that is what the prototype is for):
    //   preamble  -> "void zomdroid_init_globals();"
    //   last decl -> the definition carrying every hoisted initializer, in source order
    //   main '{'  -> the call
    const char* mb = zomdroid_find_main_brace(out);
    if (!mb) {
        free(out);
        free(hoist);
        return src;
    }
    static const char* zproto = "void zomdroid_init_globals();\n";
    static const char* zcall = "\n  zomdroid_init_globals();\n";
    size_t hl = strlen(hoist), ol = strlen(out);
    size_t off[3] = { zomdroid_preamble_end(out), last_decl_end, (size_t)(mb - out) + 1 };
    char* def = (char*)malloc(hl + 64);
    if (!def) {
        free(out);
        free(hoist);
        return src;
    }
    strcpy(def, "void zomdroid_init_globals(){\n");
    strcat(def, hoist);
    strcat(def, "}\n");
    const char* txt[3] = { zproto, def, zcall };
    for (int a = 0; a < 2; a++) // sort 3 insertions by offset (stable, tiny)
        for (int b = 0; b < 2 - a; b++)
            if (off[b] > off[b + 1]) {
                size_t to = off[b];
                const char* tt = txt[b];
                off[b] = off[b + 1];
                txt[b] = txt[b + 1];
                off[b + 1] = to;
                txt[b + 1] = tt;
            }
    char* fin = (char*)malloc(ol + hl + 256);
    if (!fin) {
        free(def);
        free(out);
        free(hoist);
        return src;
    }
    char* f = fin;
    size_t prev = 0;
    for (int a = 0; a < 3; a++) {
        memcpy(f, out + prev, off[a] - prev);
        f += off[a] - prev;
        size_t tl = strlen(txt[a]);
        memcpy(f, txt[a], tl);
        f += tl;
        prev = off[a];
    }
    memcpy(f, out + prev, ol - prev + 1);
    free(def);
    free(out);
    free(hoist);
    free(src);
    zomdroid_gltrace("HOIST %d global initializer(s) -> zomdroid_init_globals()", nh);
    return fin;
}

// ZOMDROID FIX (Mali wave 3): textureSize() returns ivec2; PZ's visibility shader does
// "texSize / textureSize(tex, 0)" — vec2/ivec2 arithmetic that Mali rejects. Wrap the
// WHOLE call in vec2() when it is an operand of +-*/ (the int LOD argument inside must
// stay int); lines mentioning ivec keep integer semantics and are left alone.
static char* zomdroid_wrap_texturesize_arith(char* src) {
    if (!src) return src;
    int n = 0;
    for (const char* p = src; (p = strstr(p, "textureSize")) != NULL; p += 11)
        n++;
    if (!n) return src;
    char* out = (char*)malloc(strlen(src) + n * 8 + 1);
    if (!out) return src;
    const char* in = src;
    char* o = out;
    while (*in) {
        if (*in == 't' && !strncmp(in, "textureSize", 11) &&
            !(in > src && (isalnum((unsigned char)in[-1]) || in[-1] == '_' || in[-1] == '.')) &&
            !(isalnum((unsigned char)in[11]) || in[11] == '_')) {
            const char* q = in + 11;
            while (*q == ' ' || *q == '\t') q++;
            if (*q == '(') {
                // full call span
                int depth = 0;
                const char* e = q;
                while (*e) {
                    if (*e == '(') depth++;
                    else if (*e == ')') {
                        depth--;
                        if (!depth) break;
                    }
                    e++;
                }
                if (*e == ')') {
                    // line must not carry ivec semantics
                    const char* ls = in;
                    while (ls > src && ls[-1] != '\n') ls--;
                    const char* le = e;
                    while (*le && *le != '\n') le++;
                    int has_ivec = 0;
                    for (const char* f = ls; f < le - 3; f++)
                        if (!strncmp(f, "ivec", 4)) {
                            has_ivec = 1;
                            break;
                        }
                    // arithmetic operand: op right before the token or right after ')'
                    const char* b = in;
                    while (b > ls && (b[-1] == ' ' || b[-1] == '\t')) b--;
                    const char* a = e + 1;
                    while (*a == ' ' || *a == '\t') a++;
                    int arith = (b > ls && (b[-1] == '+' || b[-1] == '-' || b[-1] == '*' || b[-1] == '/')) ||
                                (*a == '+' || *a == '-' || *a == '*' || *a == '/');
                    if (!has_ivec && arith) {
                        memcpy(o, "vec2(", 5);
                        o += 5;
                        memcpy(o, in, (size_t)(e - in) + 1);
                        o += e - in + 1;
                        *o++ = ')';
                        in = e + 1;
                        continue;
                    }
                }
            }
        }
        *o++ = *in++;
    }
    *o = '\0';
    free(src);
    return out;
}

// Rename a sampler declared with the reserved-in-GLSL3 name "texture" (declaration and
// argument uses; calls like "texture2D(" / converted "texture(" keep their name).
static char* zomdroid_rename_reserved_samplers(char* src) {
    if (!src) return src;
    if (!strstr(src, "sampler2D texture")) return src;
    int hits = 0;
    for (const char* p = src; (p = strstr(p, "texture")) != NULL; p += 7) {
        if ((p > src && (isalnum((unsigned char)p[-1]) || p[-1] == '_' || p[-1] == '.')) ||
            isalnum((unsigned char)p[7]) || p[7] == '_')
            continue;
        const char* q = p + 7;
        while (*q == ' ' || *q == '\t') q++;
        if (*q == '(') continue;
        hits++;
    }
    if (!hits) return src;
    char* out = (char*)malloc(strlen(src) + hits * 8 + 1);
    if (!out) return src;
    const char* in = src;
    char* o = out;
    while (*in) {
        if (*in == 't' && !strncmp(in, "texture", 7) &&
            !(in > src && (isalnum((unsigned char)in[-1]) || in[-1] == '_' || in[-1] == '.')) &&
            !(isalnum((unsigned char)in[7]) || in[7] == '_')) {
            const char* q = in + 7;
            while (*q == ' ' || *q == '\t') q++;
            if (*q != '(') {
                memcpy(o, "zsmp_texture", 12);
                o += 12;
                in += 7;
                continue;
            }
        }
        *o++ = *in++;
    }
    *o = '\0';
    free(src);
    return out;
}

// FIX: strip_uniform_initializers
//
// Project Zomboid build 42 shaders use GLSL 1.20 syntax where
// uniforms can have default values, e.g.:
//   uniform float chunkDepth = 0.0;
//   uniform int useTexture = 1;
//
// This is valid in desktop GLSL but ILLEGAL in GLSL ES (any version).
// The SPIRV-Cross / ConvertShader pipeline does NOT strip these,
// so Adreno (and other mobile drivers) reject the shader with:
//   ERROR: 'uniform' : cannot initialize this type of qualifier
//
// This function removes the "= <value>" part from uniform declarations,
// converting them to plain declarations that GLSL ES accepts.
//
// Called after every conversion path in gl4es_glShaderSource and redoShader.
// Safe to call on shaders that have no initializers — it's a no-op for those.
// ============================================================
static char* strip_uniform_initializers(char* src) {
    if (!src) return src;

    size_t len = strlen(src);
    char* result = (char*)malloc(len + 1);
    if (!result) return src;

    const char* in = src;
    char* out = result;

    while (*in) {
        const char* uni = strstr(in, "uniform");
        if (!uni) {
            size_t rest = strlen(in);
            memcpy(out, in, rest);
            out += rest;
            break;
        }

        if (uni != in) {
            char prev = *(uni - 1);
            if (isalnum((unsigned char)prev) || prev == '_') {
                size_t chunk = (uni - in) + 1;
                memcpy(out, in, chunk);
                out += chunk;
                in = uni + 1;
                continue;
            }
        }

        char after = *(uni + 7);
        if (isalnum((unsigned char)after) || after == '_') {
            size_t chunk = (uni - in) + 8;
            memcpy(out, in, chunk);
            out += chunk;
            in = uni + 8;
            continue;
        }

        const char* semi = strchr(uni, ';');
        if (!semi) {
            size_t rest = strlen(in);
            memcpy(out, in, rest);
            out += rest;
            break;
        }

        const char* eq = NULL;
        for (const char* p = uni + 7; p < semi; p++) {
            if (*p == '=') {
                char next = *(p + 1);
                char prev = (p > uni + 7) ? *(p - 1) : '\0';
                if (next == '=' || prev == '!' || prev == '<' || prev == '>')
                    continue;
                eq = p;
                break;
            }
        }

        if (eq) {
            size_t keep = eq - in;
            memcpy(out, in, keep);
            out += keep;
            *out++ = ';';
            in = semi + 1;
            SHUT_LOGD("strip_uniform_initializers: removed default value from uniform declaration\n");
        } else {
            size_t chunk = (semi - in) + 1;
            memcpy(out, in, chunk);
            out += chunk;
            in = semi + 1;
        }
    }

    *out = '\0';
    free(src);
    return result;
}
// ============================================================
// END FIX: strip_uniform_initializers
// ============================================================

static char* strip_texture_lod_bias(char* src, int is_fragment) {
    if (!src || !is_fragment) return src;

    size_t len = strlen(src);
    char* result = (char*)malloc(len + 1);
    if (!result) return src;

    const char* in = src;
    char* out = result;

    while (*in) {
        const char* tex = strstr(in, "texture(");
        if (!tex) {
            size_t rest = strlen(in);
            memcpy(out, in, rest);
            out += rest;
            break;
        }

        // Copy up to and including "texture("
        size_t chunk = (tex - in) + 8;
        memcpy(out, in, chunk);
        out += chunk;
        in = tex + 8;

        // Find the closing paren, count args
        int depth = 1;
        const char* p = in;
        const char* second_comma = NULL;
        int commas = 0;
        while (*p && depth > 0) {
            if (*p == '(') depth++;
            else if (*p == ')') { depth--; if (depth == 0) break; }
            else if (*p == ',' && depth == 1) {
                commas++;
                if (commas == 2) second_comma = p;
            }
            p++;
        }

        if (second_comma && *p == ')') {
            // 3 args — copy only up to second comma, then add ")"
            size_t keep = second_comma - in;
            memcpy(out, in, keep);
            out += keep;
            *out++ = ')';
            in = p + 1;
        } else {
            // 2 args or malformed — copy as-is up to closing paren
            size_t keep = p - in + (*p == ')' ? 1 : 0);
            memcpy(out, in, keep);
            out += keep;
            in = p + (*p == ')' ? 1 : 0);
        }
    }

    *out = '\0';
    free(src);
    return result;
}

GLuint APIENTRY_GL4ES gl4es_glCreateShader(GLenum shaderType) {
    DBG(SHUT_LOGD("glCreateShader(%s)\n", PrintEnum(shaderType)))
    // sanity check
    if (shaderType != GL_VERTEX_SHADER && shaderType != GL_FRAGMENT_SHADER && shaderType != GL_GEOMETRY_SHADER &&
        shaderType != GL_COMPUTE_SHADER) {
        DBG(SHUT_LOGD("Invalid shader type\n"))
        errorShim(GL_INVALID_ENUM);
        return 0;
    }
    static GLuint lastshader = 0;
    GLuint shader;
    // create the shader
    LOAD_GLES2(glCreateShader);
    if (gles_glCreateShader) {
        shader = gles_glCreateShader(shaderType);
        if (!shader) {
            DBG(SHUT_LOGD("Failed to create shader\n"))
            errorGL();
            return 0;
        }
    } else {
        shader = ++lastshader;
        noerrorShim();
    }
    // store the new empty shader in the list for later use
    khint_t k;
    int ret;
    khash_t(shaderlist)* shaders = glstate->glsl->shaders;
    k = kh_get(shaderlist, shaders, shader);
    shader_t* glshader = NULL;
    if (k == kh_end(shaders)) {
        k = kh_put(shaderlist, shaders, shader, &ret);
        glshader = kh_value(shaders, k) = (shader_t*)calloc(1, sizeof(shader_t));
    } else {
        glshader = kh_value(shaders, k);
    }
    glshader->id = shader;
    glshader->type = shaderType;
    if (glshader->source) {
        free(glshader->source);
        glshader->source = NULL;
    }
    glshader->need.need_texcoord = -1;

    // all done
    return shader;
}

void actually_deleteshader(GLuint shader) {
    khint_t k;
    khash_t(shaderlist)* shaders = glstate->glsl->shaders;
    k = kh_get(shaderlist, shaders, shader);
    if (k != kh_end(shaders)) {
        shader_t* glshader = kh_value(shaders, k);
        if (glshader->deleted && !glshader->attached) {
            kh_del(shaderlist, shaders, k);
            if (glshader->source) free(glshader->source);
            if (glshader->converted) free(glshader->converted);
            // ZOMDROID FIX: an owned copy since the BindFragData overflow fix
            if (glshader->before_patch) free(glshader->before_patch);
            free(glshader);
        }
    }
}

void actually_detachshader(GLuint shader) {
    khint_t k;
    khash_t(shaderlist)* shaders = glstate->glsl->shaders;
    k = kh_get(shaderlist, shaders, shader);
    if (k != kh_end(shaders)) {
        shader_t* glshader = kh_value(shaders, k);
        if ((--glshader->attached) < 1 && glshader->deleted) actually_deleteshader(shader);
    }
}

void APIENTRY_GL4ES gl4es_glDeleteShader(GLuint shader) {
    DBG(SHUT_LOGD("glDeleteShader(%d)\n", shader))
    // sanity check...
    CHECK_SHADER(void, shader)
    // delete the shader from the list
    if (!glshader) {
        noerrorShim();
        return;
    }
    // ZOMDROID FIX (Codex audit p.4, native shader leak): forward the DRIVER deletion
    // immediately — per spec the driver defers actual disposal while the shader is
    // still attached. The old code skipped the driver call for attached shaders and
    // never issued it later (the wrapper-free path has no GLES call), so every shader
    // going through the standard attach/link/delete pattern leaked its native compiler
    // objects for the life of the process — hundreds per session.
    int zwas_deleted = glshader->deleted;
    glshader->deleted = 1;
    noerrorShim();
    if (!zwas_deleted) {
        LOAD_GLES2(glDeleteShader);
        if (gles_glDeleteShader) {
            errorGL();
            gles_glDeleteShader(shader);
        }
    }
    if (!glshader->attached) {
        actually_deleteshader(shader);
    }
}

#define LOG_CHUNK_SIZE 4000

void log_long_shader(const char* tag, const char* text) {
    size_t len = strlen(text);
    for (size_t i = 0; i < len; i += LOG_CHUNK_SIZE) {
        size_t chunk_len = (i + LOG_CHUNK_SIZE < len) ? LOG_CHUNK_SIZE : (len - i);
        char buffer[LOG_CHUNK_SIZE + 1];
        memcpy(buffer, text + i, chunk_len);
        buffer[chunk_len] = '\0';
        SHUT_LOGD("%s", buffer);
    }
}

void APIENTRY_GL4ES gl4es_glCompileShader(GLuint shader) {
    DBG(SHUT_LOGD("glCompileShader(%d)\n", shader))
    // look for the shader
    CHECK_SHADER(void, shader)

    glshader->compiled = 1;
    LOAD_GLES2(glCompileShader);
    if (gles_glCompileShader) {
        struct timespec zcp0;
        clock_gettime(CLOCK_MONOTONIC, &zcp0);
        gles_glCompileShader(glshader->id);
        errorGL();
        // if(globals4es.logshader) {
        { // always log the error of shader
            // get compile status and print shaders sources if compile fail...
            LOAD_GLES2(glGetShaderiv);
            LOAD_GLES2(glGetShaderInfoLog);
            GLint status = 0;
            gles_glGetShaderiv(glshader->id, GL_COMPILE_STATUS, &status);
            ZOMDROID_VDBG("ZOMDROID_DBG: glCompileShader id=%d status=%d\n", glshader->id, status);
            {
                struct timespec zcp1;
                clock_gettime(CLOCK_MONOTONIC, &zcp1);
                long zcpms = (zcp1.tv_sec - zcp0.tv_sec) * 1000 + (zcp1.tv_nsec - zcp0.tv_nsec) / 1000000;
                extern void zomdroid_gltrace(const char* fmt, ...);
                zomdroid_gltrace("COMPILE shader=%u native=%u status=%d ms=%ld", shader, glshader->id, status, zcpms);
                if (status != GL_TRUE) {
                    // full failing source + driver log into the field-readable dump
                    FILE* zf = fopen("/data/data/com.zomdroid/files/failed_shaders.txt", "a");
                    if (zf) {
                        char zlog[1024];
                        GLint zlen = 0;
                        gles_glGetShaderInfoLog(glshader->id, sizeof(zlog) - 1, &zlen, zlog);
                        zlog[zlen > 0 ? zlen : 0] = '\0';
                        fprintf(zf, "\n==== COMPILE FAILED shader %u (native %u) ====\nDRIVER LOG:\n%s\nSOURCE:\n%s\n",
                                shader, glshader->id, zlog, glshader->converted ? glshader->converted : "(null)");
                        fclose(zf);
                    }
                }
            }
            if (status != GL_TRUE) {
                DBG(SHUT_LOGD("LIBGL: Error while compiling shader %d. Original source is:\n%s\n=======\n",
                              glshader->id, glshader->source);)
                DBG(SHUT_LOGD("ShaderConv Source is:\n%s\n=======\n", glshader->converted);)
                char tmp[500];
                GLint length;
                gles_glGetShaderInfoLog(glshader->id, 500, &length, tmp);
                SHUT_LOGD("Compiler message is\n%s\nLIBGL: End of Error log\n", tmp);
            }
        }
    } else
        noerrorShim();
}

bool can_run_essl3(int esversion, const char* glsl) {
    int glsl_version = 0;
    if (strncmp(glsl, "#version 100", 12) == 0) {
        return true;
    } else if (strncmp(glsl, "#version 300 es", 15) == 0) {
        return true;
    } else if (strncmp(glsl, "#version 310 es", 15) == 0) {
        glsl_version = 310;
    } else if (strncmp(glsl, "#version 320 es", 15) == 0) {
        glsl_version = 320;
    } else {
        return false;
    }
    if (esversion >= glsl_version) {
        return true;
    } else {
        return false;
    }
}

bool is_direct_shader(char* glsl) {
    bool es2_ability = glstate->glsl->es2 && !strncmp(glsl, "#version 100", 12);
    bool es3_ability = globals4es.es >= 3 &&
                       can_run_essl3(globals4es.esversion != -1 ? globals4es.esversion : globals4es.es * 100, glsl);
    return es2_ability || es3_ability;
}

char* replace_version_line(const char* text) {
    if (!text) return NULL;

    const char* line_start = text;
    const char* p = text;
    const char* replace_str = "#version 330 compatibility\n";
    size_t replace_len = strlen(replace_str);

    while (*p) {
        if (*p == '\n' || *(p + 1) == '\0') {
            size_t len = p - line_start + (*(p + 1) == '\0' ? 1 : 0);

            size_t i = 0;
            while (i < len && isspace((unsigned char)line_start[i]))
                i++;

            if (strncmp(line_start + i, "#version", 8) == 0) {
                size_t new_len = strlen(text) - len + replace_len;
                char* new_text = malloc(new_len + 1);
                if (!new_text) return NULL;

                memcpy(new_text, text, line_start - text);
                memcpy(new_text + (line_start - text), replace_str, replace_len);
                strcpy(new_text + (line_start - text) + replace_len, p + (*(p + 1) == '\0' ? 0 : 1));

                return new_text;
            }

            line_start = p + 1;
        }
        p++;
    }

    return strdup(text);
}

char* replace_version_line_460(char* text) {
    const char* new_version = "#version 460\n";
    if (!text || !new_version) return NULL;

    const char* p = text;
    const char* line_start = text;
    size_t text_len = strlen(text);
    size_t new_version_len = strlen(new_version);

    while (*p) {
        const char* line_end = strchr(p, '\n');
        const char* next_line = NULL;
        if (line_end) {
            next_line = line_end + 1;
        } else {
            line_end = text + text_len;
            next_line = line_end;
        }

        const char* cur = p;
        while (cur < line_end && isspace((unsigned char)*cur) && *cur != '\n' && *cur != '\r')
            cur++;

        const char* tok = "#version";
        size_t toklen = strlen(tok);
        if ((size_t)(line_end - cur) >= toklen && strncmp(cur, tok, toklen) == 0) {
            const char next_char = (cur + toklen < line_end) ? *(cur + toklen) : '\0';
            if (next_char == '\0' || isspace((unsigned char)next_char)) {
                size_t prefix_len = cur - text;
                prefix_len = p - text;

                const char* suffix_start = next_line;

                size_t suffix_len = strlen(suffix_start);
                size_t new_total = prefix_len + new_version_len + suffix_len;
                char* out = (char*)malloc(new_total + 1);
                if (!out) return NULL;

                memcpy(out, text, prefix_len);
                memcpy(out + prefix_len, new_version, new_version_len);
                memcpy(out + prefix_len + new_version_len, suffix_start, suffix_len);
                out[new_total] = '\0';
                return out;
            }
        }
        p = next_line;
    }

    return strdup(text);
}

static char* force_replace_version(const char* text, const char* new_version_line) {
    if (!text) return NULL;

    const char* version_tag = "#version";
    const char* p = text;
    // Skip leading whitespace
    while (*p && isspace((unsigned char)*p)) p++;

    // Check if the source starts with #version
    if (strncmp(p, version_tag, strlen(version_tag)) == 0) {
        const char* first_line_end = strchr(p, '\n');
        if (!first_line_end) {
            first_line_end = p + strlen(p); // Handle single-line shaders
        }

        // Calculate the start of the rest of the code
        const char* rest_of_code = first_line_end;
        if (*rest_of_code == '\n') {
            rest_of_code++; // Move past the newline character
        }

        // Allocate memory and build the new shader string
        size_t new_version_len = strlen(new_version_line);
        size_t rest_len = strlen(rest_of_code);
        char* new_text = (char*)malloc(new_version_len + 1 + rest_len + 1); // +1 for newline, +1 for null terminator
        if (!new_text) return NULL;

        strcpy(new_text, new_version_line);
        strcat(new_text, "\n");
        strcat(new_text, rest_of_code);

        return new_text;
    }

    // If #version is not found at the beginning, insert the new version line at the top
    size_t text_len = strlen(text);
    size_t new_version_len = strlen(new_version_line);
    char* new_text = (char*)malloc(new_version_len + 1 + text_len + 1);
    if (!new_text) return NULL;

    strcpy(new_text, new_version_line);
    strcat(new_text, "\n");
    strcat(new_text, text);

    return new_text;
}

static char* insert_after_preamble(const char* text, const char* insert_str) {
    if (!text || !insert_str) return NULL;

    const char* p = text;
    const char* last_preamble_pos = text;

    // Find the end of the preamble (all lines starting with #, comments, or are empty)
    while (*p) {
        const char* line_start = p;

        // Skip leading whitespace
        while (*p && (*p == ' ' || *p == '\t')) p++;

        if (*p == '#' || (*p == '/' && *(p+1) == '/') || *p == '\n' || *p == '\r' || *p == '\0') {
            // This line is a preprocessor, comment, or empty. Continue.
            const char* line_end = strchr(line_start, '\n');
            if (line_end) {
                p = line_end + 1;
                last_preamble_pos = p;
            } else {
                // End of string
                last_preamble_pos = p + strlen(p);
                break;
            }
        } else {
            // First line of actual code found. We should insert before it.
            break;
        }
    }

    // Now, build the new string
    size_t head_len = last_preamble_pos - text;
    size_t insert_len = strlen(insert_str);
    size_t tail_len = strlen(last_preamble_pos);

    char* new_text = (char*)malloc(head_len + insert_len + tail_len + 1);
    if (!new_text) return NULL;

    memcpy(new_text, text, head_len);
    memcpy(new_text + head_len, insert_str, insert_len);
    memcpy(new_text + head_len + insert_len, last_preamble_pos, tail_len);
    new_text[head_len + insert_len + tail_len] = '\0';

    return new_text;
}


bool check_version_compatibility(const char* str) {
    if (str == NULL) {
        return false;
    }

    const char* line_start = str;

    while (*line_start) {
        const char* line_end = strpbrk(line_start, "\n");
        size_t line_len;

        if (line_end) {
            line_len = line_end - line_start;
        } else {
            line_len = strlen(line_start);
        }

        if (strncmp(line_start, "#version", 8) == 0) {
            char current_line[line_len + 1];
            strncpy(current_line, line_start, line_len);
            current_line[line_len] = '\0';

            if (strstr(current_line, "compatibility") != NULL) {
                return true;
            }
        }

        if (line_end) {
            line_start = line_end + 1;
        } else {
            break;
        }
    }

    return false;
}

void remove_before_version(char* str) {
    if (str == NULL) {
        return;
    }

    char* version_ptr = strstr(str, "#version");

    if (version_ptr != NULL) {
        size_t remaining_len = strlen(version_ptr) + 1;

        memmove(str, version_ptr, remaining_len);
    }
}

int has_gl_FragColor_definition(const char* s) {
    const char* p = s;
    while (*p) {
        if (strncmp(p, "vec4", 4) == 0) {
            p += 4;
            while (*p && isspace((unsigned char)*p))
                p++;
            if (strncmp(p, "gl4es_FragColor", 12) == 0) {
                return 1;
            }
        } else {
            p++;
        }
    }
    return 0;
}

char* replace_glFragColor(const char* src) {
    if (!src) return NULL;

    const char* target = "gl_FragColor";
    const char* replacement = "gl4es_FragColor";
    size_t target_len = strlen(target);
    size_t replacement_len = strlen(replacement);

    size_t count = 0;
    const char* p = src;
    while ((p = strstr(p, target)) != NULL) {
        if ((p == src || !((p[-1] >= 'a' && p[-1] <= 'z') || (p[-1] >= 'A' && p[-1] <= 'Z') ||
                           (p[-1] >= '0' && p[-1] <= '9') || p[-1] == '_')) &&
            !((p[target_len] >= 'a' && p[target_len] <= 'z') || (p[target_len] >= 'A' && p[target_len] <= 'Z') ||
              (p[target_len] >= '0' && p[target_len] <= '9') || p[target_len] == '_')) {
            count++;
        }
        p += target_len;
    }

    size_t new_len = strlen(src) + count * (replacement_len - target_len) + 1;
    char* result = (char*)malloc(new_len);
    if (!result) return NULL;

    const char* src_p = src;
    char* dst_p = result;
    while ((p = strstr(src_p, target)) != NULL) {
        int valid =
                (p == src_p || !((p[-1] >= 'a' && p[-1] <= 'z') || (p[-1] >= 'A' && p[-1] <= 'Z') ||
                                 (p[-1] >= '0' && p[-1] <= '9') || p[-1] == '_')) &&
                !((p[target_len] >= 'a' && p[target_len] <= 'z') || (p[target_len] >= 'A' && p[target_len] <= 'Z') ||
                  (p[target_len] >= '0' && p[target_len] <= '9') || p[target_len] == '_');
        size_t copy_len = p - src_p;
        memcpy(dst_p, src_p, copy_len);
        dst_p += copy_len;

        if (valid) {
            memcpy(dst_p, replacement, replacement_len);
            dst_p += replacement_len;
        } else {
            memcpy(dst_p, target, target_len);
            dst_p += target_len;
        }
        src_p = p + target_len;
    }
    strcpy(dst_p, src_p);

    return result;
}

char* insert_gl_FragColor_if_missing(const char* shader) {
    if (has_gl_FragColor_definition(shader)) {
        return strdup(shader);
    }

    const char* insert_line = "out vec4 gl4es_FragColor;\n";

    const char* pos = shader;
    const char* insert_pos = shader;
    if (strncmp(shader, "#version", 8) == 0) {
        const char* newline = strchr(shader, '\n');
        if (newline) {
            insert_pos = newline + 1;
        }
    }

    size_t new_len = strlen(shader) + strlen(insert_line) + 1;
    char* new_shader = (char*)malloc(new_len);
    if (!new_shader) return NULL;

    size_t head_len = insert_pos - shader;
    memcpy(new_shader, shader, head_len);
    strcpy(new_shader + head_len, insert_line);
    strcpy(new_shader + head_len + strlen(insert_line), insert_pos);

    return new_shader;
}

int contains_glFragColor(const char* src) {
    if (!src) return 0;

    const char* target = "gl_FragColor";
    size_t target_len = strlen(target);

    const char* p = src;
    while ((p = strstr(p, target)) != NULL) {
        int valid = 1;

        if (p != src && (isalnum((unsigned char)p[-1]) || p[-1] == '_')) {
            valid = 0;
        }
        if (p[target_len] != '\0' && (isalnum((unsigned char)p[target_len]) || p[target_len] == '_')) {
            valid = 0;
        }

        if (valid) return 1;

        p += target_len;
    }

    return 0;
}

char* replace_fragdata0(const char* source) {
    const char* target = "gl_FragData[0].";
    const char* replacement = "vgpu_FragData0";
    size_t target_len = strlen(target);
    size_t replacement_len = strlen(replacement);

    size_t count = 0;
    const char* p = source;
    while ((p = strstr(p, target)) != NULL) {
        count++;
        p += target_len;
    }

    size_t new_len = strlen(source) + count * (replacement_len - target_len);
    char* result = (char*)malloc(new_len + 1);
    if (!result) return NULL;

    const char* src = source;
    char* dst = result;
    while ((p = strstr(src, target)) != NULL) {
        size_t len = p - src;
        memcpy(dst, src, len);
        dst += len;
        memcpy(dst, replacement, replacement_len);
        dst += replacement_len;
        src = p + target_len;
    }
    strcpy(dst, src);

    return result;
}

#define GLSL_CONVERTED_MARK "//__glsl_builtin_converted__\n"

char* mark_glsl_builtin_converted(const char* source) {
    if (!source) return NULL;
    size_t srcLen = strlen(source);
    size_t markLen = strlen(GLSL_CONVERTED_MARK);

    char* result = (char*)malloc(srcLen + markLen + 2);
    if (!result) return NULL;

    memcpy(result, source, srcLen);
    result[srcLen] = '\n';
    memcpy(result + srcLen + 1, GLSL_CONVERTED_MARK, markLen);
    result[srcLen + 1 + markLen] = '\0';

    return result;
}

int is_glsl_builtin_converted(const char* source) {
    if (!source) return 0;
    size_t srcLen = strlen(source);
    size_t markLen = strlen(GLSL_CONVERTED_MARK);

    if (srcLen < markLen) return 0;

    const char* tail = source + srcLen - markLen;
    return (strcmp(tail, GLSL_CONVERTED_MARK) == 0);
}

int first_three_lines_contains_BSL(const char* text) {
    if (!text) return 0;

    int line_count = 0;
    const char* line_start = text;
    const char* p = text;

    while (*p && line_count < 3) {
        if (*p == '\n' || *(p + 1) == '\0') {
            size_t len = p - line_start + (*(p + 1) == '\0' ? 1 : 0);

            if (len > 0) {
                if (memmem(line_start, len, "BSL", 3) != NULL) {
                    return 1;
                }
            }

            line_count++;
            line_start = p + 1;
        }
        p++;
    }
    return 0;
}

static char* replace_all(const char* src, const char* old, const char* repl) {
    if (!src || !old || !repl) return NULL;
    size_t src_len = strlen(src);
    size_t old_len = strlen(old);
    if (old_len == 0) return NULL;
    size_t repl_len = strlen(repl);

    size_t count = 0;
    const char* p = src;
    while ((p = strstr(p, old)) != NULL) {
        count++;
        p += old_len;
    }
    if (count == 0) {
        char* out = (char*)malloc(src_len + 1);
        if (out) memcpy(out, src, src_len + 1);
        return out;
    }

    size_t new_len = src_len + count * (repl_len - old_len);
    char* out = (char*)malloc(new_len + 1);
    if (!out) return NULL;

    const char* cur = src;
    char* dst = out;
    while ((p = strstr(cur, old)) != NULL) {
        size_t chunk = p - cur;
        memcpy(dst, cur, chunk);
        dst += chunk;
        memcpy(dst, repl, repl_len);
        dst += repl_len;
        cur = p + old_len;
    }
    size_t tail = strlen(cur);
    memcpy(dst, cur, tail);
    dst += tail;
    *dst = '\0';
    return out;
}

static char* insert_after_first(const char* src, const char* anchor, const char* insert_str) {
    if (!src || !anchor || !insert_str) return NULL;
    const char* p = strstr(src, anchor);
    if (!p) {
        char* out = (char*)malloc(strlen(src) + 1);
        if (out) memcpy(out, src, strlen(src) + 1);
        return out;
    }
    size_t prefix_len = (p - src) + strlen(anchor);
    size_t insert_len = strlen(insert_str);
    size_t src_len = strlen(src);
    size_t new_len = src_len + insert_len;
    char* out = (char*)malloc(new_len + 1);
    if (!out) return NULL;

    memcpy(out, src, prefix_len);
    memcpy(out + prefix_len, insert_str, insert_len);
    memcpy(out + prefix_len + insert_len, src + prefix_len, src_len - prefix_len);
    out[new_len] = '\0';
    return out;
}

char* bsl_patch(const char* glsl) {
    if (!glsl) return NULL;

    const char* old1 = "cloudBlendOpacity = step(viewLength, cloudViewLength);";
    const char* new1 = "cloudBlendOpacity = 0.7;";
    const char* old2 = "vgpu_FragData1 = vec4(temporalData, temporalColor);";
    const char* new2 = "vgpu_FragData1 = vec4(color, 1.0);";
    const char* old3 = "vgpu_FragData1 = vec4(vl, 1.0);";
    const char* new3 = "vgpu_FragData1 = color;";
    const char* old4 = "highp vec3 bloom = texture(colortex1, ((coord / vec2(scale)) + offset) * resScale).xyz;";
    const char* new4 = "highp vec3 bloom = vec3(0.0, 0.0, 0.0);";
    const char* old5 = "highp float isGlowing = texture(colortex3, texCoord).z;";
    const char* new5 = "highp float isGlowing = 0.0;";
    char* s = replace_all(glsl, old1, new1);
    if (!s) return NULL;
    char* s2 = replace_all(s, old2, new2);
    free(s);
    if (!s2) return NULL;
    char* s3 = replace_all(s2, old3, new3);
    free(s2);
    if (!s3) return NULL;
    char* s4 = replace_all(s3, old4, new4);
    free(s3);
    if (!s4) return NULL;
    char* s5 = replace_all(s4, old5, new5);
    free(s4);
    if (!s5) return NULL;

    const char* trigger = "highp float quality[";
    char* cur = s5;

    if (strstr(cur, trigger) != NULL) {
        const char* anchor0 = "layout(location = 0) out highp vec4 vgpu_FragData0;";
        const char* line0 = "\nlayout(location = 1) out highp vec4 vgpu_FragData1;";
        char* t = insert_after_first(cur, anchor0, line0);
        free(cur);
        cur = t ? t : strdup("");
        const char* anchor1 = "vgpu_FragData0 = vec4(color, 1.0);";
        const char* line1 = "\nvgpu_FragData1 = vec4(color,1.0);";
        t = insert_after_first(cur, anchor1, line1);
        free(cur);
        cur = t ? t : strdup("");
    }
    return cur;
}

static char* replace_version_to_es(const char* text, int esversion) {
    if (!text) return NULL;
    char new_version[50];
    snprintf(new_version, 50, "#version %d es\n", esversion);
    const char* p = text;
    const char* line_start = text;
    size_t text_len = strlen(text);
    while (*p) {
        const char* line_end = strchr(p, '\n');
        const char* next_line = NULL;
        if (line_end) {
            next_line = line_end + 1;
        } else {
            line_end = text + text_len;
            next_line = line_end;
        }
        const char* cur = p;
        while (cur < line_end && isspace((unsigned char)*cur) && *cur != '\n' && *cur != '\r')
            cur++;
        if (strncmp(cur, "#version", 8) == 0) {
            size_t prefix_len = p - text;
            size_t line_len = next_line - p;
            size_t suffix_len = text_len - (next_line - text);
            size_t new_total = prefix_len + strlen(new_version) + suffix_len;
            char* out = (char*)malloc(new_total + 1);
            if (!out) return NULL;
            memcpy(out, text, prefix_len);
            memcpy(out + prefix_len, new_version, strlen(new_version));
            memcpy(out + prefix_len + strlen(new_version), next_line, suffix_len);
            out[new_total] = '\0';
            return out;
        }
        p = next_line;
    }
    return strdup(text);
}

void set_es_version();
void APIENTRY_GL4ES gl4es_glShaderSource(GLuint shader, GLsizei count, const GLchar* const* string,
                                         const GLint* length) {
    if (!globals4es.esversion) set_es_version();
    DBG(SHUT_LOGD("glShaderSource(%d, %d, %p, %p)\n", shader, count, string, length))
    // sanity check
    if (count <= 0 || !string) {
        errorShim(GL_INVALID_VALUE);
        return;
    }

    for (int i = 0; i < count; i++) {
        if (!string[i]) {
            errorShim(GL_INVALID_VALUE);
            return;
        }
    }

    CHECK_SHADER(void, shader)
    // ZOMDROID FIX: new source = new uniform table. The count was never reset, so every
    // recompile of the same shader object appended duplicate entries; past 1024 entries
    // process_uniform_declarations wrote beyond the array (silent heap smash).
    {
        int zmax = glshader->uniforms_declarations_count;
        if (zmax > MAX_UNIFORM_VARIABLE_NUMBER) zmax = MAX_UNIFORM_VARIABLE_NUMBER;
        for (int zi = 0; zi < zmax; zi++) {
            glshader->uniforms_declarations[zi].variable[0] = '\0';
            glshader->uniforms_declarations[zi].type[0] = '\0';
            glshader->uniforms_declarations[zi].initial_value[0] = '\0';
        }
        glshader->uniforms_declarations_count = 0;
    }
    // ZOMDROID FIX: before_patch (an owned copy since the BindFragData overflow fix)
    // belongs to the previous source — drop it so the fallback recompile in useProgram
    // can't feed stale text.
    if (glshader->before_patch) {
        free(glshader->before_patch);
        glshader->before_patch = NULL;
    }
    // get the size of the shader sources and than concatenate in a single string
    int l = 0;
    for (int i = 0; i < count; i++)
        l += (length && length[i] >= 0) ? length[i] : strlen(string[i]);
    if (glshader->source) free(glshader->source);
    glshader->source = malloc(l + 1);
    memset(glshader->source, 0, l + 1);
    if (length) {
        for (int i = 0; i < count; i++) {
            if (length[i] >= 0)
                strncat(glshader->source, string[i], length[i]);
            else
                strcat(glshader->source, string[i]);
        }
    } else {
        for (int i = 0; i < count; i++)
            strcat(glshader->source, string[i]);
    }
    LOAD_GLES2(glShaderSource);
    if (gles_glShaderSource) {
        int isFPEShader = (strstr(glshader->source, fpeshader_signature) != NULL) ? 1 : 0;
        // adapt shader if needed (i.e. not an es2 context and shader is not #version 100)
        if (is_direct_shader(glshader->source)) {
            ZOMDROID_VDBG("ZOMDROID_DBG: shader=%d path=DIRECT\n", shader);
            glshader->converted = strdup(glshader->source);
        } else if (globals4es.simple_shaderconv && !isFPEShader) {
            // ZOMDROID DIAG (Mali field debugging): breadcrumb before conversion — a crash
            // inside the converter shows as a trailing 'CONVERT begin' without 'end'.
            struct timespec zcvt0;
            clock_gettime(CLOCK_MONOTONIC, &zcvt0);
            {
                extern void zomdroid_gltrace(const char* fmt, ...);
                zomdroid_gltrace("CONVERT begin shader=%u type=0x%X len=%d", shader, glshader->type,
                                 glshader->source ? (int)strlen(glshader->source) : -1);
            }
            size_t zcc_olen = strlen(glshader->source);
            char* zcc_orig = (char*)malloc(zcc_olen + 1);
            if (zcc_orig) memcpy(zcc_orig, glshader->source, zcc_olen + 1);
            int zmemo_hit = zcc_orig ? zomdroid_memo_try(glshader, zcc_orig, zcc_olen) : 0;
            int zcc_hit = zmemo_hit ? 1 : (zcc_orig ? zomdroid_ccache_try(glshader, zcc_orig, zcc_olen) : 0);
            if (zcc_hit) {
                glshader->is_converted_essl_320 = 0;
                extern void zomdroid_gltrace(const char* fmt, ...);
                zomdroid_gltrace("CONVERT %s shader=%u clen=%d", zmemo_hit ? "memo" : "cached", shader,
                                 (int)strlen(glshader->converted));
            }
            if (!zcc_hit) {
            ZOMDROID_VDBG("ZOMDROID_DBG: shader=%d path=SIMPLE_SHADERCONV\n", shader);glshader->converted = strip_uniform_initializers(glshader->converted);
            // ZOMDROID FIX (invisible character): strip PZ's redefinitions of GLSL
            // builtins (max/min/clamp) — GLES3 link fails on them.
            zomdroid_strip_builtin_redefs(glshader->source);
            // ZOMDROID FIX (vehicle/blur/water): Adreno has no implicit int->float —
            // legalize const-float int initializers and int literals in builtin calls.
            glshader->source = zomdroid_floatify_const_float(glshader->source);
            glshader->source = zomdroid_floatify_builtin_args(glshader->source);
            // ZOMDROID FIX (Mali wave 2): mixed arithmetic/comparisons with float locals
            // and the reserved sampler name "texture" (see pass definitions above).
            glshader->source = zomdroid_floatify_float_context_lines(glshader->source);
            glshader->source = zomdroid_rename_reserved_samplers(glshader->source);
            glshader->source = zomdroid_wrap_texturesize_arith(glshader->source);
            // wave 4: typed ivec expressions in float-vector arithmetic (see pass above)
            glshader->source = zomdroid_vecify_ivec_arith(glshader->source);
            // wave 5: globals initialized from uniforms (puddles/water on ANGLE/Mali)
            glshader->source = zomdroid_hoist_global_initializers(glshader->source);
            // ZOMDROID FIX (white-world root, final link): the SIMPLE path (the one PZ
            // actually uses) stripped GLSL uniform initializers WITHOUT recording them,
            // so set_uniforms_default_value never had anything to restore (zero DEFAULT
            // lines in traces). process_uniform_declarations strips AND records.
            glshader->source = process_uniform_declarations(glshader->source, glshader->uniforms_declarations,
                                                            &glshader->uniforms_declarations_count);
            glshader->converted = strip_texture_lod_bias(glshader->converted,
                                                         glshader->type == GL_FRAGMENT_SHADER ? 1 : 0);
            // ZOMDROID FIX (Codex pre-release audit, leak): ConvertShaderConditionally
            // stores its result in glshader->converted AND returns that same pointer —
            // the strdup here duplicated it and leaked the stored original. The callee
            // rebuilds from glshader->source and overwrites `converted` without freeing
            // it, so drop the pre-call buffer too.
            free(glshader->converted);
            glshader->converted = NULL;
            glshader->converted = ConvertShaderConditionally(glshader);
            // ZOMDROID FIX (invisible character, part 2): legalize PZ's GLSL-1.20-era
            // mixed int/float calls to builtins (clamp(x,0,1.0) etc.) whose custom
            // overloads we strip above. Guarded with #ifdef inside InsertExtension.
            {
                int zclen = (int)strlen(glshader->converted) + 1;
                int zcip = FindPositionAfterVersion(glshader->converted);
                glshader->converted =
                        InsertExtension(glshader->converted, &zclen, zcip + 1, "GL_EXT_shader_implicit_conversions");
            }
            glshader->is_converted_essl_320 = 0;
            {
                struct timespec zcvt1;
                clock_gettime(CLOCK_MONOTONIC, &zcvt1);
                long zcms = (zcvt1.tv_sec - zcvt0.tv_sec) * 1000 + (zcvt1.tv_nsec - zcvt0.tv_nsec) / 1000000;
                extern void zomdroid_gltrace(const char* fmt, ...);
                zomdroid_gltrace("CONVERT end shader=%u clen=%d ms=%ld", shader,
                                 glshader->converted ? (int)strlen(glshader->converted) : -1, zcms);
            }
            if (zcc_orig) zomdroid_ccache_store(glshader, zcc_orig, zcc_olen);
            if (zcc_orig) zomdroid_memo_store(glshader, zcc_orig, zcc_olen);
            } // !zcc_hit
            free(zcc_orig);

        } else {
            int glsl_version = getGLSLVersion(glshader->source);
            DBG(SHUT_LOGD("[INFO] [Shader] Shader source: "))
            DBG(SHUT_LOGD("%s", glshader->source))
//            int isFPEShader = (strstr(glshader->source, fpeshader_signature) != NULL) ? 1 : 0;
//            if (glsl_version < 140 && !isFPEShader) {
            //              glshader->source = replace_version_line(glshader->source);
            //            glsl_version = 460;
            //      }
            if (glsl_version < 140 || (globals4es.es < 3 && globals4es.esversion < 300)) {
                // ZOMDROID FIX: stored-and-returned pointer — the strdup leaked it
                glshader->converted = ConvertShaderConditionally(glshader);
                glshader->is_converted_essl_320 = 0;
            } else {
                if (check_version_compatibility(glshader->source)) {
                    // bool isBSL = first_three_lines_contains_BSL(glshader->source);
                    bool isBuiltInVariableConverted = is_glsl_builtin_converted(glshader->source);
                    remove_before_version(glshader->source);
                    char* convertedSource = glshader->source;
                    if (!isBuiltInVariableConverted)
                        convertedSource = ConvertShaderBuiltInVariableOnly(
                                convertedSource, glshader->type == GL_VERTEX_SHADER ? 1 : 0, &glshader->need,
                                isBuiltInVariableConverted ? 0 : 1);
                    free(glshader->source);
                    if (glshader->type == GL_FRAGMENT_SHADER) {
                        if (contains_glFragColor(glshader->source)) {
                            convertedSource = replace_glFragColor(convertedSource);
                            convertedSource = insert_gl_FragColor_if_missing(convertedSource);
                        } else {
                            int sourceLength = strlen(convertedSource) + 1;
                            convertedSource = ReplaceGLFragData(convertedSource, &sourceLength);
                        }
                    }
                    if (!isBuiltInVariableConverted) convertedSource = mark_glsl_builtin_converted(convertedSource);
                    glshader->source = strdup(convertedSource);
                    convertedSource = replace_version_line_460(convertedSource);
                    int returnCode = 0; // TODO: handle returnCode
                    char* result = GLSLtoGLSLES_c(convertedSource, glshader->type, globals4es.esversion, glsl_version,
                                                  &returnCode);
                    free(convertedSource);
                    // ZOMDROID FIX: same leak shape — both ternary arms already hand
                    // back an owned buffer; the strdup copy leaked it
                    glshader->converted =
                            result != NULL ? process_uniform_declarations(result, glshader->uniforms_declarations,
                                                                          &glshader->uniforms_declarations_count)
                                           : ConvertShaderConditionally(glshader);
                    glshader->converted = process_uniform_declarations(
                            glshader->converted, glshader->uniforms_declarations, &glshader->uniforms_declarations_count);

                    /*if (isBSL)*/ glshader->converted = bsl_patch(glshader->converted);
                    glshader->is_converted_essl_320 = 0;
                } else {
                    int returnCode = 0; // TODO: handle returnCode
                    char* result = GLSLtoGLSLES_c(glshader->source, glshader->type, globals4es.esversion, glsl_version,
                                                  &returnCode);
                    // ZOMDROID FIX: same leak shape as above
                    glshader->converted =
                            result != NULL ? process_uniform_declarations(result, glshader->uniforms_declarations,
                                                                          &glshader->uniforms_declarations_count)
                                           : ConvertShaderConditionally(glshader);
                    glshader->is_converted_essl_320 = 1;
                }
            }

            DBG(SHUT_LOGD("\n[INFO] [Shader] Converted Shader source: \n%s", glshader->converted))


            // ===================================================================
            // === BEGIN ULTIMATE FIXUP PATCH v10 (The Final Stroke)
            // ===================================================================
            if (glshader->converted && (glshader->type == GL_VERTEX_SHADER || glshader->type == GL_FRAGMENT_SHADER)) {
                //SHUT_LOGD("LIBGL: Executing ULTIMATE FIXUP PATCH v10 on Shader ID %d (Type: %s)\n", glshader->id, (glshader->type == GL_VERTEX_SHADER) ? "Vertex" : "Fragment");

                // Part 1: Force the correct GLSL ES version for BOTH shaders
                char target_version[32];
                if (globals4es.esversion >= 320) sprintf(target_version, "#version 320 es");
                else if (globals4es.esversion >= 310) sprintf(target_version, "#version 310 es");
                else sprintf(target_version, "#version 300 es");

                char* temp_v = force_replace_version(glshader->converted, target_version);
                if (temp_v) { free(glshader->converted); glshader->converted = temp_v; }

                // Part 2: Replace old texture2D with modern texture() for BOTH shaders
                if (strstr(glshader->converted, "texture2D")) {
                    char* temp_tex = replace_all(glshader->converted, "texture2D", "texture");
                    if (temp_tex) { free(glshader->converted); glshader->converted = temp_tex; }
                }

                // Part 3: Handle keywords based on shader type
                if (glshader->type == GL_VERTEX_SHADER) {
                    // In Vertex Shaders, 'attribute' becomes 'in', and 'varying' becomes 'out'
                    if (strstr(glshader->converted, "attribute")) {
                        char* temp_ain = replace_all(glshader->converted, "attribute", "in");
                        if (temp_ain) { free(glshader->converted); glshader->converted = temp_ain; }
                    }
                    if (strstr(glshader->converted, "varying")) {
                        char* temp_vout = replace_all(glshader->converted, "varying", "out");
                        if (temp_vout) { free(glshader->converted); glshader->converted = temp_vout; }
                    }
                } else { // This is a GL_FRAGMENT_SHADER
                    // Part 3.5: Add the mandatory default precision for floats in Fragment Shaders
                    if (!strstr(glshader->converted, "precision highp float")) {
                        const char* precision_qualifier = "\nprecision highp float;\n";
                        char* temp_prec = insert_after_first(glshader->converted, "\n", precision_qualifier);
                        if (temp_prec) { free(glshader->converted); glshader->converted = temp_prec; }
                    }

                    // In Fragment Shaders, 'varying' becomes 'in'
                    if (strstr(glshader->converted, "varying")) {
                        char* temp_vin = replace_all(glshader->converted, "varying", "in");
                        if (temp_vin) { free(glshader->converted); glshader->converted = temp_vin; }
                    }

                    // Part 4: Handle fragment-specific outputs (gl_FragColor/Data)
                    const char* custom_out_var = "fragColor";
                    int needs_out_declaration = 0;

                    if (strstr(glshader->converted, "gl_FragColor")) {
                        char* temp_fc = replace_all(glshader->converted, "gl_FragColor", custom_out_var);
                        if (temp_fc) { free(glshader->converted); glshader->converted = temp_fc; needs_out_declaration = 1; }
                    }
                    if (strstr(glshader->converted, "gl_FragData[0]")) {
                        char* temp_fd = replace_all(glshader->converted, "gl_FragData[0]", custom_out_var);
                        if (temp_fd) { free(glshader->converted); glshader->converted = temp_fd; needs_out_declaration = 1; }
                    }

                    // If we need to declare our variable, do it smartly with precision
                    if (needs_out_declaration && !strstr(glshader->converted, "out vec4 fragColor")) {
                        const char* declaration = "\nout highp vec4 fragColor;\n";
                        char* temp_decl = insert_after_preamble(glshader->converted, declaration);
                        if (temp_decl) { free(glshader->converted); glshader->converted = temp_decl; }
                    }
                }
            }
            // ===================================================================
            // === END ULTIMATE FIXUP PATCH v10
            // ===================================================================
        }

        GLchar* finalSource = (glshader->converted) ? glshader->converted : glshader->source;
        char* tempSource = NULL;
        if (globals4es.es >= 3 && globals4es.esversion >= 300 && glshader->is_converted_essl_320) {
            char* esSource = replace_version_to_es(finalSource, globals4es.esversion);
            if (esSource) {
                tempSource = esSource;
                finalSource = esSource;
            }
        }
        const GLchar* sources[] = {finalSource};
        ZOMDROID_VDBG("ZOMDROID_DBG: shader=%d type=%d final source:\n%s\n", shader, glshader->type, finalSource);
        gles_glShaderSource(shader, 1, sources, NULL);
        if (tempSource) free(tempSource);

        errorGL();
    } else
        noerrorShim();
}

#define SUPER()                                                                                                        \
    GO(color)                                                                                                          \
    GO(secondary)                                                                                                      \
    GO(fogcoord)                                                                                                       \
    GO(texcoord)                                                                                                       \
    GO(normalmatrix)                                                                                                   \
    GO(mvmatrix)                                                                                                       \
    GO(mvpmatrix)                                                                                                      \
    GO(notexarray)                                                                                                     \
    GO(clean)                                                                                                          \
    GO(clipvertex)                                                                                                     \
    GO2(texs)

void accumShaderNeeds(GLuint shader, shaderconv_need_t* need) {
    CHECK_SHADER(void, shader)
    if (!glshader->converted) return;
#define GO(A)                                                                                                          \
    if (need->need_##A < glshader->need.need_##A) need->need_##A = glshader->need.need_##A;
#define GO2(A) need->need_##A |= glshader->need.need_##A;
    SUPER()
#undef GO
#undef GO2
}
int isShaderCompatible(GLuint shader, shaderconv_need_t* need) {
    CHECK_SHADER(int, shader)
    if (!glshader->converted) return 0;
#define GO(A)                                                                                                          \
    if (need->need_##A > glshader->need.need_##A) return 0;
#define GO2(A)                                                                                                         \
    if (need->need_##A & glshader->need.need_##A) return 0;
    SUPER()
#undef GO
#undef GO2
    return 1;
}
#undef SUPER

void redoShader(GLuint shader, shaderconv_need_t* need) {
    LOAD_GLES2(glShaderSource);
    if (!gles_glShaderSource) return;
    CHECK_SHADER(void, shader)
    if (!glshader->converted) return;
    // test, if no changes, no need to reconvert & recompile...
    if (memcmp(&glshader->need, need, sizeof(shaderconv_need_t)) == 0) return;
    free(glshader->converted);
    memcpy(&glshader->need, need, sizeof(shaderconv_need_t));
    if (is_direct_shader(glshader->source))
        glshader->converted = strdup(glshader->source);
    else {
        int glsl_version = getGLSLVersion(glshader->source);
        DBG(SHUT_LOGD("[INFO] [Shader] Shader source: "))
        DBG(SHUT_LOGD("%s", glshader->source))
        if (glsl_version < 150 || globals4es.esversion < 300) {
            // ZOMDROID FIX: stored-and-returned pointer — the strdup leaked it
            glshader->converted = ConvertShaderConditionally(glshader);
            glshader->is_converted_essl_320 = 0;
        } else {
            int returnCode = 0;
            char* result =
                    GLSLtoGLSLES_c(glshader->source, glshader->type, globals4es.esversion, glsl_version, &returnCode);
            // ZOMDROID FIX: same leak shape — both arms hand back an owned buffer
            glshader->converted =
                    result != NULL ? process_uniform_declarations(result, glshader->uniforms_declarations,
                                                                  &glshader->uniforms_declarations_count)
                                   : ConvertShaderConditionally(glshader);
            glshader->is_converted_essl_320 = 1;
        }
        DBG(SHUT_LOGD("\n[INFO] [Shader] Converted Shader source: \n%s", glshader->converted))
    }
    // send source to GLES2 hardware if any
    gles_glShaderSource(
            shader, 1, (const GLchar* const*)((glshader->converted) ? (&glshader->converted) : (&glshader->source)), NULL);
    // recompile...
    gl4es_glCompileShader(glshader->id);
}

void APIENTRY_GL4ES gl4es_glGetShaderSource(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* source) {
    DBG(SHUT_LOGD("glGetShaderSource(%d, %d, %p, %p)\n", shader, bufSize, length, source))
    // find shader
    CHECK_SHADER(void, shader)
    if (bufSize <= 0) {
        errorShim(GL_INVALID_OPERATION);
        return;
    }
    // if no source, then it's an empty string
    if (glshader->source == NULL) {
        noerrorShim();
        if (length) *length = 0;
        source[0] = '\0';
        return;
    }
    // copy concatenated sources
    GLsizei size = strlen(glshader->source);
    if (size + 1 > bufSize) size = bufSize - 1;
    strncpy(source, glshader->source, size);
    source[size] = '\0';
    if (length) *length = size;
    noerrorShim();
}

GLboolean APIENTRY_GL4ES gl4es_glIsShader(GLuint shader) {
    DBG(SHUT_LOGD("glIsShader(%d)\n", shader))
    // find shader
    shader_t* glshader = NULL;
    khint_t k;
    {
        khash_t(shaderlist)* shaders = glstate->glsl->shaders;
        k = kh_get(shaderlist, shaders, shader);
        if (k != kh_end(shaders)) glshader = kh_value(shaders, k);
    }
    return (glshader) ? GL_TRUE : GL_FALSE;
}

shader_t* getShader(GLuint shader) {
    khint_t k;
    {
        khash_t(shaderlist)* shaders = glstate->glsl->shaders;
        k = kh_get(shaderlist, shaders, shader);
        if (k != kh_end(shaders)) return kh_value(shaders, k);
    }
    return NULL;
}

static const char* GLES_NoGLSLSupport = "No Shader support with current backend";

void APIENTRY_GL4ES gl4es_glGetShaderInfoLog(GLuint shader, GLsizei maxLength, GLsizei* length, GLchar* infoLog) {
    DBG(SHUT_LOGD("glGetShaderInfoLog(%d, %d, %p, %p)\n", shader, maxLength, length, infoLog))
    // find shader
    CHECK_SHADER(void, shader)
    // if (maxLength <= 0) {
    //     errorShim(GL_INVALID_OPERATION);
    //     return;
    // }
    LOAD_GLES2(glGetShaderInfoLog);
    if (gles_glGetShaderInfoLog) {
        gles_glGetShaderInfoLog(glshader->id, maxLength, length, infoLog);
        errorGL();
    } else {
        strncpy(infoLog, GLES_NoGLSLSupport, maxLength);
        if (length) *length = strlen(infoLog);
    }
}

void APIENTRY_GL4ES gl4es_glGetShaderiv(GLuint shader, GLenum pname, GLint* params) {
    DBG(SHUT_LOGD("glGetShaderiv(%d, %s, %p)\n", shader, PrintEnum(pname), params))
    // find shader
    CHECK_SHADER(void, shader)
    LOAD_GLES2(glGetShaderiv);
    noerrorShim();
    switch (pname) {
        case GL_SHADER_TYPE:
            *params = glshader->type;
            break;
        case GL_DELETE_STATUS:
            *params = (glshader->deleted) ? GL_TRUE : GL_FALSE;
            break;
        case GL_COMPILE_STATUS:
            if (gles_glGetShaderiv) {
                gles_glGetShaderiv(glshader->id, pname, params);
                errorGL();
            } else {
                *params = GL_FALSE; // stub, compile always fail
            }
            break;
        case GL_INFO_LOG_LENGTH:
            if (gles_glGetShaderiv) {
                gles_glGetShaderiv(glshader->id, pname, params);
                errorGL();
            } else {
                *params = strlen(GLES_NoGLSLSupport); // stub, compile always fail
            }
            break;
        case GL_SHADER_SOURCE_LENGTH:
            if (glshader->source)
                *params = strlen(glshader->source) + 1;
            else
                *params = 0;
            break;
        default:
            errorShim(GL_INVALID_ENUM);
    }
}

void APIENTRY_GL4ES gl4es_glGetShaderPrecisionFormat(GLenum shaderType, GLenum precisionType, GLint* range,
                                                     GLint* precision) {
    LOAD_GLES2(glGetShaderPrecisionFormat);
    if (gles_glGetShaderPrecisionFormat) {
        gles_glGetShaderPrecisionFormat(shaderType, precisionType, range, precision);
        errorGL();
    } else {
        errorShim(GL_INVALID_ENUM);
    }
}

void APIENTRY_GL4ES gl4es_glShaderBinary(GLsizei count, const GLuint* shaders, GLenum binaryFormat, const void* binary,
                                         GLsizei length) {
    // TODO: check consistancy of "shaders" values
    LOAD_GLES2(glShaderBinary);
    if (gles_glShaderBinary) {
        gles_glShaderBinary(count, shaders, binaryFormat, binary, length);
        errorGL();
    } else {
        errorShim(GL_INVALID_ENUM);
    }
}

void APIENTRY_GL4ES gl4es_glReleaseShaderCompiler(void) {
    LOAD_GLES2(glReleaseShaderCompiler);
    if (gles_glReleaseShaderCompiler) {
        gles_glReleaseShaderCompiler();
        errorGL();
    } else
        noerrorShim();
}

// ========== GL_ARB_shader_objects ==============

AliasExport(GLuint, glCreateShader, , (GLenum shaderType));
AliasExport(void, glDeleteShader, , (GLuint shader));
AliasExport(void, glCompileShader, , (GLuint shader));
AliasExport(void, glShaderSource, , (GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length));
AliasExport(void, glGetShaderSource, , (GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* source));
AliasExport(GLboolean, glIsShader, , (GLuint shader));
AliasExport(void, glGetShaderInfoLog, , (GLuint shader, GLsizei maxLength, GLsizei* length, GLchar* infoLog));
AliasExport(void, glGetShaderiv, , (GLuint shader, GLenum pname, GLint* params));
AliasExport(void, glGetShaderPrecisionFormat, ,
            (GLenum shaderType, GLenum precisionType, GLint* range, GLint* precision));
AliasExport(void, glShaderBinary, ,
            (GLsizei count, const GLuint* shaders, GLenum binaryFormat, const void* binary, GLsizei length));
AliasExport_V(void, glReleaseShaderCompiler);

GLhandleARB APIENTRY_GL4ES gl4es_glCreateShaderObject(GLenum shaderType) {
    return gl4es_glCreateShader(shaderType);
}

AliasExport(GLhandleARB, glCreateShaderObject, ARB, (GLenum shaderType));
AliasExport(GLvoid, glShaderSource, ARB,
            (GLhandleARB shaderObj, GLsizei count, const GLcharARB** string, const GLint* length));
AliasExport(GLvoid, glCompileShader, ARB, (GLhandleARB shaderObj));
AliasExport(GLvoid, glGetShaderSource, ARB, (GLhandleARB obj, GLsizei maxLength, GLsizei* length, GLcharARB* source));