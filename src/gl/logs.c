#include "logs.h"
#include "init.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#if defined(ANDROID) && defined(USE_ANDROID_LOG)
#include <android/log.h>
#endif
//----------------------------------------------------------------------------
static const char * const log_prefix="LIBGL: ";
//----------------------------------------------------------------------------
void LogPrintf_NoPrefix(const char *fmt,...)
{
	va_list args;
	va_start(args,fmt);
	#if defined(ANDROID) && defined(USE_ANDROID_LOG)
	__android_log_vprint(ANDROID_LOG_INFO, "LIBGL", fmt, args);
	#else
	vprintf(fmt,args);
	#endif
	va_end(args);
}
//----------------------------------------------------------------------------
void LogFPrintf(FILE *fp,const char *fmt,...)
{
	#ifndef ANDROID
	fprintf(fp,log_prefix);
	#endif
	va_list args;
	va_start(args,fmt);
	#if defined(ANDROID) && defined(USE_ANDROID_LOG)
	// also on logcat
	__android_log_vprint(ANDROID_LOG_INFO, "LIBGL", fmt, args);
	#endif
	vfprintf(fp,fmt,args);
	va_end(args);
}
//----------------------------------------------------------------------------
void LogPrintf(const char *fmt,...)
{
	#ifndef ANDROID
	printf(log_prefix);
	#endif
	va_list args;
	va_start(args,fmt);
	#if defined(ANDROID) && defined(USE_ANDROID_LOG)
	__android_log_vprint(ANDROID_LOG_INFO, "LIBGL", fmt, args);
	#else
	vprintf(fmt,args);
	#endif
	va_end(args);
}
//----------------------------------------------------------------------------

// ZOMDROID GL TRACE: log native rejections in the uniform/program path to a flushed file.
#include <stdio.h>
void zomdroid_gltrace(const char* fmt, ...) {
    static FILE* f = NULL;
    static int init = 0;
    static int count = 0;
    if (count > 12000) return;
    if (!init) {
        init = 1;
        f = fopen("/data/data/com.pridroid/files/gl_trace.txt", "w");
        // ZOMDROID (Codex minimap audit): a flush per line turned load-time CONVERT
        // bursts into syscall storms. Buffer the file; flush only the lines whose
        // survival matters at crash time — everything else also mirrors to stderr,
        // which the launcher captures independently.
        if (f) setvbuf(f, NULL, _IOFBF, 16384);
    }
    if (f) {
        va_list args;
        va_start(args, fmt);
        vfprintf(f, fmt, args);
        va_end(args);
        fputc('\n', f);
        if (!strncmp(fmt, "INIT", 4) || !strncmp(fmt, "MEMSTAT", 7) || !strncmp(fmt, "EXIT", 4) ||
            !strncmp(fmt, "GLALLOC exit", 12) || !strncmp(fmt, "CCACHE", 6) || !strncmp(fmt, "FBO", 3) ||
            !strncmp(fmt, "DRAWMIX", 7))
            fflush(f);
    }
    // ZOMDROID DIAG: mirror to stderr — it lands in the console log that testers can
    // export from Zomdroid WITHOUT adb (and it works even if the file open failed).
    {
        va_list args2;
        va_start(args2, fmt);
        fprintf(stderr, "[NGG] ");
        vfprintf(stderr, fmt, args2);
        va_end(args2);
        fputc('\n', stderr);
        fflush(stderr);
    }
    count++;
}

// ZOMDROID MEMDIAG: shadow-copy accounting + process RSS/Swap ground truth.
// The 42.19 world-gen deaths are memory deaths (LMK reclaim 1.9GB rss + 2.1GB swap,
// then malloc-abort with 325MB swap left). ZINK survives the same stand, so the
// delta is OUR overhead — this instrument shows where the gigabytes sit.
#include <string.h>
static long zshadow_total = 0;       // bytes currently held in buffer shadow copies
static long zshadow_last_logged = 0;
void zomdroid_memstat_tag(const char* tag) {
    long rss_kb = -1, swap_kb = -1;
    FILE* f = fopen("/proc/self/status", "r");
    if (f) {
        char line[128];
        while (fgets(line, sizeof(line), f)) {
            if (!strncmp(line, "VmRSS:", 6)) rss_kb = atol(line + 6);
            else if (!strncmp(line, "VmSwap:", 7)) swap_kb = atol(line + 7);
        }
        fclose(f);
    }
    zomdroid_gltrace("MEMSTAT %s shadow=%ldMB rss=%ldMB swap=%ldMB", tag, zshadow_total / (1024 * 1024),
                     rss_kb / 1024, swap_kb / 1024);
}
// Step logging with hysteresis (up 32MB, down 64MB) and its own line budget, so a
// total oscillating on a boundary can never chat its way into frame time.
static int zshadow_step_budget = 20;
void zomdroid_shadow_add(long n) {
    zshadow_total += n;
    if (zshadow_step_budget > 0 && zshadow_total - zshadow_last_logged >= 32L * 1024 * 1024) {
        zshadow_last_logged = zshadow_total;
        zshadow_step_budget--;
        zomdroid_memstat_tag("shadow-step");
    }
}
void zomdroid_shadow_sub(long n) {
    zshadow_total -= n;
    if (zshadow_total < 0) zshadow_total = 0;
    if (zshadow_step_budget > 0 && zshadow_last_logged - zshadow_total >= 64L * 1024 * 1024) {
        zshadow_last_logged = zshadow_total;
        zshadow_step_budget--;
        zomdroid_memstat_tag("shadow-step");
    }
}

// ZOMDROID DIAG (stutter hunt 2026-07-29): driver uploads from the unmap emulation
// are the prime suspect for Adreno stalls (SubData into an in-flight buffer forces an
// internal sync; PZ mapped with UNSYNCHRONIZED and expects none). Sites report here
// when a single upload takes >= 3 ms; budget-capped.
void zomdroid_slowsub(const char* tag, long ms, long size, unsigned buf) {
    static int zbudget = 40;
    if (zbudget <= 0) return;
    zbudget--;
    zomdroid_gltrace("SLOWSUB %s ms=%ld size=%ld buf=%u", tag, ms, size, buf);
}

// ZOMDROID GPU-ALLOC DIAG (2026-07-30): dumpsys caught GL mtrack at 4.99GB seconds
// before the lmkd kill on the 42.20 world load (EGL sane, JVM heap capped, shadows
// 64MB) — the killing mass is driver-side GL objects. This ledger tracks what the
// app REQUESTS through our entry points, per category and object id. live vs cum
// separates a leak (live grows, nothing deleted) from churn (live low but cum huge
// = respecification ghosts held by the driver).
#define ZGA_CATS 4 // 0=texture 1=buffer 2=renderbuffer 3=stream-scratch (cum only)
#define ZGA_CAP (1 << 18)
typedef struct {
    unsigned long long key; // (cat+1)<<32 | object id; 0 = empty slot
    long bytes;
} zga_ent_t;
static zga_ent_t* zga_tab = NULL;
static long zga_live[ZGA_CATS], zga_cum[ZGA_CATS], zga_cnt[ZGA_CATS];
static long zga_last_logged = 0;
static zga_ent_t* zga_slot(unsigned long long key) {
    if (!zga_tab) {
        zga_tab = (zga_ent_t*)calloc(ZGA_CAP, sizeof(zga_ent_t));
        if (!zga_tab) return NULL;
    }
    unsigned h = (unsigned)((key * 2654435761ull) >> 12) & (ZGA_CAP - 1);
    for (unsigned i = 0; i < ZGA_CAP; i++) {
        zga_ent_t* e = &zga_tab[(h + i) & (ZGA_CAP - 1)];
        if (e->key == key || e->key == 0) {
            e->key = key;
            return e;
        }
    }
    return NULL; // table full — stop accounting rather than loop
}
static void zga_report(const char* tag) {
    zomdroid_gltrace("GLALLOC %s live tex=%ldMB/%ld buf=%ldMB/%ld rb=%ldMB/%ld | cum tex=%ldMB buf=%ldMB rb=%ldMB scratch=%ldMB",
                     tag, zga_live[0] >> 20, zga_cnt[0], zga_live[1] >> 20, zga_cnt[1], zga_live[2] >> 20, zga_cnt[2],
                     zga_cum[0] >> 20, zga_cum[1] >> 20, zga_cum[2] >> 20, zga_cum[3] >> 20);
}
// step logging on 128MB movement of the explosion metric (live totals + scratch churn)
static void zga_check(void) {
    long total = zga_live[0] + zga_live[1] + zga_live[2] + zga_cum[3];
    long delta = total - zga_last_logged;
    if (delta < 0) delta = -delta;
    if (delta >= 128L * 1024 * 1024) {
        static int zbudget = 60;
        zga_last_logged = total;
        if (zbudget <= 0) return;
        zbudget--;
        zga_report("step");
    }
}
void zomdroid_glalloc_set(int cat, unsigned id, long bytes) {
    if (cat < 0 || cat >= ZGA_CATS || id == 0 || bytes < 0) return;
    zga_ent_t* e = zga_slot(((unsigned long long)(cat + 1) << 32) | id);
    if (!e) return;
    if (e->bytes == 0 && bytes > 0) zga_cnt[cat]++;
    if (e->bytes > 0 && bytes == 0) zga_cnt[cat]--;
    zga_live[cat] += bytes - e->bytes;
    zga_cum[cat] += bytes;
    e->bytes = bytes;
    zga_check();
}
void zomdroid_glalloc_del(int cat, unsigned id) {
    if (cat < 0 || cat >= ZGA_CATS || id == 0 || !zga_tab) return;
    zga_ent_t* e = zga_slot(((unsigned long long)(cat + 1) << 32) | id);
    if (e && e->bytes > 0) {
        zga_live[cat] -= e->bytes;
        zga_cnt[cat]--;
        e->bytes = 0;
    }
    zga_check();
}
// live texture bytes — the adaptive texture policy in texture.c reads this
long zomdroid_glalloc_live_tex(void) {
    return zga_live[0];
}
void zomdroid_glalloc_cum(int cat, long bytes) {
    if (cat < 0 || cat >= ZGA_CATS || bytes <= 0) return;
    zga_cum[cat] += bytes;
    zga_check();
}

// ZOMDROID DIAG: distinguishes "someone called exit()" from a signal kill — on the Mali
// tester the process dies at Bullet.init with NO signal traces anywhere; if this line
// shows up in the log, the death is a plain exit() inside emulated code (box64 territory).
static void zomdroid_exit_probe(void) {
    extern int zccache_hits, zccache_miss;
    extern int zmemo_hits, zmemo_entries;
    extern long zmemo_bytes;
    extern long zomdroid_fbo_bind_app, zomdroid_fbo_bind_native, zomdroid_fbo_bind_skip, zomdroid_clearbuf_native;
    zomdroid_memstat_tag("exit");
    zga_report("exit");
    zomdroid_gltrace("FBO session: binds app=%ld native=%ld skipped=%ld clearbuf-native=%ld", zomdroid_fbo_bind_app,
                     zomdroid_fbo_bind_native, zomdroid_fbo_bind_skip, zomdroid_clearbuf_native);
    {
        extern long zdrawmix_total, zdrawmix_quads, zdrawmix_ebo, zdrawmix_direct, zdrawmix_inst;
        extern long zebo_used, zebo_try_de, zebo_try_dre;
        zomdroid_gltrace("DRAWMIX session: draws=%ld quads=%ld ebo=%ld direct-ok=%ld inst=%ld direct-taken=%ld"
                         " via-DrawElements=%ld via-RangeElements=%ld",
                         zdrawmix_total, zdrawmix_quads, zdrawmix_ebo, zdrawmix_direct, zdrawmix_inst, zebo_used,
                         zebo_try_de, zebo_try_dre);
    }
    {
        extern long zomdroid_etc2_n, zomdroid_etc2_hits, zomdroid_etc2_evicted;
        extern unsigned long long zomdroid_etc2_total_ms(void);
        extern unsigned long long zomdroid_etc2_io_ms(void);
        if (zomdroid_etc2_n || zomdroid_etc2_hits)
            zomdroid_gltrace("ETC2 session: encodes=%ld cache-hits=%ld evicted=%ld encode-ms=%llu io-ms=%llu",
                             zomdroid_etc2_n, zomdroid_etc2_hits, zomdroid_etc2_evicted,
                             zomdroid_etc2_total_ms(), zomdroid_etc2_io_ms());
    }
    zomdroid_gltrace("CCACHE session: hits=%d miss=%d memo-hits=%d memo-entries=%d memo-kb=%ld", zccache_hits,
                     zccache_miss, zmemo_hits, zmemo_entries, zmemo_bytes / 1024);
    zomdroid_gltrace("EXIT-PROBE: process exiting via exit(), not a signal kill");
}
void zomdroid_exit_probe_register(void) {
    static int done = 0;
    if (!done) {
        done = 1;
        atexit(zomdroid_exit_probe);
    }
}

