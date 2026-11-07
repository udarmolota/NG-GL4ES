#include "logs.h"
#include "texture.h"
#include "zomdroid_etc2.h"

#include "../glx/hardext.h"
#include "../glx/streaming.h"
#include "array.h"
#include "blit.h"
#include "decompress.h"
#include "debug.h"
#include "enum_info.h"
#include "fpe.h"
#include "framebuffers.h"
#include "gles.h"
#include "init.h"
#include "loader.h"
#include "matrix.h"
#include "pixel.h"
#include "raster.h"

// #define DEBUG
#ifdef DEBUG
#define DBG(a) a
#else
#define DBG(a)
#endif

#ifndef GL_TEXTURE_STREAM_IMG
#define GL_TEXTURE_STREAM_IMG 0x8C0D
#endif
#ifdef TEXSTREAM
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

// expand non-power-of-two sizes
// TODO: what does this do to repeating textures?
int npot(int n) {
    if (n == 0) return 0;

    int i = 1;
    while (i < n)
        i <<= 1;
    return i;
}

static int inline nlevel(int size, int level) {
    if (size) {
        size >>= level;
        if (!size) size = 1;
    }
    return size;
}

// return the max level for that WxH size
static int inline maxlevel(int w, int h) {
    int mlevel = 0;
    while (w != 1 || h != 1) {
        w >>= 1;
        h >>= 1;
        if (!w) w = 1;
        if (!h) h = 1;
        ++mlevel;
    }
    return mlevel;
}

static int is_fake_compressed_rgb(GLenum internalformat) {
    if (internalformat == GL_COMPRESSED_RGB) return 1;
    if (internalformat == GL_COMPRESSED_RGB_S3TC_DXT1_EXT) return 1;
    if (internalformat == GL_COMPRESSED_SRGB_S3TC_DXT1_EXT) return 1;
    return 0;
}
static int is_fake_compressed_rgba(GLenum internalformat) {
    if (internalformat == GL_COMPRESSED_RGBA) return 1;
    if (internalformat == GL_COMPRESSED_RGBA_S3TC_DXT1_EXT) return 1;
    if (internalformat == GL_COMPRESSED_RGBA_S3TC_DXT3_EXT) return 1;
    if (internalformat == GL_COMPRESSED_RGBA_S3TC_DXT5_EXT) return 1;
    if (internalformat == GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT) return 1;
    if (internalformat == GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT) return 1;
    if (internalformat == GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT) return 1;
    return 0;
}

void internal2format_type(GLenum* internalformat, GLenum* format, GLenum* type) {
    DBG(char log_buffer[512]; int offset = snprintf(log_buffer, sizeof(log_buffer), "tex format converting... ");
        if (internalformat) offset +=
        snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "internalFormat: %s", PrintEnum(*internalformat));
        if (format) offset +=
        snprintf(log_buffer + offset, sizeof(log_buffer) - offset, ", format: %s", PrintEnum(*format));
        if (type) offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, ", type: %s", PrintEnum(*type));
        snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "\n"); SHUT_LOGD("%s", log_buffer))
    switch (*internalformat) {
    case GL_RED:
    case GL_R8:
    case GL_R:
        if (!hardext.rgtex) {
            *format = GL_RGB;
            *type = GL_UNSIGNED_BYTE;
        } else {
            *format = GL_RED;
            *type = GL_UNSIGNED_BYTE;
        }
        break;
    case GL_RG:
        if (!hardext.rgtex) {
            *format = GL_RGB;
            *type = GL_UNSIGNED_BYTE;
        } else {
            *format = GL_RG;
            *type = GL_UNSIGNED_BYTE;
        }
        break;
    case GL_COMPRESSED_ALPHA:
    case GL_ALPHA:
        *format = GL_ALPHA;
        *type = GL_UNSIGNED_BYTE;
        break;
    case 1: // is this here or with GL_RED?
    case GL_COMPRESSED_LUMINANCE:
    case GL_LUMINANCE:
        *format = GL_LUMINANCE;
        *type = GL_UNSIGNED_BYTE;
        break;
    case 2:
    case GL_COMPRESSED_LUMINANCE_ALPHA:
    case GL_LUMINANCE8_ALPHA8:
    case GL_LUMINANCE_ALPHA:
        if (globals4es.nolumalpha) {
            *format = GL_RGBA;
            *type = GL_UNSIGNED_BYTE;
        } else {
            *format = GL_LUMINANCE_ALPHA;
            *type = GL_UNSIGNED_BYTE;
        }
        break;
    case GL_RGB5:
    case GL_RGB565:
        *format = GL_RGB;
        *type = GL_UNSIGNED_SHORT_5_6_5;
        break;
    case GL_RGB:
        if (globals4es.avoid24bits)
            *format = GL_RGBA;
        else
            *format = GL_RGB;
        *type = GL_UNSIGNED_BYTE;
        break;
    case GL_RGB5_A1:
        *format = GL_RGBA;
        *type = GL_UNSIGNED_SHORT_5_5_5_1;
        break;
    case GL_RGBA4:
        *format = GL_RGBA;
        *type = GL_UNSIGNED_SHORT_4_4_4_4;
        break;
    case GL_RGBA:
        *format = GL_RGBA;
        *type = GL_UNSIGNED_BYTE;
        break;
    case GL_BGRA:
        if (hardext.bgra8888)
            *format = GL_BGRA;
        else
            *format = GL_RGBA;
        *type = GL_UNSIGNED_BYTE;
        break;
    case GL_DEPTH_COMPONENT16:
        if (type) *type = GL_UNSIGNED_SHORT;
        break;

    case GL_DEPTH_COMPONENT24:
        if (type) *type = GL_UNSIGNED_INT;
        break;

    case GL_DEPTH_COMPONENT32:
        *internalformat = GL_DEPTH_COMPONENT;
        if (type) *type = GL_UNSIGNED_INT;
        break;

    case GL_DEPTH_COMPONENT32F:
        if (type) *type = GL_UNSIGNED_INT;
        break;
    case GL_DEPTH_COMPONENT:
        *format = GL_DEPTH_COMPONENT;
        *type = GL_UNSIGNED_INT;
        break;
    case GL_DEPTH_STENCIL:
    case GL_DEPTH24_STENCIL8:
        *format = GL_DEPTH_STENCIL;
        *type = GL_UNSIGNED_INT_24_8;
        break;
    case GL_RGBA16F:
        *format = GL_RGBA;
        *type = (hardext.halffloattex) ? GL_HALF_FLOAT_OES : GL_UNSIGNED_BYTE;
        break;
    case GL_RGBA32F:
        *format = GL_RGBA;
        *type = (hardext.floattex) ? GL_FLOAT : GL_UNSIGNED_BYTE;
        break;
    case GL_RGB16F:
        *format = GL_RGB;
        *type = (hardext.halffloattex) ? GL_HALF_FLOAT_OES : GL_UNSIGNED_BYTE;
        break;
    case GL_RGB32F:
        *format = GL_RGB;
        *type = (hardext.floattex) ? GL_FLOAT : GL_UNSIGNED_BYTE;
        break;
    default:
        DBG(SHUT_LOGE("LIBGL: Warning, unknown Internalformat (%s)\n", PrintEnum(*internalformat)));
        *format = GL_RGBA;
        *type = GL_UNSIGNED_BYTE;
        break;
    }
    DBG(char log_buffer2[512]; int offset2 = snprintf(log_buffer, sizeof(log_buffer), "converted: ");
        if (internalformat) offset2 +=
        snprintf(log_buffer + offset2, sizeof(log_buffer) - offset2, "internalFormat: %s", PrintEnum(*internalformat));
        if (format) offset2 +=
        snprintf(log_buffer + offset2, sizeof(log_buffer) - offset2, ", format: %s", PrintEnum(*format));
        if (type) offset2 +=
        snprintf(log_buffer + offset2, sizeof(log_buffer) - offset2, ", type: %s", PrintEnum(*type));
        snprintf(log_buffer2 + offset2, sizeof(log_buffer2) - offset2, "\n"); SHUT_LOGD("%s", log_buffer))
}

static void* swizzle_texture(GLsizei width, GLsizei height, GLenum* format, GLenum* type, GLenum intermediaryformat,
                             GLenum internalformat, const GLvoid* data, gltexture_t* bound) {
    int convert = 0;
    GLenum dest_format = GL_RGBA;
    GLenum dest_type = GL_UNSIGNED_BYTE;
    int check = 1;
    // compressed format are not handled here, so mask them....
    if (is_fake_compressed_rgb(intermediaryformat)) intermediaryformat = GL_RGB;
    if (is_fake_compressed_rgba(intermediaryformat)) intermediaryformat = GL_RGBA;
    if (is_fake_compressed_rgb(internalformat)) internalformat = GL_RGB;
    if (is_fake_compressed_rgba(internalformat)) internalformat = GL_RGBA;
    if (intermediaryformat == GL_COMPRESSED_LUMINANCE) intermediaryformat = GL_LUMINANCE;
    if (internalformat == GL_COMPRESSED_LUMINANCE) internalformat = GL_LUMINANCE;

    // if (*format != intermediaryformat || intermediaryformat != internalformat) {
    //     internal2format_type(&intermediaryformat, &dest_format, &dest_type);
    //     convert = 1;
    //     check = 0;
    // }

    else {
        if ((*type) == GL_HALF_FLOAT) (*type) = GL_HALF_FLOAT_OES; // the define is different between GL and GLES...
        switch (*format) {
        case GL_R:
        case GL_RED:
            if (!hardext.rgtex) {
                dest_format = GL_RGB;
                convert = 1;
            } else
                dest_format = GL_RED;
            break;
        case GL_RG:
            if (!hardext.rgtex) {
                dest_format = GL_RGB;
                convert = 1;
            } else
                dest_format = GL_RG;
            break;
        case GL_COMPRESSED_LUMINANCE:
            *format = GL_LUMINANCE;
        case GL_LUMINANCE:
            dest_format = GL_LUMINANCE;
            break;
        case GL_LUMINANCE16F:
            dest_format = GL_LUMINANCE;
            if (hardext.halffloattex) {
                dest_type = GL_HALF_FLOAT_OES;
                check = 0;
            }
            break;
        case GL_LUMINANCE32F:
            dest_format = GL_LUMINANCE;
            if (hardext.floattex) {
                dest_type = GL_FLOAT;
                check = 0;
            }
            break;
        case GL_RGB:
            dest_format = GL_RGB;
            break;
        case GL_COMPRESSED_ALPHA:
            *format = GL_ALPHA;
        case GL_ALPHA:
            dest_format = GL_ALPHA;
            break;
        case GL_ALPHA16F:
            dest_format = GL_ALPHA;
            if (hardext.halffloattex) {
                dest_type = GL_HALF_FLOAT_OES;
                check = 0;
            }
            break;
        case GL_ALPHA32F:
            dest_format = GL_ALPHA;
            if (hardext.floattex) {
                dest_type = GL_FLOAT;
                check = 0;
            }
            break;
        case GL_RGBA:
            break;
        case GL_LUMINANCE8_ALPHA8:
        case GL_COMPRESSED_LUMINANCE_ALPHA:
            if (globals4es.nolumalpha)
                convert = 1;
            else {
                dest_format = GL_LUMINANCE_ALPHA;
                *format = GL_LUMINANCE_ALPHA;
            }
            break;
        case GL_LUMINANCE_ALPHA:
            if (globals4es.nolumalpha)
                convert = 1;
            else
                dest_format = GL_LUMINANCE_ALPHA;
            break;
        case GL_LUMINANCE_ALPHA16F:
            if (globals4es.nolumalpha)
                convert = 1;
            else
                dest_format = GL_LUMINANCE_ALPHA;
            if (hardext.halffloattex) {
                dest_type = GL_HALF_FLOAT_OES;
                check = 0;
            }
            break;
        case GL_LUMINANCE_ALPHA32F:
            if (globals4es.nolumalpha)
                convert = 1;
            else
                dest_format = GL_LUMINANCE_ALPHA;
            if (hardext.floattex) {
                dest_type = GL_FLOAT;
                check = 0;
            }
            break;
            // vvvvv all this are internal formats, so it should not happens
        case GL_RGB5:
        case GL_RGB565:
            dest_format = GL_RGB;
            dest_type = GL_UNSIGNED_SHORT_5_6_5;
            convert = 1;
            check = 0;
            break;
        case GL_RGB8:
            dest_format = GL_RGB;
            *format = GL_RGB;
            break;
        case GL_RGBA4:
            dest_format = GL_RGBA;
            dest_type = GL_UNSIGNED_SHORT_4_4_4_4;
            *format = GL_RGBA;
            check = 0;
            break;
        case GL_RGBA8:
            dest_format = GL_RGBA;
            *format = GL_RGBA;
            break;
        case GL_BGRA:
            if (hardext.bgra8888 && ((*type) == GL_UNSIGNED_BYTE || (*type) == GL_FLOAT || (*type) == GL_HALF_FLOAT ||
#ifdef __BIG_ENDIAN__
                                     (((*type) == GL_UNSIGNED_INT_8_8_8_8_REV) && hardext.rgba8888rev)
#else
                                     (((*type) == GL_UNSIGNED_INT_8_8_8_8) && hardext.rgba8888)
#endif
                                         )) {
                dest_format = GL_BGRA;
                //*format = GL_BGRA;
            } else {
                convert = 1;
                if (hardext.bgra8888 &&
#ifdef __BIG_ENDIAN__
                    (*type == GL_UNSIGNED_INT_8_8_8_8_REV)
#else
                    (*type == GL_UNSIGNED_INT_8_8_8_8)
#endif
                ) {
                    //*format = GL_BGRA;    //only type needs conversion
                    dest_format = GL_BGRA;
                    check = 0;
                }
            }
            break;
        case GL_DEPTH32F_STENCIL8:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH_STENCIL:
            // if (hardext.depthtex && hardext.depthstencil) {
            const int is32F = *format == GL_DEPTH32F_STENCIL8;
            *format = dest_format = GL_DEPTH_STENCIL;
            dest_type = is32F ? GL_FLOAT_32_UNSIGNED_INT_24_8_REV : GL_UNSIGNED_INT_24_8;
            //   check = 0;
            //}
            // else convert = 1;
            break;
        case GL_DEPTH_COMPONENT:
            // if (hardext.depthtex) {
            *format = dest_format = GL_DEPTH_COMPONENT;
            // if (dest_type != GL_UNSIGNED_INT) {
            //     convert = 1;
            // }
            dest_type = GL_UNSIGNED_INT;
            //    check = 0;
            //}
            // else
            //    convert = 1;
            break;
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            // if (hardext.depthtex) {
            //     if (dest_type == GL_UNSIGNED_BYTE) {
            *format = dest_format = GL_DEPTH_COMPONENT;
            dest_type = GL_UNSIGNED_INT;
            //       convert = 1;
            //    }
            //    check = 0;
            //}
            // else
            //    convert = 1;
            break;
        case GL_STENCIL_INDEX8:
            if (hardext.stenciltex)
                *format = dest_format = GL_STENCIL_INDEX8;
            else
                convert = 1;
            break;
        default:
            convert = 1;
            break;
        }
        if (check) switch (*type) {
            case GL_UNSIGNED_SHORT_4_4_4_4_REV:
                if (dest_format == GL_RGBA) dest_type = GL_UNSIGNED_SHORT_4_4_4_4;
                convert = 1;
                break;
            case GL_UNSIGNED_SHORT_4_4_4_4:
                if (dest_format == GL_RGBA)
                    dest_type = GL_UNSIGNED_SHORT_4_4_4_4;
                else
                    convert = 1;
                break;
            case GL_UNSIGNED_SHORT_1_5_5_5_REV:
                if (!hardext.rgba1555rev) {
                    if (dest_format == GL_RGBA) dest_type = GL_UNSIGNED_SHORT_5_5_5_1;
                    convert = 1;
                }
                break;
            case GL_UNSIGNED_SHORT_5_5_5_1:
                if (dest_format == GL_RGBA)
                    dest_type = GL_UNSIGNED_SHORT_5_5_5_1;
                else
                    convert = 1;
                break;
            case GL_UNSIGNED_SHORT_5_6_5_REV:
                if (dest_format == GL_RGB) dest_type = GL_UNSIGNED_SHORT_5_6_5;
                convert = 1;
                break;
            case GL_UNSIGNED_SHORT_5_6_5:
                if (dest_format == GL_RGB)
                    dest_type = GL_UNSIGNED_SHORT_5_6_5;
                else
                    convert = 1;
                break;
#ifdef __BIG_ENDIAN__
            case GL_UNSIGNED_INT_8_8_8_8:
#else
            case GL_UNSIGNED_INT_8_8_8_8_REV:
#endif
                *type = GL_UNSIGNED_BYTE;
                // fall through
            case GL_UNSIGNED_BYTE:
                if (dest_format == GL_RGB && globals4es.avoid24bits) {
                    dest_format = GL_RGBA;
                    convert = 1;
                }
                break;
#ifdef __BIG_ENDIAN__
            case GL_UNSIGNED_INT_8_8_8_8_REV:
                if (!hardext.rgba8888rev) {
                    dest_type = GL_UNSIGNED_BYTE;
                    convert = 1;
                }
                break;
#else
            case GL_UNSIGNED_INT_8_8_8_8:
                if (!hardext.rgba8888) {
                    dest_type = GL_UNSIGNED_BYTE;
                    convert = 1;
                }
                break;
#endif
            case GL_UNSIGNED_INT_24_8:
                if (hardext.depthtex && hardext.depthstencil) {
                    dest_type = GL_UNSIGNED_INT_24_8;
                } else {
                    *type = GL_UNSIGNED_BYTE; // will probably do nothing good!
                    convert = 1;
                }
                break;
            case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
                if (hardext.floattex && hardext.depthstencil) {
                    dest_type = GL_FLOAT_32_UNSIGNED_INT_24_8_REV;
                } else {
                    *type = GL_UNSIGNED_BYTE; // will probably do nothing good!
                    convert = 1;
                }
                break;
            case GL_FLOAT:
                if (hardext.floattex)
                    dest_type = GL_FLOAT;
                else
                    convert = 1;
                break;
            case GL_HALF_FLOAT:
            case GL_HALF_FLOAT_OES:
                if (hardext.halffloattex)
                    dest_type = GL_HALF_FLOAT_OES;
                else
                    convert = 1;
                break;
            default:
                convert = 1;
                break;
            }
    }
    if (data) {
        if (convert) {
            GLvoid* pixels = (GLvoid*)data;
            bound->inter_format = dest_format;
            bound->format = dest_format;
            bound->inter_type = dest_type;
            bound->type = dest_type;
            if (!pixel_convert(data, &pixels, width, height, *format, *type, dest_format, dest_type, 0,
                               glstate->texture.unpack_align)) {
                SHUT_LOGD("LIBGL: swizzle error: (%s, %s -> %s, %s)\n", PrintEnum(*format), PrintEnum(*type),
                          PrintEnum(dest_format), PrintEnum(dest_type));
                return NULL;
            }
            *type = dest_type;
            *format = dest_format;
            if (dest_format != internalformat) {
                GLvoid* pix2 = (GLvoid*)pixels;
                internal2format_type(&internalformat, &dest_format, &dest_type);
                bound->format = dest_format;
                bound->type = dest_type;
                if (!pixel_convert(pixels, &pix2, width, height, *format, *type, dest_format, dest_type, 0,
                                   glstate->texture.unpack_align)) {
                    SHUT_LOGD("LIBGL: swizzle error: (%s, %s -> %s, %s)\n", PrintEnum(dest_format),
                              PrintEnum(dest_type), PrintEnum(internalformat), PrintEnum(dest_type));
                    return NULL;
                }
                if (pix2 != pixels) {
                    if (pixels != data) free(pixels);
                    pixels = pix2;
                }
                *type = dest_type;
                *format = dest_format;
            }
            GLvoid* pix2 = pixels;
            if (raster_need_transform())
                if (!pixel_transform(data, &pixels, width, height, *format, *type, glstate->raster.raster_scale,
                                     glstate->raster.raster_bias)) {
                    SHUT_LOGD("LIBGL: swizzle/convert error: (%s, %s -> %s, %s)\n", PrintEnum(*format),
                              PrintEnum(*type), PrintEnum(dest_format), PrintEnum(dest_type));
                    pix2 = pixels;
                }
            if (pix2 != pixels && pixels != data) free(pixels);
            return pix2;
        } else {
            bound->inter_format = dest_format;
            bound->format = dest_format;
            bound->inter_type = dest_type;
            bound->type = dest_type;
        }
    } else {
        bound->inter_format = dest_format;
        bound->inter_type = dest_type;
        if (convert) {
            internal2format_type(&internalformat, &dest_format, &dest_type); // in case they are differents
            *type = dest_type;
            *format = dest_format;
        }
        bound->format = dest_format;
        bound->type = dest_type;
    }
    return (void*)data;
}

GLenum swizzle_internalformat(GLenum* internalformat, GLenum format, GLenum type) {
    GLenum ret = *internalformat;
    GLenum sret;
    switch (*internalformat) {
    case GL_RED:
    case GL_R:
    case GL_R8:
        if (!hardext.rgtex) {
            ret = GL_RGB;
            sret = GL_RGB;
        } else
            sret = GL_RED;
        break;
    case GL_R32F:
        ret = sret = GL_R32F;
        break;
    case GL_RGB10_A2:
        ret = sret = GL_RGB10_A2;
        break;
    case GL_RG:
        if (!hardext.rgtex) {
            ret = GL_RGB;
            sret = GL_RGB;
        } else
            sret = GL_RG;
        break;
    case GL_RGB565:
        ret = GL_RGB5;
    case GL_RGB5:
        sret = GL_RGB5;
        break;
    case GL_RGB:
        if (globals4es.avoid16bits == 0 && format == GL_RGB && type == GL_UNSIGNED_SHORT_5_6_5) {
            sret = ret = GL_RGB5;
            break;
        }
    case GL_RGB8:
    case GL_BGR:
    case GL_RGB16:
    case GL_RGB16F:
    case GL_RGB32F:
    case 3:
        ret = GL_RGB;
        sret = GL_RGB;
        break;
    case GL_RGBA4:
        sret = GL_RGBA4;
        break;
    case GL_RGB5_A1:
        sret = GL_RGB5_A1;
        break;
    case GL_RGBA:
        if (globals4es.avoid16bits == 0 && format == GL_RGBA && type == GL_UNSIGNED_SHORT_5_5_5_1) {
            sret = ret = GL_RGB5_A1;
            break;
        }
        if (globals4es.avoid16bits == 0 && format == GL_RGBA && type == GL_UNSIGNED_SHORT_4_4_4_4) {
            sret = ret = GL_RGBA4;
            break;
        }
        if (format == GL_BGRA && hardext.bgra8888) {
            sret = ret = GL_BGRA;
        }
    case GL_RGBA8:
    case GL_RGBA16:
    case GL_RGBA16F:
    case GL_RGBA32F:
    case 4:
        if (format == GL_BGRA && hardext.bgra8888) {
            ret = GL_BGRA;
            sret = GL_BGRA;
        } else {
            ret = GL_RGBA;
            sret = GL_RGBA;
        }
        break;
    case GL_ALPHA32F:
    case GL_ALPHA16F:
    case GL_ALPHA8:
    case GL_ALPHA:
        ret = GL_ALPHA;
        sret = GL_ALPHA;
        break;
    case 1:
    case GL_LUMINANCE32F:
    case GL_LUMINANCE16F:
    case GL_LUMINANCE8:
    case GL_LUMINANCE16:
    case GL_LUMINANCE:
        if (format == GL_RED && hardext.rgtex) {
            ret = GL_RED;
            sret = GL_RED;
        } else {
            ret = GL_LUMINANCE;
            sret = GL_LUMINANCE;
        }
        break;
    case 2:
    case GL_LUMINANCE4_ALPHA4:
    case GL_LUMINANCE8_ALPHA8:
    case GL_LUMINANCE16_ALPHA16:
    case GL_LUMINANCE_ALPHA32F:
    case GL_LUMINANCE_ALPHA16F:
    case GL_LUMINANCE_ALPHA:
        ret = GL_LUMINANCE_ALPHA;
        if (globals4es.nolumalpha)
            sret = GL_RGBA;
        else
            sret = GL_LUMINANCE_ALPHA;
        break;
        // compressed format...
    case GL_COMPRESSED_ALPHA:
        ret = GL_ALPHA; // GL_COMPRESSED_RGBA;
        sret = GL_ALPHA;
        break;
    case GL_COMPRESSED_LUMINANCE:
        ret = GL_LUMINANCE; // GL_COMPRESSED_RGB;
        sret = GL_LUMINANCE;
        break;
    case GL_COMPRESSED_LUMINANCE_ALPHA:
        if (globals4es.nolumalpha) {
            ret = GL_COMPRESSED_RGBA;
            sret = GL_RGBA;
        } else {
            ret = GL_LUMINANCE_ALPHA;
            sret = GL_LUMINANCE_ALPHA;
        }
        break;
    case GL_COMPRESSED_RGB:
        // ZOMDROID FIX (restored after ladder-A exonerated it): ret was not set, so
        // *internalformat stayed GL_COMPRESSED_RGB, which GLES rejects in glTexImage2D
        // (GL_INVALID_VALUE). Map to GL_RGB.
        ret = GL_RGB;
        sret = GL_RGB;
        break;
    case GL_COMPRESSED_RGBA:
        // ZOMDROID FIX: same as above -> GL_RGBA.
        ret = GL_RGBA;
        sret = GL_RGBA;
        break;
    case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
    case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT: // should be sRGB...
        ret = GL_COMPRESSED_RGB;
        sret = GL_RGB;
        break;
    case GL_COMPRESSED_RGBA_S3TC_DXT1_EXT: // not good...
    case GL_COMPRESSED_RGBA_S3TC_DXT3_EXT: // not good, but there is no DXT3 compressor
    case GL_COMPRESSED_RGBA_S3TC_DXT5_EXT:
    case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT:
    case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT:
    case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:
        ret = GL_COMPRESSED_RGBA;
        sret = GL_RGBA;
        break;
    case GL_BGRA:
        if (hardext.bgra8888) {
            ret = GL_BGRA;
            sret = GL_BGRA;
        } else {
            ret = GL_RGBA;
            sret = GL_RGBA;
        }
        break;
    case GL_DEPTH_COMPONENT:
        // if (hardext.depthtex) {
        sret = ret = GL_DEPTH_COMPONENT;
        //}
        // else {
        //    sret = ret = GL_RGBA;
        //}
        break;
    case GL_DEPTH_COMPONENT16:
    case GL_DEPTH_COMPONENT24:
    case GL_DEPTH_COMPONENT32:
    case GL_DEPTH_COMPONENT32F:
        // if (hardext.depthtex) {
        switch (type) {
        case GL_UNSIGNED_SHORT:
            sret = ret = GL_DEPTH_COMPONENT;
            break;
        case GL_UNSIGNED_INT:
            sret = ret = GL_DEPTH_COMPONENT;
            break;
        case GL_FLOAT:
            sret = ret = GL_DEPTH_COMPONENT32F;
            break;
        default:
            sret = ret = GL_DEPTH_COMPONENT;
            break;
        }
        //}
        // else {
        //    sret = ret = GL_RGBA;
        //}
        break;
    case GL_DEPTH_STENCIL:
    case GL_DEPTH24_STENCIL8:
    case GL_DEPTH32F_STENCIL8:
        if (hardext.depthtex) {
            switch (type) {
            case GL_UNSIGNED_INT:
                sret = ret = GL_DEPTH24_STENCIL8;
                break;
            case GL_FLOAT:
                sret = ret = GL_DEPTH32F_STENCIL8;
                break;
            default:
                sret = ret = GL_DEPTH_STENCIL;
                break;
            }
        } else {
            sret = ret = GL_RGBA;
        }
        break;
    case GL_STENCIL_INDEX8:
        if (hardext.stenciltex) {
            sret = ret = GL_STENCIL_INDEX8;
        } else {
            sret = ret = (hardext.rgtex) ? GL_RED : GL_LUMINANCE;
        }
        break;
    case GL_R11F_G11F_B10F:
        ret = GL_R11F_G11F_B10F;
        sret = GL_R11F_G11F_B10F;
        break;
    default:
        if (hardext.depthstencil && format == GL_DEPTH_STENCIL) {
            sret = ret = GL_DEPTH_STENCIL;
            break;
        }
        if (hardext.depthtex && format == GL_DEPTH_COMPONENT) {
            sret = ret = GL_DEPTH_COMPONENT;
            break;
        }
        ret = GL_RGBA;
        sret = GL_RGBA;
        break;
        // Default...RGBA / RGBA will be fine....
    }
    *internalformat = ret;
    return sret;
}

static int get_shrinklevel(int width, int height, int level) {
    int shrink = 0;
    int mipwidth = width << level;
    int mipheight = height << level;
    // ZOMDROID FLOOR (measured 2026-08-30): shrinking is effectively mandatory on this
    // renderer, so its visible damage is a product problem rather than a preference --
    // and the damage lands on SMALL textures. Mode 1 halves everything above one pixel,
    // which is how it turned the rain droplets to mush. The session histogram says those
    // textures are not where the memory is: everything with a longest side of 128 or
    // less accounted for 10 MB of 4285 MB uploaded, a quarter of one percent. So never
    // shrink them, in any mode. (256 would have cost 24% -- that one stays on the table
    // only as an explicit quality option.) LIBGL_SHRINKFLOOR overrides the size.
    {
        static int zfloor = -1;
        if (zfloor < 0) {
            const char* ze = getenv("LIBGL_SHRINKFLOOR");
            zfloor = ze ? atoi(ze) : 128;
            if (zfloor < 0) zfloor = 0;
        }
        if (mipwidth <= zfloor && mipheight <= zfloor) return 0;
    }
    switch (globals4es.texshrink) {
    case 0: // nothing
        break;
    case 1: // everything / 2
        if ((mipwidth > 1) && (mipheight > 1)) {
            shrink = 1;
        }
        break;
    case 2: // only > 512 /2
    case 7: // only > 512 /2 , but not for empty texture
        if (((mipwidth % 2 == 0) && (mipheight % 2 == 0)) &&
            (((mipwidth > 512) && (mipheight > 8)) || ((mipheight > 512) && (mipwidth > 8)))) {
            shrink = 1;
        }
        break;
    case 3: // only > 256 /2
        if (((mipwidth % 2 == 0) && (mipheight % 2 == 0)) &&
            (((mipwidth > 256) && (mipheight > 8)) || ((mipheight > 256) && (mipwidth > 8)))) {
            shrink = 1;
        }
        break;
    case 4: // only > 256 /2, >=1024 /4
        if (((mipwidth % 4 == 0) && (mipheight % 4 == 0)) &&
            (((mipwidth > 256) && (mipheight > 8)) || ((mipheight > 256) && (mipwidth > 8)))) {
            if ((mipwidth >= 1024) || (mipheight >= 1024)) {
                shrink = 2;
            } else {
                shrink = 1;
            }
        }
        break;
    case 5: // every > 256 is downscaled to 256, but not for empty texture
        if (((mipwidth % 4 == 0) && (mipheight % 4 == 0)) &&
            (((mipwidth > 256) && (mipheight > 8)) || ((mipheight > 256) && (mipwidth > 8)))) {
            if ((mipwidth > 256) || (mipheight > 256)) {
                while (((mipwidth > 256) && (mipheight > 4)) || ((mipheight > 256) && (mipwidth > 4))) {
                    width /= 2;
                    height /= 2;
                    mipwidth /= 2;
                    mipheight /= 2;
                    shrink += 1;
                }
            } else {
                shrink = 1;
            }
        }
        break;
    case 6: // only > 128 /2, >=512 is downscaled to 256, but not for empty texture
        if (((mipwidth % 2 == 0) && (mipheight % 2 == 0)) &&
            (((mipwidth > 128) && (mipheight > 8)) || ((mipheight > 128) && (mipwidth > 8)))) {
            if ((mipwidth >= 512) || (mipheight >= 512)) {
                while (((mipwidth > 256) && (mipheight > 8)) || ((mipheight > 256) && (mipwidth > 8))) {
                    width /= 2;
                    height /= 2;
                    mipwidth /= 2;
                    mipheight /= 2;
                    shrink += 1;
                }
            } else {
                shrink = 1;
            }
        }
        break;
    case 8: // advertise *4 max texture size, but >2048 are shrinked to 2048
        if ((mipwidth > hardext.maxsize * 2) || (mipheight > hardext.maxsize * 2)) {
            shrink = 2;
        } else if ((mipwidth > hardext.maxsize) || (mipheight > hardext.maxsize)) {
            shrink = 1;
        }
        break;
    case 9: // advertise 8192 max texture size, but >4096 are quadshrinked and >512 are shrinked, but not for empty
            // texture
        if ((mipwidth > hardext.maxsize * 2) || (mipheight > hardext.maxsize * 2)) {
            shrink = 2;
        } else if (mipwidth > (hardext.maxsize >> 2) || mipheight > (hardext.maxsize >> 2)) {
            shrink = 1;
        }
        break;
    case 10: // advertise 8192 max texture size, but >2048 are quadshrinked and >512 are shrinked, but not for empty
             // texture
        if ((mipwidth > hardext.maxsize) || (mipheight > hardext.maxsize)) {
            shrink = 2;
        } else if ((mipwidth > (hardext.maxsize >> 2)) || (mipheight > (hardext.maxsize >> 2))) {
            shrink = 1;
        }
        break;
    case 11: // scale down to maxres any dimension > maxres
        if (mipwidth > hardext.maxsize || mipheight > hardext.maxsize) shrink = 1;
        break;
    }

    return shrink;
}

int wrap_npot(GLenum wrap) {
    switch (wrap) {
    case 0:
        return (globals4es.defaultwrap) ? 1 : 0;
    case GL_CLAMP:
    case GL_CLAMP_TO_EDGE:
    case GL_CLAMP_TO_BORDER:
        return 1;
    }
    return 0;
}
int minmag_npot(GLenum mag) {
    switch (mag) {
    case 0:
        return 0; // default is not good
    case GL_NEAREST:
    case GL_LINEAR:
        return 1;
    }
    return 0;
}

GLenum minmag_forcenpot(GLenum filt) {
    switch (filt) {
    case GL_LINEAR:
    case GL_LINEAR_MIPMAP_NEAREST:
    case GL_LINEAR_MIPMAP_LINEAR:
        return GL_LINEAR;
    /*case 0:
    case GL_NEAREST:
    case GL_NEAREST_MIPMAP_NEATEST:
    case GL_NEAREST_MIPMAP_LINEAR:*/
    default:
        return GL_NEAREST;
    }
}

GLenum wrap_forcenpot(GLenum wrap) {
    switch (wrap) {
    case 0:
        return GL_CLAMP_TO_EDGE;
    case GL_CLAMP:
    case GL_CLAMP_TO_EDGE:
    case GL_CLAMP_TO_BORDER:
        return wrap;
        /*case GL_MIRROR_CLAMP_TO_EDGE_EXT:
            return wrap;*/
    }
    return GL_CLAMP_TO_EDGE;
}

GLenum minmag_float(GLenum filt) {
    switch (filt) {
    case GL_LINEAR:
        return GL_NEAREST;
    case GL_LINEAR_MIPMAP_NEAREST:
    case GL_LINEAR_MIPMAP_LINEAR:
    case GL_NEAREST_MIPMAP_LINEAR:
        return GL_NEAREST_MIPMAP_NEAREST;
    default:
        return filt;
    }
}

// ZOMDROID GLALLOC: approximate bytes per texel of an upload request (driver may pad
// RGB to 4 — this ledger tracks what the app ASKS for, not the driver's rounding)
static long zga_texel_bytes(GLenum format, GLenum type) {
    int comp;
    switch (format) {
        case GL_ALPHA: case GL_LUMINANCE: case GL_RED: case GL_DEPTH_COMPONENT: comp = 1; break;
        case GL_LUMINANCE_ALPHA: case GL_RG: comp = 2; break;
        case GL_RGB: case GL_BGR: comp = 3; break;
        default: comp = 4; break;
    }
    switch (type) {
        case GL_UNSIGNED_SHORT_5_6_5: case GL_UNSIGNED_SHORT_4_4_4_4: case GL_UNSIGNED_SHORT_5_5_5_1: return 2;
        case GL_UNSIGNED_SHORT: return comp * 2;
        case GL_FLOAT: case GL_UNSIGNED_INT: case GL_UNSIGNED_INT_24_8: return comp * 4;
        default: return comp;
    }
}

void APIENTRY_GL4ES gl4es_glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height,
                                       GLint border, GLenum format, GLenum type, const GLvoid* data) {
    DBG(SHUT_LOGD(
            "glTexImage2D on target=%s with unpack_row_length(%i), size(%i,%i) and skip(%i,%i), "
            "format(internal)=%s(%s), type=%s, data=%p, level=%i (mipmap_need=%i, mipmap_auto=%i, base_level=%i, "
            "max_level=%i) => texture=%u (streamed=%i), glstate->list.compiling=%d\n",
            PrintEnum(target), glstate->texture.unpack_row_length, width, height, glstate->texture.unpack_skip_pixels,
            glstate->texture.unpack_skip_rows, PrintEnum(format),
            (internalformat == 3) ? "3" : (internalformat == 4 ? "4" : PrintEnum(internalformat)), PrintEnum(type),
            data, level, glstate->texture.bound[glstate->texture.active][what_target(target)]->mipmap_need,
            glstate->texture.bound[glstate->texture.active][what_target(target)]->mipmap_auto,
            glstate->texture.bound[glstate->texture.active][what_target(target)]->base_level,
            glstate->texture.bound[glstate->texture.active][what_target(target)]->max_level,
            glstate->texture.bound[glstate->texture.active][what_target(target)]->texture,
            glstate->texture.bound[glstate->texture.active][what_target(target)]->streamed, glstate->list.compiling);)

    // fuck weird depth handling!!!
    if (format == GL_DEPTH_COMPONENT) {
        internalformat = GL_DEPTH_COMPONENT;
        type = GL_UNSIGNED_INT;
    }

    if (data == NULL && (internalformat == GL_RGB16F || internalformat == GL_RGBA16F))
        internal2format_type(&internalformat, &format, &type);
    if (internalformat == GL_R16F) internal2format_type(&internalformat, &format, &type);
    if (data == NULL && (internalformat == GL_RED || internalformat == GL_RGB))
        internal2format_type(&internalformat, &format, &type);

    if (internalformat == GL_DEPTH32F_STENCIL8 && type == GL_FLOAT_32_UNSIGNED_INT_24_8_REV) {
        internalformat = GL_DEPTH24_STENCIL8;
        type = GL_UNSIGNED_INT_24_8;
    }

    // ZOMDROID RT16 (texture lavina, 2026-07-30): PZ 42.20 world streaming keeps
    // thousands of 4MB 1024x1024 RGBA8 textures allocated with NULL data (chunk-bake
    // targets; the game's texture options are a proven no-op on them — identical
    // GLALLOC traces). Drop them to 16-bit: halves the UNSWAPPABLE GL mtrack mass
    // that gets the process shot by lmkd. LIBGL_RT16: 0=off 1=RGBA4444 (default,
    // keeps 4-bit alpha) 2=RGBA5551 (more color, 1-bit alpha)
    if (data == NULL && level == 0 && width >= 512 && height >= 512 &&
        (format == GL_RGBA || format == GL_BGRA) &&
        (type == GL_UNSIGNED_BYTE || type == GL_UNSIGNED_INT_8_8_8_8_REV)) {
        // DEFAULT OFF since 2026-07-31: measured in the field, this branch caught
        // exactly three textures per session (two 1024x1024 and one 2048x1024) —
        // 8 MB saved against a 2.5 GB problem. Those textures are PZ's render targets,
        // and lighting accumulates in them: at 4 bits per channel a light gradient
        // breaks into visible steps and dither dots, which is what testers reported.
        // Pure cost, no benefit. LIBGL_RT16=1 brings it back for experiments.
        static int zrt16 = -1;
        if (zrt16 < 0) {
            const char* ze = getenv("LIBGL_RT16");
            zrt16 = ze ? atoi(ze) : 0;
        }
        if (zrt16 == 1 || zrt16 == 2) {
            internalformat = GL_RGBA;
            format = GL_RGBA;
            type = (zrt16 == 2) ? GL_UNSIGNED_SHORT_5_5_5_1 : GL_UNSIGNED_SHORT_4_4_4_4;
            static int zrt16_logged = 0;
            if (zrt16_logged < 3) {
                zrt16_logged++;
                extern void zomdroid_gltrace(const char* fmt, ...);
                zomdroid_gltrace("RT16 mode=%d NULL-tex %dx%d -> 16-bit", zrt16, width, height);
            }
        }
    }
    // proxy case
    const GLuint itarget = what_target(target);
    const GLuint rtarget = map_tex_target(target);
    // ZOMDROID T16 BOOKKEEPING (black-vehicles fix, field report 2026-07-31): whenever
    // this texture ends up in a 16-bit storage format — RT16 above did it for an empty
    // texture, or the app asked for it itself — record that on the texture object.
    // glTexSubImage2D reads it back: GLES requires a sub-upload to match the storage
    // format, and PZ paints vehicle skins into an empty 2048x1024 atlas with RGBA8
    // sub-uploads. Mismatched calls are rejected by the driver (silently, since we run
    // with noerror), the atlas keeps its zeros, and every car renders black.
    if (level == 0 && (type == GL_UNSIGNED_SHORT_4_4_4_4 || type == GL_UNSIGNED_SHORT_5_5_5_1 ||
                       type == GL_UNSIGNED_SHORT_5_6_5)) {
        gltexture_t* zb0 = glstate->texture.bound[glstate->texture.active][itarget];
        if (zb0) {
            zb0->zt16_format = format;
            zb0->zt16_type = type;
        }
    }
    // ZOMDROID ADAPTIVE TEXTURE BUDGET (RC29, field report 2026-07-31): T16 halved the
    // lavina but on 8 GB devices the game still walks past the ceiling — a Dimensity
    // tester was killed at 1756 MB of live textures and climbing. So: keep full
    // resolution while there is room, and once live textures cross the budget, halve
    // NEW uploads too (gl4es shrink mode 7: >512 with data, never empty/FBO textures).
    // Already-resident textures stay sharp; when the game's own cache drops us back
    // under the low-water mark, new uploads go back to full size. The visible cost
    // (tile seams on chunks loaded while over budget) is paid only by devices that
    // would otherwise be killed. LIBGL_TEXBUDGET=<MB>, 0 disables; a non-zero
    // LIBGL_SHRINK from the user always wins.
    {
        static int zbud_hi = -1, zbud_lo = 0, zbud_on = 0, zbud_manual = 0;
        if (zbud_hi < 0) {
            // DEFAULT OFF: halving resolution is a deal — worse textures instead of a
            // kill — and the player has to make it knowingly, not discover it after an
            // update. The launcher exposes it as a "memory saver" switch that sets this
            // variable (800 MB suits 8 GB devices); unset means nothing degrades.
            const char* ze = getenv("LIBGL_TEXBUDGET");
            zbud_hi = ze ? atoi(ze) : 0;
            if (zbud_hi < 0) zbud_hi = 0;
            zbud_lo = (zbud_hi > 300) ? zbud_hi - 300 : zbud_hi; // hysteresis
            // A non-zero shrink mode chosen by the user stands the budget down: modes 1 and
            // 7 already halve at least as much. LIBGL_SHRINK=0 is not such a choice -- it is
            // how the launcher writes "no shrink" (its presets keep the line and set 0) -- and
            // counting it as manual disarmed the budget for nearly every NG player: both
            // saver-on field reports carried LIBGL_SHRINK and logged no TEXBUDGET line at
            // 1167 and 1835 MB of live textures.
            const char* zs = getenv("LIBGL_SHRINK");
            zbud_manual = (zs && atoi(zs) != 0) ? 1 : 0;
        }
        if (zbud_hi > 0 && !zbud_manual) {
            long zlive = zomdroid_glalloc_live_tex() >> 20;
            if (!zbud_on && zlive >= zbud_hi) {
                zbud_on = 1;
                globals4es.texshrink = 7;
                zomdroid_gltrace("TEXBUDGET over %ldMB >= %dMB — new uploads halved", zlive, zbud_hi);
            } else if (zbud_on && zlive <= zbud_lo) {
                zbud_on = 0;
                globals4es.texshrink = 0;
                zomdroid_gltrace("TEXBUDGET back under %ldMB <= %dMB — full size restored", zlive, zbud_lo);
            }
        }
    }
    // ZOMDROID ETC2 (prototype, opt-in via LIBGL_ETC2=1): same eligibility as T16 below,
    // but instead of 16-bit the pixels are ETC2-encoded at FULL resolution: 4 bytes/px
    // -> 1 (RGBA8_ETC2_EAC) or 0.5 (opaque -> RGB8_ETC2). The upload goes through our own
    // glCompressedTexImage2D, whose generic branch passes non-DXT formats straight to
    // the driver with full bookkeeping. ES3-only (ETC2 is core there). Textures built
    // by sub-uploads are never taken (data==NULL creations skip this), and a later
    // glTexSubImage2D into an ETC2 texture is refused + logged, not silently corrupted.
    //
    // Soft alpha stays out (2026-09-09, "rain and fog visible inside buildings" with ETC2
    // on). Measured against an exact model of our EAC search: a block of hard 0/255 edges
    // reproduces both exactly, an anti-aliased edge drifts a zero to at most 2, but a
    // smooth gradient drifts it to 16 -- a 6% floor where the texture should be empty,
    // which over a full-screen fog or rain layer is plainly visible. Sprite atlases are
    // hard-edged (models atlases measure 5-10% mid-alpha pixels, all edges) and keep the
    // 4:1 EAC; textures whose alpha is mostly a gradient (fog 93-100%, vehicle rust and
    // damage decals 10-90%, blood 38%) take the T16 path instead, where zero is exact.
    // LIBGL_ETC2_SOFTALPHA=<percent of 0<a<255 pixels> is the cut (default 10);
    // -1 skips any transparency at all, 100 never skips (the RC50 behaviour).
    //
    // Channel-coded masks stay out too (2026-09-10, "red textures at wall seams" on a
    // PowerVR tester already carrying the texture-unit fix). PZ's cutaway masks
    // (wallcutaways.png, windowframe_cutaways*.png) are data, not art: red = the cut,
    // green/blue = door and window frames, and the shaders test channels exactly --
    // "if (g + b > 0.0) discard;". ETC1 cannot keep a channel at zero: the uniform-block
    // fast path adds +2 to all three channels, so (255,0,0) comes back (255,2,2) and the
    // shader throws the wall pixel away. Measured on the real masks through an exact model
    // of our encoder: 90-92% of the pure-red pixels discarded. A texture whose opaque
    // pixels are mostly pure R/G/B is such a mask; across all 2805 large textures in
    // 42.20 exactly the four cutaway masks qualify. They take the T16 path, where 4-bit
    // channels hold 0 and 255 exactly. LIBGL_ETC2_CHANNELMASK=<percent of opaque pixels
    // that are pure primaries> is the cut (default 50); 101 never skips.
    if (data != NULL && target != GL_PROXY_TEXTURE_2D && width > 0 && height > 0 &&
        type == GL_UNSIGNED_BYTE && (format == GL_RGBA || format == GL_RGB) &&
        glstate->vao->unpack == NULL && glstate->texture.unpack_row_length == 0 &&
        glstate->texture.unpack_skip_pixels == 0 && glstate->texture.unpack_skip_rows == 0 &&
        globals4es.esversion >= 300) {
        static int zetc2_on = -1;
        if (zetc2_on < 0) {
            const char* ze = getenv("LIBGL_ETC2");
            zetc2_on = ze ? atoi(ze) : 0; // prototype: default OFF
        }
        gltexture_t* zeb = glstate->texture.bound[glstate->texture.active][what_target(target)];
        if (zetc2_on && zeb) {
            // A base-level TexImage redefines the object. Clear the previous decision first;
            // otherwise an ETC2 object reused for alpha data would still reject sub-updates.
            if (level == 0)
                zeb->zetc2_format = 0;
            GLenum zfmt = 0;
            // >= 512x512 (lowered from 1024 on her word 2026-08-31, "дожми 512"): the
            // 513..1023 band held hundreds of MB in the size histogram. 128..512 stays
            // uncompressed -- that is UI and close-up art, and the shrink floor territory.
            if (level == 0 && (long)width * height >= 512L * 512) {
                if (format == GL_RGB) {
                    zfmt = 0x9274; // GL_COMPRESSED_RGB8_ETC2
                } else {
                    // one pass over alpha: fully opaque -> RGB8_ETC2 (8:1); hard-edged
                    // -> RGBA8_ETC2_EAC (4:1); gradient alpha -> not ours, see above
                    static int zsoft_cut = -2;
                    if (zsoft_cut == -2) {
                        const char* zs = getenv("LIBGL_ETC2_SOFTALPHA");
                        zsoft_cut = zs ? atoi(zs) : 10;
                    }
                    static int zmask_cut = -2;
                    if (zmask_cut == -2) {
                        const char* zm = getenv("LIBGL_ETC2_CHANNELMASK");
                        zmask_cut = zm ? atoi(zm) : 50;
                    }
                    const unsigned char* zp = (const unsigned char*)data;
                    long zn = (long)width * height;
                    long zmid = 0, znon = 0, zopq = 0, zprim = 0;
                    for (long zi = 3, ze = zn * 4; zi < ze; zi += 4) {
                        unsigned char za = zp[zi];
                        if (za != 255) {
                            znon++;
                            if (za != 0) zmid++;
                        }
                        if (za != 0) {
                            zopq++;
                            unsigned char zr = zp[zi - 3], zg = zp[zi - 2], zb = zp[zi - 1];
                            if ((zr == 255 && zg == 0 && zb == 0) || (zg == 255 && zr == 0 && zb == 0) ||
                                (zb == 255 && zr == 0 && zg == 0))
                                zprim++;
                        }
                    }
                    int zsoft = (zsoft_cut < 0) ? (znon > 0) : (zmid * 100 > zn * (long)zsoft_cut);
                    int zmask = (zopq > 0) && (zprim * 100 >= zopq * (long)zmask_cut);
                    if (zmask) {
                        static int zmask_log = 0;
                        if (zmask_log < 6) {
                            zmask_log++;
                            zomdroid_gltrace("ETC2 channel-mask %dx%d L%d prim=%ld%% (cut %d) skipped",
                                             width, height, level, zprim * 100 / zopq, zmask_cut);
                        }
                    } else if (znon == 0) {
                        zfmt = 0x9274; // GL_COMPRESSED_RGB8_ETC2
                    } else if (!zsoft) {
                        zfmt = 0x9278; // GL_COMPRESSED_RGBA8_ETC2_EAC
                    } else {
                        static int zsoft_log = 0;
                        if (zsoft_log < 6) {
                            zsoft_log++;
                            zomdroid_gltrace("ETC2 soft-alpha %dx%d L%d mid=%ld%% (cut %d) skipped",
                                             width, height, level, zmid * 100 / zn, zsoft_cut);
                        }
                    }
                }
            } else if (level > 0 && zeb->zetc2_format) {
                zfmt = zeb->zetc2_format; // mip chain follows level 0, never mixes
            }
            if (zfmt) {
                size_t zsz = 0;
                static int zcache_on = -1;
                if (zcache_on < 0) {
                    const char* zc = getenv("LIBGL_ETC2CACHE");
                    zcache_on = zc ? atoi(zc) : 1; // the cache IS the point: default on
                }
                const void* zblocks =
                        zcache_on ? zomdroid_etc2_cached(zfmt, width, height, (const uint8_t*)data,
                                                         (format == GL_RGB) ? 3 : 4, &zsz)
                                  : zomdroid_etc2_encode(zfmt, width, height, (const uint8_t*)data,
                                                         (format == GL_RGB) ? 3 : 4, &zsz);
                if (zblocks) {
                    gl4es_glCompressedTexImage2D(target, level, zfmt, width, height, border,
                                                 (GLsizei)zsz, zblocks);
                    if (level == 0) {
                        zeb->zetc2_format = zfmt;
                        zeb->alpha = (zfmt == 0x9278) ? 1 : 0;
                        zeb->width = width;
                        zeb->height = height;
                        zeb->nwidth = width;
                        zeb->nheight = height;
                        zeb->adjustxy[0] = zeb->adjustxy[1] = 1.0f;
                        zeb->shrink = 0;
                    }
                    {
                        extern void zomdroid_gltrace(const char* fmt, ...);
                        static int zlog = 0;
                        if (zlog < 6) {
                            zlog++;
                            zomdroid_gltrace("ETC2 %dx%d L%d %s -> %zu bytes (encode total %llums)",
                                             width, height, level,
                                             (zfmt == 0x9278) ? "RGBA8_EAC" : "RGB8",
                                             zsz, (unsigned long long)zomdroid_etc2_total_ms());
                        }
                    }
                    return;
                }
                // encoder refused (size cap / OOM): fall through to the normal path
            }
        }
    }
    // ZOMDROID T16 (texture lavina, RC28): PZ streams thousands of 1024x1024 RGBA8
    // world textures — measured 3.1GB live / 6900 objects at the lmkd kill. They are
    // NOT DXT (the decompression crumb never fired) and the game's own texture options
    // do not touch them, so the only lever left is the upload width. Convert big
    // uploads to 16-bit at FULL resolution: half the driver mass without the tile-seam
    // grid that halving resolution produced in RC26. The format follows the actual
    // alpha content, exactly like gl4es' DXT path does. LIBGL_T16=0 disables.
    if (data != NULL && target != GL_PROXY_TEXTURE_2D && width > 0 && height > 0 &&
        type == GL_UNSIGNED_BYTE && (format == GL_RGBA || format == GL_RGB) &&
        glstate->texture.unpack_row_length == 0 && glstate->texture.unpack_skip_pixels == 0 &&
        glstate->texture.unpack_skip_rows == 0) {
        static int zt16 = -1;
        if (zt16 < 0) {
            const char* ze = getenv("LIBGL_T16");
            // (the RC34 probe that defaulted this to 0 was never handed out — the black
            // plots turned out to be gated by the puddles option, see the hoist pass)
            zt16 = ze ? atoi(ze) : 1;
        }
        gltexture_t* zb = glstate->texture.bound[glstate->texture.active][itarget];
        if (zt16 == 1 && zb) {
            GLenum zf = 0, zt = 0;
            // >= 1024x1024 only (raised from 512 on 2026-07-31): the mass is the world
            // tile stream, thousands of 1024x1024 uploads at 2 MB each once converted.
            // The 512-and-below crowd is UI, atlases and other art the player looks at
            // from close up — half a megabyte saved each, not worth the visible cost.
            if (level == 0 && (long)width * height >= 1024L * 1024) {
                if (format == GL_RGB) {
                    zf = GL_RGB;
                    zt = GL_UNSIGNED_SHORT_5_6_5;
                } else {
                    // one pass over the alpha channel: opaque -> 565 (best colors),
                    // cutout (0/255 only) -> 5551, soft alpha -> 4444
                    const unsigned char* zp = (const unsigned char*)data;
                    int zopaque = 1, zsimple = 1;
                    for (long zi = 3, zn = (long)width * height * 4; zi < zn; zi += 4) {
                        unsigned char za = zp[zi];
                        if (za != 255) {
                            zopaque = 0;
                            if (za != 0) {
                                zsimple = 0;
                                break;
                            }
                        }
                    }
                    zf = zopaque ? GL_RGB : GL_RGBA;
                    zt = zopaque ? GL_UNSIGNED_SHORT_5_6_5
                                 : (zsimple ? GL_UNSIGNED_SHORT_5_5_5_1 : GL_UNSIGNED_SHORT_4_4_4_4);
                }
            } else if (level > 0 && zb->zt16_type) {
                zf = zb->zt16_format;
                zt = zb->zt16_type;
            }
            if (zt) {
                GLvoid* zdst = NULL;
                if (pixel_convert(data, &zdst, width, height, format, type, zf, zt, 0,
                                  glstate->texture.unpack_align) &&
                    zdst && zdst != data) {
                    // deferred free: the driver has copied the previous buffer long ago
                    static GLvoid* zt16_prev = NULL;
                    if (zt16_prev) free(zt16_prev);
                    zt16_prev = zdst;
                    data = zdst;
                    format = zf;
                    type = zt;
                    internalformat = zf; // so swizzle_internalformat keeps the 16-bit form
                    if (level == 0) {
                        zb->zt16_format = zf;
                        zb->zt16_type = zt;
                    }
                    static int zt16_logged = 0;
                    if (zt16_logged < 4) {
                        zt16_logged++;
                        zomdroid_gltrace("T16 %dx%d RGBA8 -> %s", width, height,
                                         zt == GL_UNSIGNED_SHORT_5_6_5
                                                 ? "565"
                                                 : (zt == GL_UNSIGNED_SHORT_5_5_5_1 ? "5551" : "4444"));
                    }
                }
            }
        }
    }
    // ZOMDROID GLALLOC: driver-side texture storage the app requests
    // (level 0 = live size keyed by app id; higher levels count as churn only)
    if (target != GL_PROXY_TEXTURE_2D && width > 0 && height > 0) {
        gltexture_t* zbt = glstate->texture.bound[glstate->texture.active][itarget];
        long zbytes = (long)width * height * zga_texel_bytes(format, type);
        // mirror what shrink mode 7 will actually hand the driver, otherwise the budget
        // above would never see the saving it just caused and could not fall back
        if (globals4es.texshrink == 7 && data != NULL && width > 512 && height > 512) zbytes /= 4;
        if (level == 0 && zbt)
            zomdroid_glalloc_set(0, zbt->texture, zbytes);
        else
            zomdroid_glalloc_cum(0, zbytes);
    }
    LOAD_GLES(glTexImage2D);
    LOAD_GLES(glTexSubImage2D);
    void gles_glTexParameteri(glTexParameteri_ARG_EXPAND); // LOAD_GLES(glTexParameteri);

    if (globals4es.force16bits) {
        if (internalformat == GL_RGBA || internalformat == 4 || internalformat == GL_RGBA8)
            internalformat = GL_RGBA4;
        else if (internalformat == GL_RGB || internalformat == 3 || internalformat == GL_RGB8)
            internalformat = GL_RGB5;
    }

    if (rtarget == GL_PROXY_TEXTURE_2D) {
        int max1 = hardext.maxsize;
        glstate->proxy_width = ((width << level) > max1) ? 0 : width;
        glstate->proxy_height = ((height << level) > max1) ? 0 : height;
        glstate->proxy_intformat = swizzle_internalformat((GLenum*)&internalformat, format, type);
        return;
    }
    // actually bound if targeting shared TEX2D
    realize_bound(glstate->texture.active, target);

    if (glstate->list.pending) {
        gl4es_flush();
    } else {
        PUSH_IF_COMPILING(glTexImage2D);
    }

#ifdef __BIG_ENDIAN__
    if (type == GL_UNSIGNED_INT_8_8_8_8)
#else
    if (type == GL_UNSIGNED_INT_8_8_8_8_REV)
#endif
        type = GL_UNSIGNED_BYTE;

    if (type == GL_HALF_FLOAT) type = GL_HALF_FLOAT_OES;

    /*if(format==GL_COMPRESSED_LUMINANCE)
        format = GL_RGB;*/    // Danger from the Deep does that.
        //That's odd, probably a bug (line 453 of src/texture.cpp, it should be interformat instead of format)

    GLvoid* datab = (GLvoid*)data;

    if (glstate->vao->unpack) datab = (char*)datab + (uintptr_t)glstate->vao->unpack->data;

    GLvoid* pixels = (GLvoid*)datab;
    border = 0; // TODO: something?
    noerrorShim();

    gltexture_t* bound = glstate->texture.bound[glstate->texture.active][itarget];

    // Special case when resizing an attached to FBO texture, that is attached to depth and/or stencil => resizing is
    // specific then
    if (bound->binded_fbo &&
        (bound->binded_attachment == GL_DEPTH_ATTACHMENT || bound->binded_attachment == GL_STENCIL_ATTACHMENT ||
         bound->binded_attachment == GL_DEPTH_STENCIL_ATTACHMENT)) {
        // non null data should be handled, but need to convert then...
        if (data != NULL) {
            SHUT_LOGD("LIBGL: Warning, Depth/stencil texture resized and with data\n");
        }
        // get new size...
        GLsizei nheight = (hardext.npot) ? height : npot(height);
        GLsizei nwidth = (hardext.npot) ? width : npot(width);
        bound->npot = (nheight != height || nwidth != width);
        bound->nwidth = nwidth;
        bound->nheight = nheight;
        bound->width = width;
        bound->height = height;
        // resize depth texture of renderbuffer?
        if (bound->binded_attachment == GL_DEPTH_ATTACHMENT ||
            bound->binded_attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
            if (bound->renderdepth) {
                gl4es_glBindRenderbuffer(GL_RENDERBUFFER, bound->renderdepth);
                gl4es_glRenderbufferStorage(GL_RENDERBUFFER,
                                            (bound->binded_attachment == GL_DEPTH_ATTACHMENT) ? GL_DEPTH_COMPONENT16
                                                                                              : GL_DEPTH24_STENCIL8,
                                            nwidth, nheight);
                gl4es_glBindRenderbuffer(GL_RENDERBUFFER, 0);
            } else {
                errorGL();
                DBG(SHUT_LOGD("gles_glTexImage2D(%d, %d, %s, %d, %d, %d, %s, %s, 0x%x)\n", GL_TEXTURE_2D, 0,
                              PrintEnum(bound->format), bound->nwidth, bound->nheight, 0, PrintEnum(bound->format),
                              PrintEnum(bound->type), NULL);)
                gles_glTexImage2D(GL_TEXTURE_2D, 0, bound->format, bound->nwidth, bound->nheight, 0, bound->format,
                                  bound->type, NULL);
                DBG(CheckGLError(1);)
            }
        }
        if ((bound->binded_attachment == GL_STENCIL_ATTACHMENT ||
             bound->binded_attachment == GL_DEPTH_STENCIL_ATTACHMENT) &&
            bound->renderstencil) {
            gl4es_glBindRenderbuffer(GL_RENDERBUFFER, bound->renderstencil);
            gl4es_glRenderbufferStorage(GL_RENDERBUFFER, GL_STENCIL_INDEX8, nwidth, nheight);
            gl4es_glBindRenderbuffer(GL_RENDERBUFFER, 0);
        }
        // all done, exit
        errorGL();
        return;
    }

    if (target == GL_TEXTURE_RECTANGLE_ARB) {
        // change sampler state
        bound->sampler.min_filter = minmag_forcenpot(bound->sampler.min_filter);
        bound->sampler.wrap_s = wrap_forcenpot(bound->sampler.wrap_s);
        bound->sampler.wrap_t = wrap_forcenpot(bound->sampler.wrap_t);
        bound->sampler.wrap_r = wrap_forcenpot(bound->sampler.wrap_r);
    }

    if (target == GL_TEXTURE_RECTANGLE_ARB) {
        // change sampler state
        bound->sampler.min_filter = minmag_forcenpot(bound->sampler.min_filter);
        bound->sampler.wrap_s = wrap_forcenpot(bound->sampler.wrap_s);
        bound->sampler.wrap_t = wrap_forcenpot(bound->sampler.wrap_t);
        bound->sampler.wrap_r = wrap_forcenpot(bound->sampler.wrap_r);
    }

    bound->alpha = pixel_hasalpha(format);
    // fpe internal format tracking
    if (glstate->fpe_state) {
        switch (internalformat) {
        case GL_COMPRESSED_ALPHA:
        case GL_ALPHA4:
        case GL_ALPHA8:
        case GL_ALPHA16:
        case GL_ALPHA16F:
        case GL_ALPHA32F:
        case GL_ALPHA:
            bound->fpe_format = FPE_TEX_ALPHA;
            break;
        case 1:
        case GL_COMPRESSED_LUMINANCE:
        case GL_LUMINANCE4:
        case GL_LUMINANCE8:
        case GL_LUMINANCE16:
        case GL_LUMINANCE16F:
        case GL_LUMINANCE32F:
        case GL_LUMINANCE:
            bound->fpe_format = FPE_TEX_LUM;
            break;
        case 2:
        case GL_COMPRESSED_LUMINANCE_ALPHA:
        case GL_LUMINANCE4_ALPHA4:
        case GL_LUMINANCE8_ALPHA8:
        case GL_LUMINANCE16_ALPHA16:
        case GL_LUMINANCE_ALPHA16F:
        case GL_LUMINANCE_ALPHA32F:
        case GL_LUMINANCE_ALPHA:
            bound->fpe_format = FPE_TEX_LUM_ALPHA;
            break;
        case GL_COMPRESSED_INTENSITY:
        case GL_INTENSITY8:
        case GL_INTENSITY16:
        case GL_INTENSITY16F:
        case GL_INTENSITY32F:
        case GL_INTENSITY:
            bound->fpe_format = FPE_TEX_INTENSITY;
            break;
        case 3:
        case GL_RED:
        case GL_RG:
        case GL_RGB:
        case GL_RGB5:
        case GL_RGB565:
        case GL_RGB8:
        case GL_RGB16:
        case GL_RGB16F:
        case GL_RGB32F:
        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT:
        case GL_COMPRESSED_RGB:
        case GL_R11F_G11F_B10F:
        case GL_R32F:
        case GL_RGB10_A2:
            bound->fpe_format = FPE_TEX_RGB;
            break;
        /*case GL_DEPTH_COMPONENT:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_STENCIL:
        case GL_DEPTH24_STENCIL8:
            bound->fpe_format = FPE_TEX_COMPONENT;*/
        default:
            bound->fpe_format = FPE_TEX_RGBA;
        }
    }
    if (globals4es.automipmap) {
        if (level > 0)
            if ((globals4es.automipmap == 1) || (globals4es.automipmap == 3) || bound->mipmap_need) {
                return; // has been handled by auto_mipmap
            } else if (globals4es.automipmap == 2)
                bound->mipmap_need = 1;
    }
    if (level > 0 && (bound->npot && globals4es.forcenpot)) return; // no mipmap...
    if (level == 0 || !bound->valid) {
        bound->wanted_internal = internalformat; // save it before transformation
    }
    GLenum new_format = swizzle_internalformat((GLenum*)&internalformat, format, type);
    if (level == 0 || !bound->valid) {
        bound->orig_internal = internalformat;
        bound->internalformat = new_format;
    }
    // shrink checking
    int mipwidth = width << level;
    int mipheight = height << level;
    int shrink = 0;
    if (!bound->valid)
        bound->shrink = shrink = get_shrinklevel(width, height, level);
    else
        shrink = bound->shrink;

    if (((width >> shrink) == 0) && ((height >> shrink) == 0)) return; // nothing to do
    if (datab) {

        // implements GL_UNPACK_ROW_LENGTH
        if ((glstate->texture.unpack_row_length && glstate->texture.unpack_row_length != width) ||
            glstate->texture.unpack_skip_pixels || glstate->texture.unpack_skip_rows) {
            int imgWidth, pixelSize, dstWidth;
            pixelSize = pixel_sizeof(format, type);
            imgWidth = ((glstate->texture.unpack_row_length) ? glstate->texture.unpack_row_length : width) * pixelSize;
            GLubyte* dst = (GLubyte*)malloc(width * height * pixelSize);
            pixels = (GLvoid*)dst;
            dstWidth = width * pixelSize;
            const GLubyte* src = (GLubyte*)datab;
            src += glstate->texture.unpack_skip_pixels * pixelSize + glstate->texture.unpack_skip_rows * imgWidth;

            for (int y = height; y; --y) { /*
                 if (dst == NULL || src == NULL) {
                     SHUT_LOGD("LIBGL: Invalid memory pointers in memcpy (src=%p, dst=%p)\n", src, dst);
                     return;  // Exit early or handle the error
                 }
                 // Before the unpacking loop
                 if (dstWidth <= 0 || imgWidth <= 0) {
                     SHUT_LOGD("LIBGL: Invalid buffer sizes for memcpy (dstWidth=%d, imgWidth=%d)\n", dstWidth,
                 imgWidth); return;  // Exit early or handle the error
                 }
                 if ((uintptr_t)src % sizeof(void*) != 0 || (uintptr_t)dst % sizeof(void*) != 0) {
                     SHUT_LOGD("LIBGL: Memory is not aligned correctly for memcpy (src=%p, dst=%p)\n", src, dst);
                     return;  // Exit early or handle the error
                 }
                 if (width <= 0 || height <= 0) {
                     SHUT_LOGD("LIBGL: Invalid width or height for texture update (width=%d, height=%d)\n", width,
                 height); return;  // Exit early or handle the error
                 }
                 if ((uintptr_t)src + height * imgWidth > (uintptr_t)(src + height * imgWidth)) {
                     SHUT_LOGD("LIBGL: Source buffer overflow detected (src=%p, expected=%p)\n", src, (src + height *
                 imgWidth)); return;  // Exit early or handle the error
                 }
                 if ((uintptr_t)dst + height * dstWidth > (uintptr_t)(dst + height * dstWidth)) {
                     SHUT_LOGD("LIBGL: Destination buffer overflow detected (dst=%p, expected=%p)\n", dst, (dst + height
                 * dstWidth)); return;  // Exit early or handle the error
                 }*/

                memcpy(dst, src, dstWidth);
                src += imgWidth;
                dst += dstWidth;
            }
        }

        GLvoid* old = pixels;
        pixels = (GLvoid*)swizzle_texture(width, height, &format, &type, internalformat, new_format, old, bound);
        if (old != pixels && old != datab) {
            free(old);
        }

        if (bound->shrink != 0) {
            switch (globals4es.texshrink) {
            case 0: // nothing ???
                break;
            case 1: // everything / 2
            case 11:
                if ((mipwidth > 1) && (mipheight > 1)) {
                    GLvoid* out = pixels;
                    GLfloat ratiox, ratioy;
                    int newwidth = mipwidth;
                    int newheight = mipheight;
                    if (globals4es.texshrink == 11) {
                        if (mipwidth > hardext.maxsize) newwidth = hardext.maxsize;
                        if (mipheight > hardext.maxsize) newheight = hardext.maxsize;
                    } else {
                        newwidth = mipwidth / 2;
                        newheight = mipheight / 2;
                        if (!newwidth) newwidth = 1;
                        if (!newheight) newheight = 1;
                    }
                    if (level && bound->valid) {
                        // don't recalculate ratio...
                        ratiox = bound->ratiox;
                        ratioy = bound->ratioy;
                    } else {
                        ratiox = newwidth / ((float)mipwidth);
                        ratioy = newheight / ((float)mipheight);
                    }
                    newwidth = width * ratiox;
                    newheight = height * ratioy;
                    if (ratiox == 0.5f && ratioy == 0.5f && npot(width) == width && npot(height) == height) {
                        // prefer the fast and clean way first
                        pixel_halfscale(pixels, &out, width, height, format, type);
                    } else {
                        bound->ratiox = ratiox;
                        bound->ratioy = ratioy;
                        bound->useratio = 1;
                        pixel_scale(pixels, &out, width, height, newwidth, newheight, format, type);
                    }
                    if (out != pixels && pixels != datab) free(pixels);
                    pixels = out;
                    width = newwidth;
                    height = newheight;
                    bound->shrink = 1;
                }
                break;
            default:
                if (!bound->valid) bound->ratiox = bound->ratioy = 1.0f;
                while (shrink) {
                    int toshrink = (shrink > 1) ? 2 : 1;
                    GLvoid* out = pixels;
                    if (toshrink == 1) {
                        pixel_halfscale(pixels, &out, width, height, format, type);
                        if (!bound->valid) {
                            bound->ratiox *= 0.5f;
                            bound->ratioy *= 0.5f;
                        }
                    } else {
                        pixel_quarterscale(pixels, &out, width, height, format, type);
                        if (!bound->valid) {
                            bound->ratiox *= 0.25f;
                            bound->ratioy *= 0.25f;
                        }
                    }
                    if (out != pixels && pixels != datab) free(pixels);
                    pixels = out;
                    width = nlevel(width, toshrink);
                    height = nlevel(height, toshrink);
                    shrink -= toshrink;
                }
            }
        }

        if (globals4es.texdump) {
            pixel_to_ppm(pixels, width, height, format, type, bound->texture, glstate->texture.pack_align);
        }
    } else {
#ifdef TEXSTREAM
        if (globals4es.texstream && (target == GL_TEXTURE_2D || target == GL_TEXTURE_RECTANGLE_ARB) &&
                (width >= 256 && height >= 224) &&
                ((internalformat == GL_RGB) || (internalformat == 3) || (internalformat == GL_RGB8) ||
                 (internalformat == GL_BGR) || (internalformat == GL_RGB5) || (internalformat == GL_RGB565)) ||
            (globals4es.texstream == 2)) {
            bound->streamingID = AddStreamed(width, height, bound->texture);
            if (bound->streamingID > -1) { // success
                bound->shrink = 0;         // no shrink on Stream texture
                bound->streamed = true;
                glsampler_t* sampler = glstate->samplers.sampler[glstate->texture.active];
                if (!sampler) sampler = &bound->sampler;
                ApplyFilterID(bound->streamingID, sampler->min_filter, sampler->mag_filter);
                GLboolean tmp = IS_ANYTEX(glstate->enable.texture[glstate->texture.active]);
                LOAD_GLES(glDisable);
                LOAD_GLES(glEnable);
                if (tmp) gles_glDisable(GL_TEXTURE_2D);
                ActivateStreaming(bound->streamingID); // Activate the newly created texture
                format = GL_RGB;
                type = GL_UNSIGNED_SHORT_5_6_5;
                if (tmp) gles_glEnable(GL_TEXTURE_STREAM_IMG);
            }
            glstate->bound_stream[glstate->texture.active] = 1;
        }
#endif
        if (!bound->streamed && internalformat != GL_R11F_G11F_B10F && internalformat != GL_R32F &&
            internalformat != GL_RGB10_A2)
            swizzle_texture(width, height, &format, &type, internalformat, new_format, NULL,
                            bound); // convert format even if data is NULL
        if (internalformat == GL_R11F_G11F_B10F || internalformat == GL_R32F) type = GL_FLOAT;
        if (internalformat == GL_RGB10_A2) {
            if (format == GL_BGRA) format = GL_RGBA;
            type = GL_UNSIGNED_INT_2_10_10_10_REV;
        }
        if (bound->shrink != 0) {
            switch (globals4es.texshrink) {
            case 1: // everything / 2
            case 11:
                if ((mipwidth > 1) && (mipheight > 1)) {
                    GLfloat ratiox, ratioy;
                    if (globals4es.texshrink == 11) {
                        if (mipwidth > hardext.maxsize)
                            ratiox = hardext.maxsize / ((float)mipwidth);
                        else
                            ratiox = 1.0f;
                        if (mipheight > hardext.maxsize)
                            ratioy = hardext.maxsize / ((float)mipheight);
                        else
                            ratioy = 1.0f;
                    } else
                        ratiox = ratioy = 0.5;
                    bound->ratiox = ratiox;
                    bound->ratioy = ratioy;
                    bound->useratio = 1;
                    int newwidth = width * bound->ratiox;
                    int newheight = height * bound->ratioy;
                    if (globals4es.texshrink == 11 && newwidth > hardext.maxsize)
                        newwidth = hardext.maxsize; // in case of some rounding error
                    if (globals4es.texshrink == 11 && newheight > hardext.maxsize)
                        newheight = hardext.maxsize; // in case of some rounding error
                    width = newwidth;
                    height = newheight;
                    bound->shrink = 1;
                }
                break;
            default:
                bound->ratiox = bound->ratioy = 1.0f;
                while (shrink) {
                    int toshrink = (shrink > 1) ? 2 : 1;
                    if (toshrink == 1) {
                        bound->ratiox *= 0.5f;
                        bound->ratioy *= 0.5f;
                    } else {
                        bound->ratiox *= 0.25f;
                        bound->ratioy *= 0.25f;
                    }
                    width = nlevel(width, toshrink);
                    height = nlevel(height, toshrink);
                    shrink -= toshrink;
                }
            }
        }
    }

    /* TODO:
    GL_INVALID_VALUE is generated if border is not 0.
    GL_INVALID_OPERATION is generated if type is
    GL_UNSIGNED_SHORT_5_6_5 and format is not GL_RGB.

    GL_INVALID_OPERATION is generated if type is one of
    GL_UNSIGNED_SHORT_4_4_4_4, or GL_UNSIGNED_SHORT_5_5_5_1
    and format is not GL_RGBA.
    */

    int limitednpot = 0;
    {
        GLsizei nheight = (hardext.npot == 3) ? height : npot(height);
        GLsizei nwidth = (hardext.npot == 3) ? width : npot(width);
        bound->npot = (nheight != height || nwidth != width); // hardware that fully support NPOT doesn't care anyway
        if (bound->npot) {
            if (target == GL_TEXTURE_RECTANGLE_ARB && hardext.npot)
                limitednpot = 1;
            else if (hardext.npot == 1 &&
                     ((bound->base_level <= 0 && bound->max_level == 0) || (globals4es.automipmap == 3) ||
                      (globals4es.automipmap == 4 && width != height) || (globals4es.forcenpot == 1)) &&
                     (wrap_npot(bound->sampler.wrap_s) && wrap_npot(bound->sampler.wrap_t)))
                limitednpot = 1;
            else if (hardext.esversion > 1 && hardext.npot == 1 &&
                     (!bound->mipmap_auto || !minmag_npot(bound->sampler.min_filter) ||
                      !minmag_npot(bound->sampler.mag_filter)) &&
                     (wrap_npot(bound->sampler.wrap_s) && wrap_npot(bound->sampler.wrap_t)))
                limitednpot = 1;
            else if (hardext.esversion > 1 && hardext.npot == 2 &&
                     (wrap_npot(bound->sampler.wrap_s) && wrap_npot(bound->sampler.wrap_t)))
                limitednpot = 1;

            if (limitednpot) {
                nwidth = width;
                nheight = height;
            }
        }
#ifdef PANDORA
#define NO_1x1
#endif
#ifdef NO_1x1
#define MIN_SIZE 2
        if (level == 0 && hardext.esversion == 1) {
            if (nwidth < MIN_SIZE && nheight < MIN_SIZE) {
                nwidth = MIN_SIZE;
                nheight = MIN_SIZE;
            }
        }
#undef MIN_SIZE
#endif
        if (globals4es.texstream && bound->streamed) {
            nwidth = width;
            nheight = height;
        }

        if (bound->npot) {
            if (limitednpot && rtarget == GL_TEXTURE_2D) {
                bound->sampler.wrap_t = bound->sampler.wrap_s = GL_CLAMP_TO_EDGE;
            } else if (!wrap_npot(bound->sampler.wrap_s) || !wrap_npot(bound->sampler.wrap_t)) {
                // resize to npot boundaries (not ideal if the wrap value is change after upload of the texture)
                if (level == 0 || bound->width == 0) {
                    nwidth = npot(width);
                    nheight = npot(height);
                } else {
                    nwidth = npot(nlevel(bound->width, level));
                    nheight = npot(nlevel(bound->height, level));
                }
                float ratiox, ratioy;
                ratiox = ((float)nwidth) / width;
                ratioy = ((float)nheight) / height;

                GLvoid* out = pixels;
                if (pixels) pixel_scale(pixels, &out, width, height, nwidth, nheight, format, type);
                if (out != pixels && pixels != datab) free(pixels);
                pixels = out;
                width = nwidth;
                height = nheight;
                limitednpot = 0;

                if (level == 0) {
                    if (!bound->useratio) {
                        bound->useratio = 1;
                        if (bound->ratiox == 0.f) bound->ratiox = bound->ratioy = 1.0f;
                    }
                    bound->ratiox *= ratiox;
                    bound->ratioy *= ratioy;
                    bound->npot = 0;
                    bound->shrink = 1;
                }
            }
        }
        if ((globals4es.automipmap == 4) && (nwidth != nheight)) bound->mipmap_auto = 0;

        if (level == 0) {
            bound->width = width;
            bound->height = height;
            bound->nwidth = nwidth;
            bound->nheight = nheight;
            if (target == GL_TEXTURE_RECTANGLE_ARB && hardext.esversion == 2) {
                bound->adjust = 0; // because this test is used in a lot of places
                bound->adjustxy[0] = 1.0f / width;
                bound->adjustxy[1] = 1.0f / height;
            } else {
                // TEXTURE_RECTANGLE could be mutualize with npot texture probably
                bound->adjust = (width != nwidth || height != nheight);
                bound->adjustxy[0] = (float)width / nwidth;
                bound->adjustxy[1] = (float)height / nheight;
            }
            bound->compressed = 0;
            bound->valid = 1;
        }

        int callgeneratemipmap = 0;
        if (!(globals4es.texstream && bound->streamed)) {
            if ((target != GL_TEXTURE_RECTANGLE_ARB) && (globals4es.automipmap != 3) &&
                (bound->mipmap_need || bound->mipmap_auto) && !(bound->npot && hardext.npot < 2) &&
                (bound->max_level == -1)) {
                if (hardext.esversion < 2)
                    gles_glTexParameteri(rtarget, GL_GENERATE_MIPMAP, GL_TRUE);
                else
                    callgeneratemipmap = 1;
            } else {
                // if(target!=GL_TEXTURE_RECTANGLE_ARB) gles_glTexParameteri( rtarget, GL_GENERATE_MIPMAP, GL_FALSE );
                if ((itarget != ENABLED_CUBE_MAP && target != GL_TEXTURE_RECTANGLE_ARB) &&
                    (bound->mipmap_need || globals4es.automipmap == 3)) {
                    // remove the need for mipmap...
                    bound->mipmap_need = 0;
                    gl4es_glTexParameteri(rtarget, GL_TEXTURE_MIN_FILTER,
                                          bound->sampler.min_filter); // forcing min_filter to be recomputed
                }
            }

            if (height != nheight || width != nwidth) {
                errorGL();
                gles_glTexImage2D(rtarget, level, internalformat, nwidth, nheight, border, format, type, NULL);
                DBG(CheckGLError(1);)
                if (pixels) {
                    gles_glTexSubImage2D(rtarget, level, 0, 0, width, height, format, type, pixels);
                    DBG(CheckGLError(1);)
                }
#ifdef NO_1x1
                if (level == 0 && (width == 1 && height == 1 && pixels)) {
                    // complete the texture, juste in case it use GL_REPEAT
                    // also, don't keep the fact we have resized, the non-adjusted coordinates will work (as the texture
                    // is enlarged)
                    if (width == 1) {
                        gles_glTexSubImage2D(rtarget, level, 1, 0, width, height, format, type, pixels);
                        nwidth = 1;
                    }
                    if (height == 1) {
                        gles_glTexSubImage2D(rtarget, level, 0, 1, width, height, format, type, pixels);
                        nheight = 1;
                    }
                    if (width == 1 && height == 1) { // create a manual mipmap just in case_state
                        gles_glTexSubImage2D(rtarget, level, 1, 1, width, height, format, type, pixels);
                        gles_glTexImage2D(rtarget, 1, format, 1, 1, 0, format, type, pixels);
                    }
                }
#endif
            } else {
                errorGL();
                gles_glTexImage2D(rtarget, level, internalformat, width, height, border, format, type, pixels);
                DBG(CheckGLError(1);)
            }
            // check if base_level is set... and calculate lower level mipmap
            if (bound->base_level == level && !(bound->max_level == level && level == 0)) {
                int leveln = level, nw = width, nh = height, nww = nwidth, nhh = nheight;
                int pot = (nh == nhh && nw == nww);
                void* ndata = pixels;
                while (leveln) {
                    if (pixels) {
                        GLvoid* out = ndata;
                        pixel_doublescale(ndata, &out, nw, nh, format, type);
                        if (out != ndata && ndata != pixels) free(ndata);
                        ndata = out;
                    }
                    nw <<= 1;
                    nh <<= 1;
                    nww <<= 1;
                    nhh <<= 1;
                    --leveln;
                    gles_glTexImage2D(rtarget, leveln, format, nww, nhh, border, format, type, (pot) ? ndata : NULL);
                    if (!pot && pixels) gles_glTexSubImage2D(rtarget, leveln, 0, 0, nw, nh, format, type, ndata);
                }
                if (ndata != pixels) free(ndata);
            }
            if (globals4es.automipmap == 5 && !level) bound->mipmap_done = 0;
            // check if max_level is set... and calculate higher level mipmap
            if (((bound->max_level == level && (level || bound->mipmap_need)) || (callgeneratemipmap && level == 0) ||
                 (globals4es.automipmap == 5 && level && !bound->mipmap_done)) &&
                !(bound->max_level == bound->base_level && bound->max_level == 0)) {
                if (globals4es.automipmap == 5 && level == 1) bound->mipmap_done = 1;
                int leveln = level, nw = nwidth, nh = nheight, nww = width, nhh = height;
                int pot = (nh == nhh && nw == nww);
                void* ndata = pixels;
                while (nw != 1 || nh != 1) {
                    if (pixels) {
                        GLvoid* out = ndata;
                        pixel_halfscale(ndata, &out, nww, nhh, format, type);
                        if (out != ndata && ndata != pixels) free(ndata);
                        ndata = out;
                    }
                    nw = nlevel(nw, 1);
                    nh = nlevel(nh, 1);
                    nww = nlevel(nww, 1);
                    nhh = nlevel(nhh, 1);
                    ++leveln;
                    gles_glTexImage2D(rtarget, leveln, format, nw, nh, border, format, type, (pot) ? ndata : NULL);
                    if (!pot && pixels) gles_glTexSubImage2D(rtarget, leveln, 0, 0, nww, nhh, format, type, ndata);
                }
                if (ndata != pixels) free(ndata);
            }
            /*if (bound && bound->mipmap_need && !bound->mipmap_auto && (globals4es.automipmap!=3))
                gles_glTexParameteri( rtarget, GL_GENERATE_MIPMAP, GL_FALSE );*/
        } else {
            if (pixels)
                gl4es_glTexSubImage2D(rtarget, level, 0, 0, width, height, format, type,
                                      pixels); // (should never happens) updload the 1st data...
        }
        if (bound->mipmap_need && !bound->mipmap_done) {
            if (bound->mipmap_auto) bound->mipmap_done = 1;
            if (level == maxlevel(bound->width, bound->height)) bound->mipmap_done = 1;
        }
    }
    if ((target == GL_TEXTURE_2D) && globals4es.texcopydata &&
        ((globals4es.texstream && !bound->streamed) || !globals4es.texstream)) {
        if (bound->data)
            bound->data = realloc(bound->data, width * height * 4);
        else
            bound->data = malloc(width * height * 4);
        if (datab) {
            if (!pixel_convert(pixels, &bound->data, width, height, format, type, GL_RGBA, GL_UNSIGNED_BYTE, 0,
                               glstate->texture.unpack_align))
                SHUT_LOGD("LIBGL: Error on pixel_convert when TEXCOPY in glTexImage2D\n");
        } else {
            // memset(bound->data, 0, width*height*4);
        }
    }
    if (pixels != datab) {
        free(pixels);
    }
    // update max bound to be sure "sampler" is applied
    if (glstate->bound_changed < glstate->texture.active + 1) glstate->bound_changed = glstate->texture.active + 1;
}

static size_t pad_to(size_t v, GLint align) {
    if (align <= 1) return v;
    size_t rem = v % (size_t)align;
    return rem ? v + ((size_t)align - rem) : v;
}

void APIENTRY_GL4ES gl4es_glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width,
                                          GLsizei height, GLenum format, GLenum type, const GLvoid* data) {
    if (glstate->list.pending) {
        gl4es_flush();
    } else {
        PUSH_IF_COMPILING(glTexSubImage2D);
    }

    realize_bound(glstate->texture.active, target);

    // ZOMDROID ETC2: a plain TexSubImage into compressed storage is GL_INVALID_OPERATION;
    // with noerror forced it would corrupt silently instead. Refuse and say so -- field
    // data (RC36 sub-upload counter) says PZ re-uploads these atlases whole, so this
    // line firing at all is itself a finding.
    {
        gltexture_t* zeb = glstate->texture.bound[glstate->texture.active][what_target(target)];
        if (zeb && zeb->zetc2_format) {
            extern void zomdroid_gltrace(const char* fmt, ...);
            static int zlog = 0;
            if (zlog < 6) {
                zlog++;
                zomdroid_gltrace("ETC2SUB refused: %dx%d at %d,%d into compressed tex=%u", width,
                                 height, xoffset, yoffset, zeb->texture);
            }
            noerrorShim();
            return;
        }
    }

    if (width <= 0 || height <= 0) {
        return;
    }
    if (!data) {
        SHUT_LOGD("LIBGL: glTexSubImage2D called with NULL data\n");
        return;
    }
    int pixelSize = pixel_sizeof(format, type);
    if (pixelSize <= 0) {
        SHUT_LOGD("LIBGL: invalid pixel size (format/type) in glTexSubImage2D\n");
        return;
    }

    const GLubyte* pixels_src = (const GLubyte*)data;
    GLubyte* temp_pixels = NULL;

    if (glstate->texture.unpack_row_length != 0 || glstate->texture.unpack_skip_pixels != 0 ||
        glstate->texture.unpack_skip_rows != 0 || glstate->texture.unpack_align != 4) {

        size_t ui_width = (size_t)width;
        size_t ui_height = (size_t)height;
        size_t up_row_pixels =
            (glstate->texture.unpack_row_length ? (size_t)glstate->texture.unpack_row_length : ui_width);
        GLint up_align = glstate->texture.unpack_align;
        if (up_align <= 0) up_align = 1;

        size_t src_row_raw = up_row_pixels * (size_t)pixelSize;
        if (up_row_pixels != 0 && src_row_raw / up_row_pixels != (size_t)pixelSize) {
            SHUT_LOGD("LIBGL: overflow src_row_raw\n");
            return;
        }
        size_t src_row_bytes = pad_to(src_row_raw, up_align);

        size_t dst_row_bytes = ui_width * (size_t)pixelSize;
        if (ui_width != 0 && dst_row_bytes / ui_width != (size_t)pixelSize) {
            SHUT_LOGD("LIBGL: overflow dst_row_bytes\n");
            return;
        }

        size_t total_dst = dst_row_bytes * ui_height;
        if (ui_height != 0 && total_dst / ui_height != dst_row_bytes) {
            SHUT_LOGD("LIBGL: overflow total_dst\n");
            return;
        }

        size_t skip_pixels = (size_t)glstate->texture.unpack_skip_pixels;
        size_t skip_rows = (size_t)glstate->texture.unpack_skip_rows;

        size_t skip_pixels_bytes = skip_pixels * (size_t)pixelSize;
        if (skip_pixels != 0 && skip_pixels_bytes / skip_pixels != (size_t)pixelSize) {
            SHUT_LOGD("LIBGL: overflow skip_pixels_bytes\n");
            return;
        }

        size_t skip_rows_bytes = skip_rows * src_row_bytes;
        if (skip_rows != 0 && skip_rows_bytes / skip_rows != src_row_bytes) {
            SHUT_LOGD("LIBGL: overflow skip_rows_bytes\n");
            return;
        }

        if (up_row_pixels < (skip_pixels + ui_width)) {
            SHUT_LOGD("LIBGL: unpack_row_length (%zu) too small for skip+width (%zu)\n", up_row_pixels,
                      (size_t)(skip_pixels + ui_width));
            return;
        }

        size_t src_offset = skip_rows_bytes + skip_pixels_bytes;
        if (src_offset < skip_rows_bytes || src_offset < skip_pixels_bytes) {
            SHUT_LOGD("LIBGL: overflow src_offset\n");
            return;
        }

        temp_pixels = (GLubyte*)malloc(total_dst);
        if (!temp_pixels) {
            SHUT_LOGD("LIBGL: malloc failed in glTexSubImage2D (bytes=%zu)\n", total_dst);
            return;
        }

        const GLubyte* src_base = (const GLubyte*)pixels_src;
        const GLubyte* src_start = src_base + src_offset;

        size_t y;
        for (y = 0; y < ui_height; ++y) {
            const GLubyte* src_row = src_start + y * src_row_bytes;
            GLubyte* dst_row = temp_pixels + y * dst_row_bytes;
            memcpy(dst_row, src_row, dst_row_bytes);
        }

        pixels_src = (const GLubyte*)temp_pixels;
    }

    // ZOMDROID SHRINK FIX (black patches under LIBGL_SHRINK, 2026-08-07): this
    // rewritten TexSubImage never learned about shrunk textures — patches were sent
    // with FULL-SIZE coordinates into a HALF-SIZE texture, the driver rejected them
    // (silently under noerror), and every partially-painted object (vehicle skins,
    // farming plots) stayed black for anyone running the community LIBGL_SHRINK=1
    // advice. Scale offsets/dimensions by the texture's shrink factor and downscale
    // the pixels to match.
    GLvoid* zshrunk = NULL;
    {
        gltexture_t* zsb = glstate->texture.bound[glstate->texture.active][what_target(target)];
        int zf = 0;
        if (zsb) {
            if (zsb->shrink > 0) zf = zsb->shrink;
            else if (zsb->useratio) {
                if (zsb->ratiox <= 0.26f) zf = 2;
                else if (zsb->ratiox <= 0.51f) zf = 1;
            }
        }
        if (zf > 0 && type == GL_UNSIGNED_BYTE &&
            (format == GL_RGBA || format == GL_RGB || format == GL_BGRA || format == GL_LUMINANCE)) {
            GLvoid* zcur = (GLvoid*)pixels_src;
            int zw = width, zh = height;
            for (int zi = 0; zi < zf && zw > 1 && zh > 1; zi++) {
                GLvoid* zout = zcur;
                if (!pixel_halfscale(zcur, &zout, zw, zh, format, type)) break;
                if (zcur != (GLvoid*)pixels_src && zcur != zout) free(zcur);
                zcur = zout;
                zw = (zw + 1) / 2;
                zh = (zh + 1) / 2;
            }
            if (zcur != (GLvoid*)pixels_src) {
                zshrunk = zcur;
                pixels_src = (const GLubyte*)zcur;
                xoffset >>= zf;
                yoffset >>= zf;
                width = zw;
                height = zh;
                static int zshr_logged = 0;
                if (zshr_logged < 3) {
                    zshr_logged++;
                    zomdroid_gltrace("SHRINK sub-upload rescaled to %dx%d at %d,%d (factor %d)", width, height,
                                     xoffset, yoffset, zf);
                }
            }
        }
    }
    // ZOMDROID T16 (black-vehicles fix): if this texture is stored in one of our 16-bit
    // formats, the sub-upload has to speak the same format or the driver drops it. PZ
    // builds vehicle skins exactly this way — empty 16-bit atlas, then RGBA8 patches.
    GLvoid* zconv = NULL;
    {
        gltexture_t* zb = glstate->texture.bound[glstate->texture.active][what_target(target)];
        if (zb && zb->zt16_type && (format != zb->zt16_format || type != zb->zt16_type) &&
            type == GL_UNSIGNED_BYTE && (format == GL_RGBA || format == GL_RGB || format == GL_BGRA)) {
            if (pixel_convert(pixels_src, &zconv, width, height, format, type, zb->zt16_format, zb->zt16_type, 0,
                              glstate->texture.unpack_align) &&
                zconv) {
                pixels_src = (const GLubyte*)zconv;
                format = zb->zt16_format;
                type = zb->zt16_type;
                static int zsub_logged = 0;
                if (zsub_logged < 3) {
                    zsub_logged++;
                    zomdroid_gltrace("T16 sub-upload %dx%d converted to texture storage format", width, height);
                }
            }
        }
    }

    LOAD_GLES2(glTexSubImage2D);
    gles_glTexSubImage2D(target, level, xoffset, yoffset, width, height, format, type, (const GLvoid*)pixels_src);

    if (zconv) free(zconv);
    if (zshrunk) free(zshrunk);
    if (temp_pixels) free(temp_pixels);
}

// 1d stubs
void APIENTRY_GL4ES gl4es_glTexImage1D(GLenum target, GLint level, GLint internalFormat, GLsizei width, GLint border,
                                       GLenum format, GLenum type, const GLvoid* data) {

    // TODO: maybe too naive to force GL_TEXTURE_2D here?
    gl4es_glTexImage2D(GL_TEXTURE_1D, level, internalFormat, width, 1, border, format, type, data);
}
void APIENTRY_GL4ES gl4es_glTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format,
                                          GLenum type, const GLvoid* data) {

    gl4es_glTexSubImage2D(GL_TEXTURE_1D, level, xoffset, 0, width, 1, format, type, data);
}

GLboolean APIENTRY_GL4ES gl4es_glIsTexture(GLuint texture) {
    DBG(SHUT_LOGD("glIsTexture(%d):", texture);)
    if (!glstate) {
        DBG(SHUT_LOGD("GL_FALSE\n");) return GL_FALSE;
    }
    noerrorShim();
    if (!texture) {
        DBG(SHUT_LOGD("%s\n", glstate->texture.zero->valid ? "GL_TRUE" : "GL_FALSE");)
        return glstate->texture.zero->valid;
    }
    khint_t k;
    khash_t(tex)* list = glstate->texture.list;
    if (!list) {
        DBG(SHUT_LOGD("GL_FALSE\n");)
        return GL_FALSE;
    }
    k = kh_get(tex, list, texture);
    gltexture_t* tex = NULL;
    if (k == kh_end(list)) {
        DBG(SHUT_LOGD("GL_FALSE\n");)
        return GL_FALSE;
    }
    DBG(SHUT_LOGD("GL_TRUE\n");)
    return GL_TRUE;
}

void APIENTRY_GL4ES gl4es_glTexStorage1D(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width) {
    DBG(SHUT_LOGD("glTexStorage1D(%s, %d, %s, %d)\n", PrintEnum(target), levels, PrintEnum(internalformat), width);)
    gl4es_glTexImage1D(target, 0, internalformat, width, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}
void APIENTRY_GL4ES gl4es_glTexStorage2D(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width,
                                         GLsizei height) {
    // (could be implemented in GLES3.0)
    DBG(SHUT_LOGD("glTexStorage2D(%s, %d, %s, %d, %d)\n", PrintEnum(target), levels, PrintEnum(internalformat), width,
                  height);)
    if (!levels) {
        noerrorShim();
        return;
    }
    if ((internalformat == GL_COMPRESSED_RGB_S3TC_DXT1_EXT || internalformat == GL_COMPRESSED_SRGB_S3TC_DXT1_EXT) &&
        !globals4es.avoid16bits)
        gl4es_glTexImage2D(target, 0, internalformat, width, height, 0, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, NULL);
    else if (((internalformat == GL_COMPRESSED_RGBA_S3TC_DXT1_EXT ||
               internalformat == GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT)) &&
             !globals4es.avoid16bits)
        gl4es_glTexImage2D(target, 0, internalformat, width, height, 0, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, NULL);
    else if ((internalformat == GL_COMPRESSED_RGBA_S3TC_DXT3_EXT ||
              internalformat == GL_COMPRESSED_RGBA_S3TC_DXT5_EXT ||
              internalformat == GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT ||
              internalformat == GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT) &&
             !globals4es.avoid16bits)
        gl4es_glTexImage2D(target, 0, internalformat, width, height, 0, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4, NULL);
    else
        gl4es_glTexImage2D(target, 0, internalformat, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    int mlevel = maxlevel(width, height);
    gltexture_t* bound = gl4es_getCurrentTexture(target);
    if (levels > 1 && isDXTc(internalformat)) {
        // no mipmap will be uploaded, but they will be calculated from level 0
        bound->mipmap_need = 1;
        bound->mipmap_auto = 1;
        for (int i = 1; i <= mlevel; ++i)
            gl4es_glTexImage2D(target, i, internalformat, nlevel(width, i), nlevel(height, i), 0, bound->format,
                               bound->type, NULL);
        noerrorShim();
        return;
    }
    // no more compressed format here...
    if (mlevel > levels - 1) {
        bound->max_level = levels - 1;
        if (levels > 1 && globals4es.automipmap != 3) bound->mipmap_need = 1;
    }

    for (int i = 1; i < levels; ++i)
        gl4es_glTexImage2D(target, i, internalformat, nlevel(width, i), nlevel(height, i), 0, bound->format,
                           bound->type, NULL);

    noerrorShim();
}

// Direct wrapper
AliasExport(void, glTexImage2D, ,
            (GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLint border,
             GLenum format, GLenum type, const GLvoid* data));
AliasExport(void, glTexImage1D, ,
            (GLenum target, GLint level, GLint internalFormat, GLsizei width, GLint border, GLenum format, GLenum type,
             const GLvoid* data));
AliasExport(void, glTexSubImage2D, ,
            (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format,
             GLenum type, const GLvoid* data));
AliasExport(void, glTexSubImage1D, ,
            (GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const GLvoid* data));
AliasExport(GLboolean, glIsTexture, , (GLuint texture));

// TexStorage
AliasExport(void, glTexStorage1D, , (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width));
AliasExport(void, glTexStorage2D, ,
            (GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height));
