#ifndef ZOMDROID_ETC2_H
#define ZOMDROID_ETC2_H
#include <stdint.h>
#include <stddef.h>
// Encode w*h pixels (stride 4 = RGBA, 3 = RGB treated as opaque) into ETC2 blocks.
// etc2fmt: GL_COMPRESSED_RGBA8_ETC2_EAC (0x9278) or GL_COMPRESSED_RGB8_ETC2 (0x9274).
// Returns a thread-local buffer valid until the next call, NULL on failure; *out_sz = bytes.
const void* zomdroid_etc2_encode(uint32_t etc2fmt, int32_t w, int32_t h, const uint8_t* rgba, int stride,
                                 size_t* out_sz);
uint64_t zomdroid_etc2_total_ms(void); // cumulative wall time inside the encoder
// Same contract as zomdroid_etc2_encode, but backed by the on-disk cache: a content hash
// of the pixels keys one file per texture; hits skip encoding entirely.
const void* zomdroid_etc2_cached(uint32_t etc2fmt, int32_t w, int32_t h, const uint8_t* rgba, int stride,
                                 size_t* out_sz);
uint64_t zomdroid_etc2_io_ms(void); // cumulative hash+read+write time of the cache
#endif
