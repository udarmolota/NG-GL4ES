cmake_minimum_required(VERSION 3.10)

project(ng_gl4es)

enable_language(CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ZOMDROID: Android 15+/Google Play require 16 KB page-size support — all ELF LOAD
# segments must be aligned to 16384. The prebuilt libspirv-cross-c-shared.so in
# library/android/arm64-v8a is likewise a 16 KB-aligned rebuild (SPIRV-Cross
# vulkan-sdk-1.4.313.0).
if(ANDROID)
    add_link_options("-Wl,-z,max-page-size=16384")
endif()

# ZOMDROID CANARY: the protector is deliberately ON to make the known shaderconv
# stack smash abort AT the guilty function (tombstone names it) instead of silently
# corrupting the process (suspected cause of the Mali/MediaTek death at Bullet.init).
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -DNO_GBM -DDEFAULT_ES=3 -fvisibility=hidden -fstack-protector-strong")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-strong")

if(ANDROID)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNOX11")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include )

set(C_SRC_FILES
        src/gl/string_utils.c
        src/gl/cJSON.c
        src/gl/config.c
        src/gl/arbconverter.c
        src/gl/arbgenerator.c
        src/gl/arbhelper.c
        src/gl/arbparser.c
        src/gl/array.c
        src/gl/blend.c
        src/gl/blit.c
        src/gl/buffers.c
        src/gl/build_info.c
        src/gl/debug.c
        src/gl/decompress.c
        src/gl/depth.c
        src/gl/directstate.c
        src/gl/drawing.c
        src/gl/enable.c
        src/gl/envvars.c
        src/gl/eval.c
        src/gl/face.c
        src/gl/fog.c
        src/gl/fpe.c
        src/gl/fpe_cache.c
        src/gl/fpe_shader.c
        src/gl/framebuffers.c
        src/gl/gl_lookup.c
        src/gl/gl4es.c
        src/gl/glstate.c
        src/gl/hint.c
        src/gl/init.c
        src/gl/light.c
        src/gl/line.c
        src/gl/list.c
        src/gl/listdraw.c
        src/gl/listrl.c
        src/gl/loader.c
        src/gl/logs.c
        src/gl/matrix.c
        src/gl/matvec.c
        src/gl/oldprogram.c
        src/gl/pixel.c
        src/gl/planes.c
        src/gl/pointsprite.c
        src/gl/preproc.c
        src/gl/program.c
        src/gl/pridroid_diag.c
        src/gl/queries.c
        src/gl/raster.c
        src/gl/render.c
        src/gl/shader.c
        src/gl/shaderconv.c
        src/gl/shader_hacks.c
        src/gl/stack.c
        src/gl/stencil.c
        src/gl/stubs.c
        src/gl/texenv.c
        src/gl/texgen.c
        src/gl/texture.c
        src/gl/texture_compressed.c
        src/gl/texture_params.c
        src/gl/texture_read.c
        src/gl/zomdroid_etc2.c
        src/gl/texture_3d.c
        src/gl/uniform.c
        src/gl/vertexattrib.c
        src/gl/wrap/gl4eswraps.c
        src/gl/wrap/gles.c
        src/gl/wrap/glstub.c
        src/gl/wrap/glesnative.cpp
        src/gl/math/matheval.c
        src/glx/hardext.c
        src/glx/lookup.c
        src/glx/gbm.c
        src/glx/streaming.c
        src/gl/vgpu/shaderconv.c
        src/gl/samplers.c
        src/gl/libtxc_dxtn/txc_compress_dxtn.c
        )

if(NOT ANDROID)
    list(APPEND C_SRC_FILES src/glx/glx.c)
endif()

set(CPP_SRC_FILES
        src/gl/glsl/glsl_for_es.cpp
        src/gl/getter.cpp
        )

set(SRC_FILES ${CPP_SRC_FILES} ${C_SRC_FILES})

find_library(GLSLANG_LIB glslang PATHS ${CMAKE_CURRENT_SOURCE_DIR}/libraries/arm64-v8a/)

add_library(ng_gl4es SHARED ${SRC_FILES})

if(ANDROID)
    # Android
    find_library(GLSLANG_LIB glslang PATHS ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/)
    target_link_libraries(ng_gl4es
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libglslang.a
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libspirv-cross-c-shared.so
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libGenericCodeGen.a
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libglslang-default-resource-limits.a
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libMachineIndependent.a
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libOSDependent.a
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libSPIRV-Tools-diff.a
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libSPIRV-Tools-link.a
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libSPIRV-Tools-lint.a
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libSPIRV-Tools-opt.a
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libSPIRV-Tools-reduce.a
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libSPIRV-Tools.a
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libSPIRV.a
            ${CMAKE_CURRENT_SOURCE_DIR}/library/android/${ANDROID_ABI}/libSPVRemapper.a
        android
        log
    )
endif()

target_include_directories(ng_gl4es PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
