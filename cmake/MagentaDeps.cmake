# MagentaDeps.cmake — fetch the native deps that magentart::core needs, then add
# the core library, WITHOUT pulling in the magenta-realtime root CMakeLists (which
# unconditionally builds all 7 example apps + runs npm). The three FetchContent
# blocks below are replicated verbatim from
#   ~/Code/magenta-realtime/CMakeLists.txt:60-119
# (MLX v0.31.1 + space-path patch, SentencePiece v0.2.0, TFLite v2.21.0 + the
# FETCHCONTENT_SOURCE_DIR_TENSORFLOW workaround). Keep in sync if that repo bumps.

include(FetchContent)

# CMake 4.x rejects projects declaring cmake_minimum_required < 3.5 (some TFLite
# subdeps do). Allow them.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "" FORCE)

if(NOT DEFINED MAGENTA_RT_DIR)
    set(MAGENTA_RT_DIR "$ENV{HOME}/Code/magenta-realtime" CACHE PATH
        "Path to the magenta-realtime checkout providing core/")
endif()
if(NOT EXISTS "${MAGENTA_RT_DIR}/core/CMakeLists.txt")
    message(FATAL_ERROR "MAGENTA_RT_DIR='${MAGENTA_RT_DIR}' has no core/CMakeLists.txt")
endif()

# --- MLX (static, from source) ------------------------------------------------
FetchContent_Declare(
    mlx
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
    GIT_TAG        v0.31.1
    GIT_SHALLOW    ON
)
set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_PYTHON_STUBS OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_GGUF OFF CACHE BOOL "" FORCE)
set(MLX_BUILD_CUDA OFF CACHE BOOL "" FORCE)
message(STATUS "Fetching MLX... (this may take a moment)")
FetchContent_MakeAvailable(mlx)

# Patch MLX header inclusion to support spaces in path.
# TODO: Remove once https://github.com/ml-explore/mlx/pull/3607 is merged.
file(READ "${mlx_SOURCE_DIR}/mlx/backend/metal/make_compiled_preamble.sh" MLX_CONTENT)
string(REPLACE
  "declare -a HDRS_LIST=($HDRS)"
  "declare -a HDRS_LIST=(); while read -r dots path; do [ -n \"$dots\" ] && HDRS_LIST+=(\"$dots\" \"$path\"); done <<< \"$HDRS\""
  MLX_CONTENT
  "${MLX_CONTENT}"
)
file(WRITE "${mlx_SOURCE_DIR}/mlx/backend/metal/make_compiled_preamble.sh" "${MLX_CONTENT}")

# --- SentencePiece ------------------------------------------------------------
FetchContent_Declare(
  sentencepiece
  GIT_REPOSITORY https://github.com/google/sentencepiece.git
  GIT_TAG        v0.2.0
  GIT_SHALLOW    ON
)
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
message(STATUS "Fetching sentencepiece... (this may take a moment)")
FetchContent_MakeAvailable(sentencepiece)

# --- TensorFlow Lite ----------------------------------------------------------
FetchContent_Declare(
  tensorflow-lite
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG        v2.21.0
  SOURCE_SUBDIR  tensorflow/lite
  GIT_SHALLOW    ON
)
set(TFLITE_ENABLE_XNNPACK OFF CACHE BOOL "Disable XNNPACK delegate" FORCE)
set(TFLITE_ENABLE_GPU OFF CACHE BOOL "Disable GPU delegate" FORCE)
set(TFLITE_BUILD_TESTS OFF CACHE BOOL "Disable tests" FORCE)
# Point TF Lite's internal TF fetch at our already-cloned v2.21.0 source so it
# doesn't clone a second (older) TF and break FlatBuffers compatibility.
set(FETCHCONTENT_SOURCE_DIR_TENSORFLOW "${CMAKE_BINARY_DIR}/_deps/tensorflow-lite-src" CACHE PATH
    "Point TF Lite's internal TF fetch at our already-cloned v2.21.0 source" FORCE)
message(STATUS "Fetching tensorflow-lite... (this may take a long time due to repo size)")
FetchContent_MakeAvailable(tensorflow-lite)

# --- magentart::core (no examples) -------------------------------------------
add_subdirectory("${MAGENTA_RT_DIR}/core" "${CMAKE_BINARY_DIR}/magentart_core")

# Reusable host helpers from the magenta examples (compiled into consumers).
set(MAGENTA_COMMON_DIR "${MAGENTA_RT_DIR}/examples/common")
set(MAGENTA_PATHS_SRC  "${MAGENTA_COMMON_DIR}/cpp/magenta_paths.cpp")
set(MAGENTA_OBJC_SRCS
    "${MAGENTA_COMMON_DIR}/objc/MagentaModelDownloader.mm"
    "${MAGENTA_COMMON_DIR}/objc/MagentaModelManager.mm"
    "${MAGENTA_COMMON_DIR}/objc/MagentaSettings.mm")

# Helper: copy mlx.metallib next to a target's binary + ad-hoc sign it (MLX needs
# the compiled Metal kernels colocated with the executable at runtime).
function(magenta_deploy_metallib target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib"
            "$<TARGET_FILE_DIR:${target}>/mlx.metallib"
        COMMAND codesign --force --sign - "$<TARGET_FILE_DIR:${target}>/mlx.metallib" || true
        VERBATIM)
endfunction()
