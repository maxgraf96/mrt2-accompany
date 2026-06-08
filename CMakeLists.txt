cmake_minimum_required(VERSION 3.27)

# OBJCXX is required: magentart::core's autorelease pool + the reused Obj-C++
# downloader/paths helpers. C is required by the native deps (TFLite).
project(mrt2_accompany LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "Minimum macOS deployment" FORCE)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(MagentaDeps)   # fetches MLX/SPM/TFLite, adds magentart::core

# ── M0: headless streaming smoke test ────────────────────────────────────────
add_executable(m0_smoke tools/m0_smoke.cpp ${MAGENTA_PATHS_SRC})
target_include_directories(m0_smoke PRIVATE "${MAGENTA_COMMON_DIR}/cpp")
target_link_libraries(m0_smoke PRIVATE magentart::core)
magenta_deploy_metallib(m0_smoke)

# ── M1: offline prefill + conditioning ───────────────────────────────────────
add_executable(m1_prefill tools/m1_prefill.cpp ${MAGENTA_PATHS_SRC})
target_include_directories(m1_prefill PRIVATE "${MAGENTA_COMMON_DIR}/cpp")
target_link_libraries(m1_prefill PRIVATE magentart::core)
magenta_deploy_metallib(m1_prefill)

# ── #6 + #7: analyzer + control mapper (no engine dependency) ────────────────
add_library(mrt2_analyzer STATIC src/KeyChordAnalyzer.cpp src/Mrt2ControlMapper.cpp)
target_include_directories(mrt2_analyzer PUBLIC src)

# ── M2: analyzer validation tool ─────────────────────────────────────────────
add_executable(m2_analyze tools/m2_analyze.cpp)
target_link_libraries(m2_analyze PRIVATE mrt2_analyzer)

enable_testing()
add_executable(test_analyzer tests/test_analyzer.cpp)
target_link_libraries(test_analyzer PRIVATE mrt2_analyzer)
add_test(NAME analyzer COMMAND test_analyzer)

add_executable(test_mapper tests/test_mapper.cpp)
target_link_libraries(test_mapper PRIVATE mrt2_analyzer)
add_test(NAME mapper COMMAND test_mapper)

# ── M3: chord->MIDI conditioning end-to-end (engine + analyzer + mapper) ──────
add_executable(m3_condition tools/m3_condition.cpp)
target_include_directories(m3_condition PRIVATE "${MAGENTA_COMMON_DIR}/cpp")
target_sources(m3_condition PRIVATE ${MAGENTA_PATHS_SRC})
target_link_libraries(m3_condition PRIVATE magentart::core mrt2_analyzer)
magenta_deploy_metallib(m3_condition)
