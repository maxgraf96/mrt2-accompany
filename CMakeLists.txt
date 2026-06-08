cmake_minimum_required(VERSION 3.27)

# OBJCXX is required: magentart::core's autorelease pool + the reused Obj-C++
# downloader/paths helpers. C is required by the native deps (TFLite).
project(mrt2_accompany VERSION 0.1.0 LANGUAGES C CXX OBJCXX)

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

# ── #4/#5/#6/#7: analyzer, mapper, host grid, loop capture (no engine dep) ────
add_library(mrt2_analyzer STATIC
    src/KeyChordAnalyzer.cpp
    src/Mrt2ControlMapper.cpp
    src/HostGridClock.cpp
    src/LoopCapture.cpp)
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

add_executable(test_grid tests/test_grid.cpp)
target_link_libraries(test_grid PRIVATE mrt2_analyzer)
add_test(NAME grid COMMAND test_grid)

# ── M3: chord->MIDI conditioning end-to-end (engine + analyzer + mapper) ──────
add_executable(m3_condition tools/m3_condition.cpp)
target_include_directories(m3_condition PRIVATE "${MAGENTA_COMMON_DIR}/cpp")
target_sources(m3_condition PRIVATE ${MAGENTA_PATHS_SRC})
target_link_libraries(m3_condition PRIVATE magentart::core mrt2_analyzer)
magenta_deploy_metallib(m3_condition)

# ── M4: JUCE plugin (AU / VST3 / Standalone) ─────────────────────────────────
option(MRT2_BUILD_PLUGIN "Build the JUCE AU/VST3/Standalone plugin" ON)
if(MRT2_BUILD_PLUGIN)
    FetchContent_Declare(juce
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG        8.0.4
        GIT_SHALLOW    ON)
    message(STATUS "Fetching JUCE 8.0.4...")
    FetchContent_MakeAvailable(juce)

    juce_add_plugin(MRT2Accompany
        VERSION 0.1.0
        PRODUCT_NAME "MRT2-Accompany"
        COMPANY_NAME "MRT2"
        BUNDLE_ID com.mrt2.accompany
        PLUGIN_MANUFACTURER_CODE Mrt2
        PLUGIN_CODE Mac1
        FORMATS AU VST3 Standalone
        IS_SYNTH FALSE
        NEEDS_MIDI_INPUT FALSE
        NEEDS_MIDI_OUTPUT FALSE
        IS_MIDI_EFFECT FALSE
        COPY_PLUGIN_AFTER_BUILD TRUE)

    target_sources(MRT2Accompany PRIVATE
        src/PluginProcessor.cpp
        src/PluginEditor.cpp
        src/AccompanyRunner.cpp
        ${MAGENTA_PATHS_SRC})
    target_include_directories(MRT2Accompany PRIVATE src "${MAGENTA_COMMON_DIR}/cpp")
    target_compile_definitions(MRT2Accompany PRIVATE
        JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0 JUCE_DISPLAY_SPLASH_SCREEN=0)
    target_link_libraries(MRT2Accompany PRIVATE
        magentart::core mrt2_analyzer
        juce::juce_audio_utils
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags)

    # mlx.metallib must sit next to each plugin binary. JUCE's COPY_PLUGIN_AFTER_BUILD
    # installs the bundle BEFORE our post-build runs, so we copy the metallib into
    # both the build artefact AND the standard user install location per format.
    set(_metallib "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib")
    set(_install_AU  "$ENV{HOME}/Library/Audio/Plug-Ins/Components/MRT2-Accompany.component/Contents/MacOS")
    set(_install_VST3 "$ENV{HOME}/Library/Audio/Plug-Ins/VST3/MRT2-Accompany.vst3/Contents/MacOS")
    foreach(fmt Standalone AU VST3)
        if(TARGET MRT2Accompany_${fmt})
            add_custom_command(TARGET MRT2Accompany_${fmt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_metallib}" "$<TARGET_FILE_DIR:MRT2Accompany_${fmt}>/mlx.metallib"
                VERBATIM)
            if(DEFINED _install_${fmt})
                add_custom_command(TARGET MRT2Accompany_${fmt} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${_install_${fmt}}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_metallib}" "${_install_${fmt}}/mlx.metallib"
                    VERBATIM)
            endif()
        endif()
    endforeach()

    # ── M5: realtime-paced host harness for drift measurement (headless) ─────
    juce_add_console_app(m5_drift PRODUCT_NAME "m5_drift")
    target_sources(m5_drift PRIVATE
        tools/m5_drift.cpp
        src/PluginProcessor.cpp src/PluginEditor.cpp src/AccompanyRunner.cpp
        ${MAGENTA_PATHS_SRC})
    target_include_directories(m5_drift PRIVATE src "${MAGENTA_COMMON_DIR}/cpp")
    target_compile_definitions(m5_drift PRIVATE
        JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_STANDALONE_APPLICATION=1)
    target_link_libraries(m5_drift PRIVATE
        magentart::core mrt2_analyzer
        juce::juce_audio_utils juce::juce_recommended_config_flags)
    add_custom_command(TARGET m5_drift POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_metallib}" "$<TARGET_FILE_DIR:m5_drift>/mlx.metallib" VERBATIM)
endif()
