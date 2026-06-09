// First-run asset download. Thin C++ wrapper over the reused Obj-C++
// MagentaModelDownloader (examples/common/objc) — fetches the MusicCoCa +
// SpectroStream shared resources and the mrt2_base model from HuggingFace into
// the standard ~/Documents/Magenta/magenta-rt-v2 layout. Implemented in
// AssetManager.mm so the rest of the plugin stays plain C++.

#pragma once
#include <functional>
#include <string>

namespace mrt2 {

class AssetManager {
public:
    // True iff the shared resources AND the mrt2_base model files are all
    // present locally (i.e. the engine can load without downloading).
    static bool assets_ready();

    // Kick off the download on a background queue. `progress` is called with
    // (0..1, status text), `done` with success — both may fire on arbitrary
    // threads. Downloads only the missing pieces (resources and/or model).
    static void start_download(std::function<void(float, const std::string&)> progress,
                               std::function<void(bool)> done);

    // Approximate total download size for the UI (GB).
    static constexpr double kDownloadGB = 3.0;
};

}  // namespace mrt2
