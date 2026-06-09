#include "AssetManager.h"
#include "magenta_paths.h"

#import "MagentaModelDownloader.h"
#import <Foundation/Foundation.h>

#include <cstdlib>
#include <filesystem>

namespace mrt2 {

static bool model_files_present() {
    namespace fs = std::filesystem;
    const std::string dir = magentart::paths::get_default_model_dir();
    std::error_code ec;
    return fs::exists(dir + "/mrt2_base.mlxfn", ec) &&
           fs::exists(dir + "/mrt2_base_state.safetensors", ec);
}

bool AssetManager::assets_ready() {
    if (std::getenv("MRT2_FORCE_DOWNLOAD")) return false;  // UI testing hook
    return [MagentaModelDownloader areSharedResourcesValid] && model_files_present();
}

void AssetManager::start_download(std::function<void(float, const std::string&)> progress,
                                  std::function<void(bool)> done) {
    auto prog = std::make_shared<std::function<void(float, const std::string&)>>(std::move(progress));
    auto fin  = std::make_shared<std::function<void(bool)>>(std::move(done));

    auto report = [prog](float p, NSString* s) {
        (*prog)(p, std::string(s ? s.UTF8String : ""));
    };

    // Phase 2: the model (the bulk of the download). Skipped if already present.
    auto downloadModel = [report, fin]() {
        if (model_files_present()) { (*fin)(true); return; }
        [MagentaModelDownloader downloadModel:@"mrt2_base"
            progress:^(double p, NSString* status) { report(0.15f + 0.85f * (float)p, status); }
            completion:^(BOOL ok, NSError* err) {
                (void)err; (*fin)((bool)ok);
            }];
    };

    // Phase 1: shared resources (MusicCoCa + SpectroStream). Skipped if valid.
    if ([MagentaModelDownloader areSharedResourcesValid]) {
        report(0.15f, @"resources ready");
        downloadModel();
    } else {
        [MagentaModelDownloader initializeSharedResourcesWithProgress:^(double p, NSString* status) {
                report(0.15f * (float)p, status);
            }
            completion:^(BOOL ok, NSError* err) {
                (void)err;
                if (!ok) { (*fin)(false); return; }
                downloadModel();
            }];
    }
}

}  // namespace mrt2
