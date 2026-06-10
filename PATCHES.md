# Required patches to the sibling `magenta-realtime` checkout

MRT2-Accompany builds `magentart::core` from a sibling checkout at
`~/Code/magenta-realtime` (override with `MAGENTA_RT_DIR`). That checkout needs the
local patches below. **They are not part of that repo's git history** — re-apply
them whenever magenta-realtime is freshly cloned or reset.

```sh
./scripts/apply-magenta-patch.sh        # idempotent per patch; respects $MAGENTA_RT_DIR
```

## `patches/magenta-musiccoca-thread-join.patch` — MusicCoCa worker thread lifetime

`MLXEngine::Impl::start_inference_thread_if_needed()` spawned the MusicCoCa
encode worker as a **detached** `std::thread` capturing `this` raw, and nothing
synchronized it with destruction: `~Impl()` ran `unload()` (freeing the TFLite
interpreters the thread was actively using) and then destroyed the mutexes the
thread locks via `add_log`. Removing the plugin from a host while an encode was
in flight aborted the host (observed: Ableton 12 crash, `std::mutex::lock()` on
a destroyed mutex → `std::system_error` → `abort`). MRT2-Accompany makes this
window wide: the loop-latent style blend runs a multi-second audio encode at
every context re-ground.

The patch keeps the thread handle (`musiccoca_thread_`), reaps the previous
thread before spawning a new one, and joins it in `~Impl()` before `unload()`.
Worst case, plugin removal now blocks for one encode (~seconds) instead of
crashing the host.

## `patches/magenta-encoder-28s.patch` — SpectroStream encoder input length

`core/src/mlx_engine.cpp` hardcodes `kEncoderInputSamples = 2880000` (60 s) at two
sites, but the **released `spectrostream_encoder.mlxfn` is traced at 28 s
(1,344,000 samples)**. The mismatch makes `prefill_state` throw
`expected (1,1344000,2), called (1,2880000,2)` and breaks all prefill.

The proper fix (re-export at 60 s via `mrt mlx export-spectrostream`) is impossible
here: it requires the raw JAX checkpoint `checkpoints/mrt2_base.safetensors`, which
404s on HuggingFace and is `gcloud`-gated on GCS. The patch sets both sites to
`1344000`. 28 s ≫ the model's ~19.7 s receptive field, so prefill quality is
unaffected; the 25/25-frame trim defaults are length-independent and stay valid.
