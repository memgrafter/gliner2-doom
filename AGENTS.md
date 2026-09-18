# gliner2-doom — agent notes

GLiNER2.5 (zero-shot classifier, 287M) plays Doom. Tickets live in `./.tickets/` (this directory, `tk` from here): pro-2tst (e2e build, closed), pro-yjme (direct policy, closed), pro-zekx (multi-map + holdout + RWR, open: results in, no head passed every gate, default stays v3), gd-ngu7 (Metal port, closed: Metal is the default), gd-eok7 (Metal encoder overhead, open). gd-lee5 is the handoff. Read the open ones first.

## Rules

- No system deps. Nothing via brew/apt. Allowed: cargo, uv, Xcode CLT, crates, pip wheels inside a venv, files downloaded into this directory.
- Anything new is Rust. The only Python is `bridge/vizdoom_bridge.py`, which is engine glue with no app logic.
- Only build on maintained libs (freshness table in the ticket). Pin versions.
- Scaffolding and experiments go in `lab/` and get deleted before the ticket closes.

## Python virtualenvs (uv-managed, outside the repo)

All Python envs live in `~/virtualenvs/`, one per purpose, named `gliner2-doom-<purpose>`. Never create a `.venv` inside this directory. Use `uv venv` + `uv pip`, never bare `pip`.

| venv | purpose | create |
|---|---|---|
| `~/virtualenvs/gliner2-doom-vizdoom` | ViZDoom engine for the bridge (vizdoom 1.3.0, pygame-ce, numpy, gymnasium; ~175 MB) | `uv venv ~/virtualenvs/gliner2-doom-vizdoom --python 3.12 && uv pip install --python ~/virtualenvs/gliner2-doom-vizdoom/bin/python vizdoom==1.3.0` |
| `~/virtualenvs/gliner2-doom-torch` | only if a PyTorch cross-check of the ONNX path is ever needed (`gliner2[local]`, ~2.5 GB). Not created by default. | `uv venv ~/virtualenvs/gliner2-doom-torch --python 3.12 && uv pip install --python ~/virtualenvs/gliner2-doom-torch/bin/python "gliner2[local]==2.0.0"` |

Run things with the venv's interpreter directly, no activation needed:

```sh
~/virtualenvs/gliner2-doom-vizdoom/bin/python bridge/vizdoom_bridge.py --scenario defend_the_center
```

The Rust binary spawns the bridge with that interpreter; override with `GLINER2_DOOM_PYTHON=/path/to/python`.

Python 3.12 was chosen because vizdoom 1.3.0 ships macOS arm64 wheels for cp310–cp314 and uv already has 3.12 installed; 3.14 is the system default here and is avoided for ML wheels.

## Model weights and ONNX Runtime

- Weights, Metal encoder (default): `fastino/gliner2.5-multi-v1/model.safetensors` (1.15 GB; only the `encoder.*` tensors are read, mmap) plus `encoder_config/config.json` and `tokenizer.json`, fetched by hf-hub into `~/.cache/huggingface/hub/models--fastino--gliner2.5-multi-v1/`. A complete copy in `models/gliner2.5-multi-v1/` (gitignored) takes precedence. `GLINER2_DOOM_ENCODER_DTYPE=f16` opts into f16 (10% faster, ~6x noisier: never for training caches).
- Weights, ONNX (`--encoder onnx` and the keyword policy): `jugaadsrl/gliner2.5-multi-v1-onnx` (fp16 variant, ~590 MB) fetched by the `gliner25-rs` crate into the shared HF cache (`~/.cache/huggingface`). Set `HF_HOME` to move it.
- ONNX Runtime: the crate loads it dynamically. `just fetch-ort` downloads `onnxruntime-osx-arm64-1.28.2.tgz` into `vendor/onnxruntime/` (gitignored) and the justfile exports `ORT_DYLIB_PATH` to the dylib inside it. No system install.

## Reference clones (read-only, not dependencies)

- `~/clones/open-jev/demo/doom/` — describe.py / game.py semantics we reimplement (repo has no LICENSE: do not copy code).
- `~/clones/openjev/code/` (branch openjev-doom, MIT) — doom_live.py async loop, hypothesis-phrasing results.
- `~/clones/gliner25-rs/` — the crate's source, bench and parity scripts.

## Disk

This machine runs close to full. Build release-only (`cargo build --release`), fetch only the fp16 weights, and do not create the torch venv unless needed.

## gliner25-rs gotchas (measured 2026-09-17)

- **Pin `ExecutionMode::Standard`** (done in `src/scorer.rs`). `Auto` resolves to the bound IoBinding path on macOS, and that path binds routed_gather's `[1,K,768]` output straight into `classifier.onnx`, which declares `[K,768]`: `Invalid rank for input: choice_states`. Standard reshapes on the way through. Candidate upstream issue.
- **CPU beats CoreML here.** fp16, 8 labels: CPU p50 80 ms; `GLINER2_DEVICE=coreml` p50 95 ms. Leave the device on auto/CPU.
- **Latency scales with label count** because every label is in the prompt: 3 labels 72 ms, 8 labels 82 ms, 12 labels 104 ms (p50, fp16, M4). Keep the play-time schema under ~8 labels total.
- **Schema shape:** several small single-label tasks in one call (`scorer::classify_multi`) beat one softmax over many unrelated statements, which comes out flat.
- **Do not ask the model numeric questions** (is ammo low, is health low): it answers "low" for ammo 26 and health 100. Compute thresholds in Rust; ask the model only about things that are genuinely textual.
- RSS: ~1.0 GB right after load, ~420-630 MB steady. Warm load 2.4 s. Weights live in `~/.cache/huggingface/hub/models--jugaadsrl--gliner2.5-multi-v1-onnx/` (578 MB, fp16 only).
- **System One questions: classify the bare state.** Prepending instructions to the text made every support message "furious" and broke urgency. Labels carry the semantics instead: choice labels are `name: description`, score labels are the level descriptions, and a noul question's yes label is `criteria.true` or the instruction text without its `?` (bare "yes"/"no" carry nothing for a label matcher).
- **Keyword fields for game state.** Natural-language descriptions confuse LEFT/RIGHT (30-70%); `Side: RIGHT. Offset: SMALL.` with bare uppercase labels scores 97-100%. Never put shared tokens in sibling labels ("SLIGHTLY LEFT" next to "LEFT" broke side detection); use a separate field instead.

## Status (2026-09-17)

All seven steps of ticket pro-2tst are done; see its notes for every measurement. Open follow-ups: an upstream issue on gliner25-rs (Auto execution mode + classifier rank), a flatmachines YAML machine calling `/v1/systemone`, and turning `--record` JSONL into a trained jevlike head.

## Direct policy pipeline (ticket pro-yjme)

- `collect` appends to `data/{train,val,test}.jsonl`, split by episode index (8 -> val, 9 -> test). Delete `data/` to start over. Rows: `{context, options, label (oracle), action (pressed), reward, ep, tic, scenario}`.
- `train` encodes every distinct context once and caches states per backend: `data*/states.metal.bin` (Metal, default) or `data*/states.bin` (`--encoder onnx`); append-only, f16 on disk, ~5 GB for 85k contexts, gitignored, regenerable (~13 ms/context on Metal). The whole file is read into RAM at train/eval start. Epochs: 53 s on Metal / 108 s on CPU for 77k rows. Best-val checkpoint is kept; temperature is fit on val by NLL grid search and stored in `heads/<name>.json`.
- `src/encoder.rs` is an enum over two backends. Metal (`src/encoder_metal.rs`): the original safetensors under `vb.pp("encoder")` into the vendored `src/debertav2.rs` (candle-transformers 0.11.0 + a three-constant f16 fix; keep the diff against upstream small). ONNX: `encoder_fp16.onnx` through `ort` directly, no label prompt (~34 ms/context). gliner25-rs is used only by the keyword policy and the ONNX download. The direct path needs no `ORT_DYLIB_PATH`.
- Tools: `encoder-parity --reference onnx-fp32` (the bar: max <= 5e-3, mean <= 5e-4 against the fp32 export; the fp16 export and the f16 cache are ~1e-3 noisy themselves, so never use them as the reference), `encoder-bench --bucket` (real contexts, batch sweep). Length-bucketing batches halves padding waste on both backends.
- Vendored DeBERTa (`src/debertav2.rs`) carries gd-eok7's overhead work: cached gather indices per (batch,q,k), one additive mask per forward, position projections computed once per layer. Any edit there: `encoder-parity --reference onnx-fp32` must still PASS (it did at max 2.7e-4). Play is engine-bound at 34.6 decisions/s; the batched path is compute-bound (~245 us per padded token, f32).
- Metal gotcha: `head::device()` is process-wide and shared by the head and the Metal encoder; `--device cpu --encoder metal` runs the DeBERTa port on the CPU, which is slow. The head alone on Metal costs ~1 ms per single-row predict (vs 0.1 ms on CPU) and is worth it only because the encoder is on the same device.
- The head is not `Send` either (it holds nothing special, but `DirectBrain` owns an ORT session), so `Brain` is built on the worker thread like the scorer.
- `--policy direct` disables the stuck-recovery controller and the pulse rules; the option table's `hold` is the only timing.
- Oracle (`state::oracle`) is the label source. Changing it changes what the head imitates; the closed-loop numbers, not offline top-1, are the judge.

## Status (2026-09-18 08:00)

pro-zekx pipeline done (v4 -> v5 DAgger -> v6 RWR, held-out maps map09/map10/map12/e2m1/e2m2): v6 is the best map policy (items 4.7/ep = oracle, kills 3.5, 80% survival) but fails the defend_the_center real-time regression (6.8; v5 8.8; v3 14.0) because the 12 maps outweigh the scenario 5:1 and both heads learned `move forward`. Off-oracle shots fail for every head. Default head stays `heads/v3.safetensors`; the ticket is open with the numbers. gd-ngu7 done: Metal is the default encoder and device (f32), verified by parity, identical head scores, 32.5 decisions/s real-time; latency targets not met, continued in gd-eok7. Old `data*/states.bin` ONNX caches can be deleted once `states.metal.bin` exists.

## Status (2026-09-17 22:10)

pro-yjme done: `--policy direct` with `heads/v3.safetensors` is the default. Heads kept for the record: v1 (oracle data only), v2 (+DAgger 1), v3 (+DAgger 2 on map01), ablation (contexts permuted: proves the head reads the state). `data/` is regenerable in seconds with `just collect`; `data/states.bin` is the encoder cache (rebuild cost ~10 min).
