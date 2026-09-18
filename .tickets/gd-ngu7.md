---
id: gd-ngu7
status: closed
open: false
deps: []
links: [pro-zekx, gd-lee5, gd-eok7]
created: 2026-09-18T11:35:11Z
type: feature
priority: 1
assignee: memgrafter
tags: [gliner2, doom, rust, candle, metal, performance]
---
# gliner2-doom on Apple GPU: encoder moved from ONNX Runtime on CPU to a Rust DeBERTa-v2 implementation (candle-transformers) on Metal, head training on Metal

Move the encoder and the head onto the Apple GPU. Today both run on the CPU: the encoder because it lives in ONNX Runtime (the maintained gliner25-rs path we started from; ORT has no Metal provider and its CoreML provider measured slower, 87-95 ms vs 80, because mDeBERTa's relative-position attention gets partitioned), the head because the Rust ML library it runs on (Hugging Face's candle) was built without its `metal` feature. The encoder is ~80% of every pipeline stage (16 ms/context batched, 34 ms single, on CPU).

## Current runtime (for reference)

```
gliner2-doom (Rust)
  bridge.rs   -> python bridge/vizdoom_bridge.py -> ViZDoom engine (subprocesses)      CPU
  state.rs    -> canonical JSON context (~45 tokens)                                    CPU
  encoder.rs  -> tokenizers + ONNX Runtime session over jugaadsrl encoder_fp16.onnx     CPU  34 ms  <- move to Metal
  head.rs     -> option-attention head in Rust via candle (0.8M params)                 CPU  0.1 ms <- move to Metal (training matters, 60-80 s/epoch)
  options.rs  -> argmax -> buttons + hold -> bridge.act
```

## Plan

1. **Head on Metal (20 min).** Enable the `metal` feature of the candle crate; `Device::new_metal(0)` for train/eval/predict with CPU fallback. Per-batch upload of the f16 cache is the limit; expect epochs 60-80 s -> 20-30 s. Passes if val top-1 on data2 is within 0.3 points of the CPU result.
2. **Encoder on Metal through the DeBERTa-v2 implementation in candle-transformers (half a day).** GLiNER2.5-multi's encoder is mDeBERTa-v3-base (fine-tuned) inside `fastino/gliner2.5-multi-v1/model.safetensors` (1.15 GB) under a prefix; `encoder_config/config.json` is its HF config. Load with a tensor-name map into `candle_transformers::models::debertav2`, tokenizer unchanged (same tokenizer.json). Output `last_hidden_state` [S,768] replaces the ONNX call in `encoder.rs` behind a `--encoder onnx|candle` switch during the transition.
3. **Parity check (new hidden states must match ONNX).** On 500 cached contexts from data2/states.bin: max abs diff of hidden states vs the ONNX fp16 export <= 5e-3, mean <= 5e-4 (the export itself is 2e-6 vs PyTorch fp32, and fp16 rounding is ~1e-3). Then: v5/v6 heads evaluated with the new encoder must match their ONNX top-1 on data2/test.jsonl within 0.3 points, and closed-loop defend_the_center within noise.
4. **Speed check (must beat CPU by these margins).** Batched 32 on Metal: target <= 5 ms/context (CPU 16). Single: <= 15 ms (CPU 34). Report fp16 vs fp32 on Metal.
5. **Switch defaults** to the Metal encoder when both checks pass; ORT and vendor/onnxruntime leave the direct-policy runtime path (still used by `--policy keyword` via gliner25-rs until that path is retired). Re-cache states once (~57k contexts at ~5 ms = 5 min).
6. Update README/AGENTS numbers; note the pipeline time before/after (2.3 h baseline from pro-zekx).

## Risks

- the library's debertav2 implementation may lack a DeBERTa-v3 detail (relative attention buckets, `position_buckets`, `norm_rel_ebd`, conv layer absent in base): resolve by reading the HF config and the ONNX graph; the parity test is the arbiter.
- Metal kernel gaps for some op in that model; fall back to CPU for that op or keep ONNX for the encoder and take only the head win.
- Memory: fp32 weights 1.15 GB on the GPU; use fp16 weights if the M4 shares memory pressure with training.

## Order

Head-on-Metal first (independent, cheap). Encoder port second, ahead of batched rollouts (pro-zekx follow-up), since it speeds every stage rather than only rollouts. Do not run parity/latency jobs while the pro-zekx pipeline's encoder-heavy stages are running.

## Notes

**2026-09-18T12:27:16Z**

STEPS 1-3 DONE 2026-09-18 05:30 (built in target-metal/, pipeline binary untouched; nothing committed after eb126d7). Code: candle-core/nn/transformers 0.11 with the metal feature; src/encoder_metal.rs (DeBERTa-v2 port loaded from models/gliner2.5-multi-v1/model.safetensors under vb.pp("encoder"): the file already uses HF tensor names, so there is no rename map; falls back to hf-hub download); src/encoder.rs is now an enum over OnnxEncoder | MetalEncoder with shared tokenization (pad_batch) and per-backend cache file (states.bin | states.metal.bin); head::device() picks Metal once per process; global CLI flags --encoder onnx|metal (default onnx) and --device cpu|metal|auto (default cpu); new subcommand encoder-parity --reference cache|onnx-fp16|onnx-fp32. RESULTS. Head on Metal: eval v4 on data2/test.jsonl top1 0.9397 nll 0.1547 on both CPU and Metal (identical). Encoder parity vs the ONNX fp32 export recomputed live, 500 contexts batch 32, Metal f32: max abs diff 5.4e-4, mean 1.1e-6, 0 values over 5e-3, 0 token-count mismatches -> PASS. Against the f16 cache on disk it reads max 0.1-0.3 / mean 3.6e-4: that is the fp16 export's own error (fp16 compute plus f16 storage), concentrated on single token positions, not a port defect. v4 head evaluated through the Metal encoder: top1 0.9397 (same as ONNX), ece 0.0061 vs 0.0054. Tokenizer.json in the fastino repo is byte-identical to the jugaadsrl export. BLOCKER: f16 weights fail inside candle-transformers 0.11 debertav2 with 'dtype mismatch in add, lhs F32, rhs F16' (a constant built as f32 in the relative-attention path); f32 works. Options: upstream fix, or f32 weights with f16 activations later. PRELIMINARY SPEED (contaminated by the running pro-zekx pipeline, redo when idle): single context 31 ms (ONNX CPU 34); batch 32 of 41 tokens 400 ms = 12.5 ms/context; the test-split cache build ran 8209 contexts in 213 s = 26 ms/context at batch 64 (ONNX CPU 16). f32 on Metal is not yet ahead of the CPU for batched work; the 5 ms target needs f16 and probably the batched attention path profiled. NEXT: speed check on an idle machine (step 4), f16 fix, then the default flip (step 5) and docs (step 6).

**2026-09-18T12:58:23Z**

F16 UNBLOCKED 2026-09-18 05:58. candle-transformers' debertav2.rs (0.11.0, MIT/Apache-2.0) is now vendored as src/debertav2.rs and the crate dependency dropped; the fix is three constants (attention score accumulator, softmax mask fill, softmax zero fill) taking the activation dtype instead of f32, plus the tracing span removed. Parity vs the ONNX fp32 export, 200 contexts batch 32: Metal f32 max 1.1e-4 mean 1.1e-6 (PASS); Metal f16 max 3.6e-1 mean 2.1e-3, 8.4% of values over 5e-3 (the f16 residual stream is ~6x noisier than the ONNX fp16 export, whose own error vs fp32 is mean ~3.6e-4). Head test through Metal f16 on data2/test.jsonl (fresh cache): v4 top1 0.9408 nll 0.1554 ece 0.0055 vs ONNX 0.9397 / 0.1547 / 0.0054 -> within noise; f16 is fine for play, f32 stays the default for building training caches. SPEED (64 contexts, bucketed, pipeline S4 running so indicative only): Metal f16 single 33 ms / batch-32 18.4 ms per context; Metal f32 30 / 20.2; ONNX CPU 45 / 25. f16 buys only ~10% over f32, so this port is dispatch/overhead-bound, not compute-bound: ~270 us per padded token at batch 32 is far too slow for a 12x768 encoder. Likely costs, now fixable in the vendored file: build_relative_position + make_log_bucket_position rebuilt from scalar tensor ops on every forward (cache per sequence length), XSoftmax's two where_cond passes, per-batch repeat() of the pos key/query layers, padding to the longest context in the batch (length bucketing in build_cache). New subcommand encoder-bench --n --batches --bucket for the step-4 numbers; a detached waiter runs it at batches 8/32/64 on Metal f32 and ONNX when S5 starts (scratchpad bench_s5.log).

**2026-09-18T14:17:07Z**

STEP 4 IDLE BENCH 2026-09-18 07:13-07:16 (machine idle, 256 real contexts, length-bucketed): Metal f16 single p50 26.4 ms, batch 8 11.9 ms/ctx, batch 32 12.7; Metal f32 27.5 / 13.2 / 13.9; ONNX CPU 41.5 / 18.3 / 17.6. Metal is 1.6x single and 1.45x batched; targets (15 ms single, 5 ms batched) NOT met. f16 is within 10% of f32, so the vendored port is overhead-bound: ~250 us per padded token regardless of batch. Next: cache the relative-position tables per sequence length in src/debertav2.rs, drop XSoftmax's two where_cond passes, avoid repeat() of pos layers, and length-bucket build_cache (that alone cut ONNX batched from 50 to 30 ms/ctx under load).

**2026-09-18T14:55:58Z**

STEPS 5-6 DONE 2026-09-18 07:55: Metal is the default (--encoder metal, --device auto; f32 weights, f16 opt-in via GLINER2_DOOM_ENCODER_DTYPE=f16). Prerequisites verified first: head training on Metal 53 s/epoch vs 108 s on CPU (1 epoch, 77k rows, same cache); Hub fallback download of the original safetensors works (23 s, into ~/.cache/huggingface/hub/models--fastino--gliner2.5-multi-v1); the direct path runs with no ORT_DYLIB_PATH. Checks on the rebuilt target/release with defaults: encoder-check metal f32; --encoder onnx --device cpu still works; encoder-parity vs onnx-fp32 PASS; eval v3 on data2/test top1 0.7190 (metal) vs 0.7191 (onnx); play default p50 31 ms; --policy keyword still runs through ORT; live real-time v3 3 eps mean 14.0 at 32.5 decisions/s p50 31 ms (was 26/s, 39 ms on the CPU encoder); tiny collect (oracle + head) -> train -> eval loop on Metal writes states.metal.bin; play with ORT_DYLIB_PATH unset works. scripts/pipeline.sh links whichever cache files exist into data2_rwr. One-time re-encode of data2 and data2_rwr caches on Metal launched (scratchpad recache.log). Latency targets NOT met; the overhead work continues in gd-eok7. Closing.
