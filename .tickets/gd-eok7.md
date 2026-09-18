---
id: gd-eok7
status: open
open: true
deps: []
links: [gd-ngu7]
created: 2026-09-18T14:55:58Z
type: feature
priority: 2
assignee: memgrafter
tags: [gliner2, doom, rust, metal, performance]
---
# gliner2-doom Metal encoder: cut per-forward overhead in the vendored DeBERTa-v2 (target 5 ms/context batched, 15 ms single)

The Metal path (gd-ngu7) is the default since 2026-09-18 but the port is overhead-bound: ~250 us per padded token at every batch size, f16 within 10% of f32. Idle numbers (256 real contexts, bucketed): Metal f32 single p50 27.5 ms, batch 8 13.2 ms/ctx, batch 32 13.9; ONNX CPU 41.5 / 18.3 / 17.6. Targets from gd-ngu7: <= 5 ms/context batched, <= 15 ms single. Work, all in src/debertav2.rs and src/direct.rs: (1) cache build_relative_position + make_log_bucket_position per (query,key) length instead of rebuilding them from scalar tensor ops on every forward; (2) replace XSoftmax's two where_cond passes with one additive mask; (3) avoid repeat() of the pos key/query layers per batch (broadcast instead); (4) length-bucket the contexts in build_cache (sort by token count before chunking; cut ONNX batched from 50 to 30 ms/ctx under load); (5) profile what remains with the Metal capture tool before touching kernels. Pass condition per change: encoder-parity --reference onnx-fp32 still PASS, encoder-bench improves. Do not change numerics silently: f32 stays the default dtype.

## Notes

**2026-09-18T15:50:23Z**

ALL FOUR DONE 2026-09-18 08:50 (src/debertav2.rs + build_cache in src/direct.rs). (1) RelIdx: c2p/p2c gather indices built on the CPU per (batch,q,k) with a scalar twin of make_log_bucket_position, cached in the encoder (bounded to 8 shapes); (2) one additive mask per forward (-1e9 f32 / -1e4 f16, finite so padded rows stay finite) replaces XSoftmax's two select passes per layer; (3) position key/query projections computed once per layer, kept as [1,H,d,2span] and broadcast_matmul'd instead of repeat(); the layer-normed rel_embeddings cached too; (4) build_cache sorts contexts by byte length before chunking. Parity vs onnx-fp32 unchanged: batch 32 n=500 max 2.7e-4 mean 1.1e-6 PASS; batch 1 max 9.0e-5 PASS; f16 opt-in still runs (mean 2.1e-3 as before). Head scores unchanged (v3 on data2/test 0.7190). IDLE BENCH f32 bucketed, 256 real contexts: single p50 11.6 ms (was 27.5; target 15 MET); batch 8 10.7 ms/ctx (was 13.2); batch 32 12.4 (was 13.9); batch 64 15.5 (was ~14). Unbucketed batch 32 24.2, so (4) matters for cache builds. f16 batch 32 11.4. REAL-TIME defend_the_center v3, 5 eps: mean 19.8 (was 14.0) at 34.6 decisions/s = one per engine tic, p50 15 ms (was 26/s, 39 ms on the CPU encoder; 32.5/s, 31 ms before these changes). Play is now engine-bound. REMAINING GAP: batched target 5 ms/ctx not met. With the overhead gone the batched path is ~245 us per padded token, ~0.8 TFLOPS effective, i.e. compute-bound on candle's f32 Metal matmuls (the c2p/p2c terms multiply every token against all 512 relative positions per head per layer, as in HF). Levers left: f16 matmuls with f32 accumulation for the position terms only (measured ~30% on batched), or better kernels; not overhead. Batch-32 p95 is ~2.3x p50 because bucketed calls change shape every call (allocator churn), a possible small win.
