---
id: gd-eok7
status: open
open: true
deps: []
links: [gd-ngu7, gd-88vd, gd-jf3i]
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

ALL FOUR DONE 2026-09-18 08:50. (1) c2p/p2c gather indices built on the CPU per (batch,q,k), cached (8 shapes); (2) one additive mask per forward instead of XSoftmax's two select passes per layer; (3) position projections computed once per layer and broadcast, rel_embeddings cached; (4) build_cache sorts contexts by length. Parity vs onnx-fp32 unchanged (max 2.7e-4, mean 1.1e-6, PASS); head scores unchanged. IDLE BENCH f32, 256 contexts, bucketed: single p50 11.6 ms (was 27.5; target 15 met); batch 8 10.7 ms/ctx (was 13.2); batch 32 12.4 (was 13.9). Unbucketed batch 32 24.2. REAL-TIME defend v3: mean 19.8 (was 14.0), 34.6 decisions/s = engine-bound, p50 15 ms. REMAINING: batched target 5 ms not met; ~245 us per padded token, compute-bound on f32 matmuls (c2p/p2c multiply every token against 512 positions per head per layer). Full f16 measured: batch 32 11.4 (8% gain), single 12.3 (slower), ~6x noisier. f16 for the position terms only is untried. No measured path to 5 ms; profile with Metal capture before touching kernels.
