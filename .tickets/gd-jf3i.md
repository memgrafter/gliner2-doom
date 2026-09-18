---
id: gd-jf3i
status: open
open: true
deps: []
links: [gd-u0r3, gd-eok7]
created: 2026-09-18T18:44:32Z
type: feature
priority: 2
assignee: memgrafter
tags: [gliner2, doom, rust, metal, training]
---
# gliner2-doom: fine-tune the GLiNER2 encoder end to end with the head (unfreeze, same rows)

Today the encoder is frozen and only the 0.8M head learns; the encoder reads, the head knows what buttons do. Unfreezing makes GLiNER2's own weights carry the game knowledge. Possible now because the encoder runs in our Rust code (src/debertav2.rs) on Metal, where gradients are available; it was impossible through ONNX. APPROACH: train --unfreeze: build the encoder from the safetensors with a trainable VarMap, forward live (the states cache cannot be used since its outputs change), backprop through head and encoder, AdamW with a low encoder learning rate (1e-5 order) and the head at 1e-3; start from v3 or v5 as the head warm start. First run: top 2-4 layers only; full unfreeze second. Same rows and oracle labels as pro-zekx; RWR later. COST: 278M params; f32 weights + AdamW state ~3.3 GB, fine on 16 GB. Each epoch runs the encoder forward and backward on ~77k contexts: at ~12 ms forward batched expect roughly 40-60 min per epoch on Metal, so 3-5 epochs per run. Save encoder deltas or the full encoder per run (1.1 GB). RISKS: forgetting (the pretrained reading is the whole premise; measure by re-running the frozen head on the fine-tuned encoder's states), candle autograd coverage for gather/where_cond/layer_norm in the vendored model (check with one backward step before anything else), memory at batch 32 with activations kept for backward (drop to 8-16 if needed). PASS: beats the frozen-encoder head on the same evaluation, 10 episodes per map per head on the held-out maps plus the defend real-time regression; keep v3 as the control. The frozen path stays the default until then.
