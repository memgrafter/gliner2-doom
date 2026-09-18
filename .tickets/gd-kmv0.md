---
id: gd-kmv0
status: closed
open: false
deps: []
links: []
created: 2026-09-18T22:08:31Z
type: task
priority: 2
assignee: memgrafter
tags: [gliner2, doom, metal, ablation]
---
# Eval the trained heads through stock mDeBERTa-v3-base weights, no retraining

Swap the encoder weights for microsoft/mdeberta-v3-base (same architecture) and run heads v3-v6 on data2/test.jsonl plus v3 on defend_the_center real-time, through the stock encoder. No retraining. Measures how far the stock representation is from Fastino's fine-tuned one, since the heads were fit to Fastino's; a collapse says the fine-tune moved the representation, not that stock reads worse. Needs: env overrides for encoder dir and tensor prefix (stock uses deberta.*), the stock tokenizer as tokenizer.json, and the vocab-size difference handled.

## Notes

**2026-09-18T22:14:32Z**

DONE 2026-09-18 15:14, no retraining. Stock microsoft/mdeberta-v3-base (pytorch_model.bin, fp16, converted to safetensors with a torch-free unpickler, deberta.* renamed to encoder.*; the tokenizer is identical to GLiNER2's: same 250,101-piece vocabulary, GLiNER2 only pads the embedding to 250,112 vs stock 251,000). Loader gained GLINER2_DOOM_ENCODER_DIR. Heads on data2/test.jsonl through the stock encoder, top-1: v3 0.132 (Fastino 0.719), v4 0.166 (0.940), v5 0.108 (0.948), v6 0.090 (0.936); shuffled-context control 0.10-0.15, majority label 0.36: every head is at or below chance. v3 real-time defend_the_center: mean 1.0 (Fastino 19.8), i.e. random. READING: the heads are fit to Fastino's representation and the fine-tune moved it far enough that stock states are unrelated along the directions the heads read. This does NOT say stock reads worse; that needs a head retrained on stock states (same recipe, ~25 epochs) and is the ablation that would settle whether the GLiNER2 fine-tune matters. Not run, by decision.
