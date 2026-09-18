---
id: pro-yjme
status: closed
open: false
deps: [pro-2tst]
links: [pro-2tst, pro-zekx]
created: 2026-09-18T03:28:11Z
type: feature
priority: 1
assignee: memgrafter
tags: [gliner2, doom, rust, jev, candle, direct-policy]
---
# gliner2-doom direct policy: state JSON -> frozen GLiNER2.5 encoder -> trained option-attention head (Rust) -> action; no describer, no label binding

Here is the concrete proposal. Everything hand-written today (describer, labels, thresholds, action binding) is replaced by one trained scoring head; the only remaining hand-written pieces are the state serializer, the option table, and the oracle used to produce training labels.

## Architecture

```
engine state ──serialize──▶ context JSON (~50 tokens)
                                   │
                    frozen GLiNER2.5 encoder (existing encoder_fp16.onnx, ort)  ──▶ H [S×768]
                                   │
options ["attack","turn left",…] ──▶ same encoder, once at startup, mean-pooled ──▶ o_j [768] (cached)
                                   │
                         option-attention head (new, ~0.8M params, Rust)
                         q_j = Wq·o_j   K = Wk·H   V = Wv·H
                         c_j = softmax(q_j·Kᵀ/√d)·V                 d = 256
                         s_j = MLP([c_j ; Wo·o_j ; c_j ⊙ Wo·o_j]) → scalar
                         p = softmax(s / T)                          T fit on held-out
                                   │
                         argmax option → its buttons + hold tics (from the option table)
```

This is jevlike's head on top of the encoder we already run. Encoder stays frozen and in ONNX, so nothing is re-exported. The head is trained and run in Rust with Hugging Face's candle ML library (0.11, June 2026, active), so no torch venv and no Python beyond the engine glue. Latency stays where it is: the encoder pass dominates, and the head is microseconds.

**Context serialization** is data, not judgment. Fixed key order, integers, nearest three enemies and items, angle signed with left positive, plus one memory field so the model can learn to unstick itself:

```json
{"hp":72,"ammo":18,"moved":0,"depth":{"l":37,"c":49,"r":19},
 "enemies":[{"t":"imp","d":210,"a":-14},{"t":"demon","d":640,"a":88}],
 "items":[{"t":"medikit","d":80,"a":-30}]}
```

**Option table** is data too, in `options.json`: name, buttons, hold tics. defend_the_center gets attack, turn left, turn right, turn left a little, turn right a little. Maps add move forward, move backward, strafe left, strafe right, use. The fine/coarse choice becomes the model's, not a rule.

## Pipeline

1. **Collect.** `gliner2-doom collect --episodes 40 --epsilon 0.2` runs the scripted oracle headless in sync mode with 20% random actions, so the data includes states the oracle never visits on its own. One row per decision: context, options, oracle label, reward, episode seed. Headless sync runs at ~300 tics/s with no model in the loop, so 20k rows take about a minute. Split by seed, 80/10/10.
2. **Train.** `gliner2-doom train data/ --out head.safetensors`. Hidden states are computed on the fly through the same ONNX encoder used at play time, so train and inference see identical inputs. Cross-entropy against the oracle action, label smoothing 0.05, AdamW at 1e-3, batch 64, 10 epochs. Minutes on the M4 CPU. Temperature T fit on the validation split for calibration.
3. **Evaluate offline.** `gliner2-doom eval head.safetensors data/test.jsonl` reports top-1 against the oracle, expected calibration error, and jevlike's shuffled-context control (each context paired with the wrong row's label set; a real model must beat it). Passes if top-1 ≥ 95%, ECE ≤ 0.05, and the shuffled control is near chance.
4. **Evaluate closed-loop.** `live --policy direct --head head.safetensors`, sync and real-time, 5 episodes each, against the three numbers we already have: oracle 15 to 17, keyword policy 17 sync / 11 real-time, random 1.
5. **One DAgger round.** Collect again with the trained head driving and the oracle labeling, retrain on the union. This is the step that fixes the states a cloned policy drifts into. mikesmullin and the theoriclabs issue both needed exactly one round.
6. **Ablation that answers your question.** Train once with the serialized state and once with the state's fields shuffled per row. If the closed-loop score survives shuffling, the head is imitating the option prior, not reading the state. That tells us whether the layer disappeared or moved.

## What stays hand-written, and what does not

| kept | removed |
|---|---|
| serializer (which fields, units) | describer sentences and keyword buckets |
| option table (name → buttons, hold) | side/offset/ahead label schema |
| oracle, as a label source only | CENTER 3°, SMALL 12°, WALL 12, MIN_CONF 0.45 |
| engine glue | policy binding and stuck-recovery controller |

Confidence is displayed, never used to block an action. The model's argmax is pressed every time, as in the Jev demos.

**Time:** about one day. Morning for serializer, option table, collect, train, offline eval; afternoon for the runtime flag, closed-loop numbers, and the DAgger round. Disk need is small since no torch venv is involved, roughly 300 MB for the candle crate in the build dir.

**Two honest limits.** First, this head is a specialist trained on one task's labels; Jev is one general model across tasks, and getting there means training across many option sets, which is a separate project. Second, the ceiling is the oracle until you switch the training signal from oracle labels to engine reward, which is the step after DAgger and is where it stops being imitation.

## Notes

**2026-09-18T03:37:07Z**

Progress 2026-09-18 ~04:10: modules options.rs/state.rs/encoder.rs/head.rs/direct.rs written and building; collect done in ~10 s: 24,487 rows (defend 40 eps + level 24 eps, eps=0.2), split by episode 19,875/2,311/2,301; oracle mean reward 14.6 on defend. Encoder direct path: 41 tokens/context, 34 ms single, 16 ms/text batched. v1 training running (10 epochs, AdamW 1e-3, batch 64, label smoothing 0.05, T fit on val). --policy direct wired into play and live (no controller in direct mode; moved tracked per tic).

**2026-09-18T03:51:51Z**

v1 RESULT (2026-09-17 20:47-20:51 local): head trained 10 epochs (22 s/epoch after an 8-min state cache; 3.9 MB safetensors). Val top1 91.3% (still rising), T=0.65, val ECE 0.016. Test: top1 91.7% (defend 92.0%, level 91.5%), NLL 0.258, ECE 0.016; shuffled-context control 29.0% vs majority-label 42.1% -> the head reads the state. Top confusion: turn left -> turn right (58/2301), i.e. the sign of the angle. CLOSED LOOP defend_the_center, no rules at all: sync 20/15/15/23/15 mean 17.6 at 26 decisions/s p50 32 ms (keyword policy 17.3, oracle 15-17, random 1); real-time 14/4/16/14/10 mean 11.6 at 34.5 tics/s and 29 decisions/s p50 34 ms (keyword 11.3). Direct path is 2x faster per decision than the keyword schema (no labels in the prompt). Added data/states.bin disk cache for encoder states. DAgger round running: v1 drives 20 defend + 12 level episodes, oracle labels, retrain v2 for 25 epochs.

**2026-09-18T04:29:38Z**

v2 RESULT (DAgger round 1, 20:51-21:28 local): v1 drove 20 defend eps (mean reward 18.1) + 12 level eps with oracle labels -> 36,461 rows total. Trained 25 epochs (31 s/epoch; 28,170 contexts encoded in 625 s, now cached in data/states.bin). Best epoch 22: val top1 93.1%, T=0.70, val ECE 0.008. Test: top1 92.3% (defend 91.2%, level 93.4%), NLL 0.224, ECE 0.007; shuffled-context control 30.4% vs majority 43.6%. Remaining confusions: angle sign (turn left->turn right 33, reverse 18), coarse vs fine turn (~50), attack vs small turn (~40): all boundary cases at +-3 and +-12 deg. CLOSED LOOP defend_the_center, direct policy, no rules: sync 18/20/19/21/22 mean 20.0 (v1 17.6, keyword 17.3, oracle 15-17) at 26 decisions/s p50 33 ms; real-time 14/6/13/8/12 mean 10.6 (v1 11.6, keyword 11.3) at 34.4 tics/s, 30 decisions/s. map01 real-time: path 1,873 units, 0 kills (keyword policy with controller: 15,396): exploration is the imitation gap; running a level-only DAgger round 2 -> v3. Ablation (contexts permuted) running.

**2026-09-18T05:07:13Z**

ABLATION (21:29-21:37): same 29,635 rows with contexts permuted across rows, 12 epochs: val top1 43.4%, test top1 43.6% = majority-label baseline 43.6%, shuffled control 43.6%. Closed loop defend_the_center: sync reward -1/-1/-1/-1/-1 (dies with 0 kills, episodes 240-344 tics), real-time -1/-1/-1. The head with real contexts scores 92-96% and 18-20 kills. The layer did not move into the weights as a prior; the head reads the state.

**2026-09-18T05:07:13Z**

v3 RESULT (DAgger round 2, level only: v2 drove 16 map01 eps, 8,889 rows; 45,350 rows total; 25 epochs at 39 s): best epoch 19, val top1 95.9%, T=0.60, val ECE 0.014. Test 3,977 rows: top1 95.6% (defend 95.6%, level 95.6%), NLL 0.144, ECE 0.014, shuffled control 28.5% vs majority 40.6%. CHECKS: top1 >= 95% MET, ECE <= 0.05 MET, shuffled near chance MET. CLOSED LOOP defend: sync mean 18.0; real-time 17/13/16/14/12 mean 14.4 at 34.6 tics/s, 29.6 decisions/s, p50 34 ms -> best real-time result of any policy (keyword 11.3, v1 11.6, v2 10.6). map01 real-time: path 14,865 units (v2 1,873; keyword+controller 15,396), 0 kills in 120 s (no policy has landed a kill on map01 inside the cap). --policy direct with heads/v3.safetensors is now the default for play and live; --policy keyword keeps the old path.

**2026-09-18T05:07:13Z**

DONE 2026-09-17 22:10 local, ready for review (left open for review). Timeline: 20:28 ticket; 20:45 modules; 20:50 collect (24k rows, 10 s); 20:35-20:47 v1 train; 20:51-21:28 DAgger 1 + v2; 21:29-21:37 ablation; 21:40-22:06 DAgger 2 + v3. Hand-written pieces remaining: state::serialize (fields, units), options table (name -> buttons, hold), state::oracle (labels only). Removed from the runtime: describe.rs keyword fields, label schema, thresholds, Controller. Follow-ups: reward-weighted or RL training signal (ceiling is the oracle), a general multi-task head (Jev shape across question sets), kills on real maps need longer episodes or a closer-engagement oracle.
