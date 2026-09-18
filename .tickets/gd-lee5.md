---
id: gd-lee5
status: open
open: true
deps: []
links: [pro-zekx, gd-ngu7]
created: 2026-09-18T11:37:18Z
type: task
priority: 1
assignee: memgrafter
tags: [gliner2, doom, handoff]
---
# HANDOFF: gliner2-doom state at 2026-09-18 04:37, running pipeline, pass/fail checks, commands, gotchas, next steps

Handoff written 2026-09-18 04:37 local for the next session. Read this, then `tk show pro-zekx` and `tk show gd-ngu7` (run tk from this directory).

## What this project is

GLiNER2.5 plays Doom. Rust binary `gliner2-doom` (this dir), ViZDoom engine through one Python glue file (`bridge/vizdoom_bridge.py`, venv `~/virtualenvs/gliner2-doom-vizdoom`), GLiNER2.5-multi encoder via ONNX Runtime (vendored dylib in `vendor/onnxruntime`, weights in the HF cache), trained option-attention head run and trained with candle (Hugging Face's Rust ML library). Rules: no system deps, new code in Rust, maintained libs only, no agnt.gg. Design direction: input -> output through a learned model; no describer/label/rule layers (see pro-yjme). README.md and AGENTS.md are current except for the pro-zekx results.

## Where things stand

- pro-2tst (closed): keyword policy over the zero-shot classifier; still available as `--policy keyword`.
- pro-yjme (closed): direct policy, `--policy direct` is the default with `heads/v3.safetensors`. defend_the_center real-time mean reward 14.4, sync 18-20; ablation proved the head reads the state.
- pro-zekx (OPEN, running): multi-map training with held-out maps, corpse/pickup fixes, item-seeking oracle, reward-weighted v6. Pipeline = `scripts/pipeline.sh` (resumable per stage; log `scripts/pipeline.log`). At handoff it is in S2 (v4 training, pid 17283, started 03:58, ~25 epochs x 60-80 s after a 16-min cache; expect v4 ~04:50, v5 ~05:30, v6 ~06:00, held-out eval ~06:30). The job was launched from the previous Claude session's shell; if it died with the session, `pgrep -f "gliner2-doom (train|collect)"` shows nothing and `scripts/pipeline.log` stops before "ALL DONE": then run `scripts/pipeline.sh` (it skips stages whose artifacts exist: data2/train.jsonl, heads/v4..v6, data2/.dagger_done, data2_rwr/train.jsonl). The original session log copy is `scripts/pipeline.log`; the live log of the still-running job is the scratchpad path in the previous session (same content, filtered), so prefer checking `heads/` mtimes and `pgrep`.
- gd-ngu7 (OPEN, not started): Metal port (head first, then the encoder through the DeBERTa-v2 implementation in candle-transformers, that Rust ML library's model collection). Do not run its parity/latency jobs while pro-zekx encoder-heavy stages run.

## When the pipeline finishes (pro-zekx)

1. Read `scripts/pipeline.log` S7 (held-out maps: oracle | v3 | v5 | v6 kills/items/ammo_left/hp per episode) and S8 (defend real-time for v5, v6).
2. Pass/fail checks (ticket body): items >= oracle-20% on held-out maps; path >= 10k (S7 uses sync play, path not printed: run `live --no-tui --wad ... --map ...` for path if needed); defend real-time mean >= 12; shots at corpses ~0 (check `ammo_left` vs kills).
3. Flip the default head in `src/main.rs` (two `default_value = "heads/v3.safetensors"`) to the best passing head (expect v5 or v6), rebuild, update README "Numbers" and AGENTS "Status", add a results note to pro-zekx, close it.
4. If v6 (RWR) is worse than v5: keep v5, note it; RWR is the first oracle-free stage and beta=5 / reward weights are untested guesses.

## Commands

```
cd ~/code/gliner2-doom
export ORT_DYLIB_PATH=$PWD/vendor/onnxruntime/lib/libonnxruntime.dylib
cargo build --release
./target/release/gliner2-doom live            # real-time, TUI (needs a TTY); --window opens the game window; --no-tui for headless
./target/release/gliner2-doom play --episodes 5 [--wad $W/freedoom2.wad --map map09]
./target/release/gliner2-doom collect|train|eval   # see README "Direct policy"
W=~/virtualenvs/gliner2-doom-vizdoom/lib/python3.12/site-packages/vizdoom
```
Claude's shell is not a TTY: use `--no-tui` or `--window`. ViZDoom drops `_vizdoom.ini` in cwd (gitignored).

## Gotchas learned (all also in AGENTS.md)

- gliner25-rs: pin ExecutionMode::Standard (Auto picks a bound path that breaks the classifier); CPU beats CoreML.
- Episode-end frame from the bridge carries the last known counters (fixed 2026-09-18); before that, end-of-episode kills/items read as 0.
- Corpses keep their class name: filtered as monster labels wider than tall. Pickups whitelisted. DeadMarine etc excluded.
- Split rule: of every 8 episodes, 7th -> val, 8th -> test (map collections use 8 eps).
- Rebuilding while a pipeline runs is OK on macOS (each stage launches the binary fresh), but CPU contention slows encoder stages; ORT uses 8 threads.
- Disk: data*/states.bin caches are big (3.6 GB for data2) and regenerable. 31 GB free at handoff after removing Colima (at the maintainer's request; Docker is gone from this machine).

## Not committed

Nothing in gliner2-doom/ has ever been committed; `.tickets/` lives inside it. The maintainer decides when to commit.

## Next after pro-zekx and gd-ngu7 (ideas, not tickets)

Batched rollouts (8 engines lockstep, ~2.3x on head-driven stages); smaller encoder (gliner2.5-small, ~4x, quality unknown); general multi-task head (Jev shape); flatmachines YAML machine calling /v1/systemone; upstream issue on gliner25-rs.

## Notes

**2026-09-18T14:55:58Z**

UPDATE 2026-09-18 08:00: pro-zekx pipeline finished (results in its notes; no head passed all gates; default head stays v3). gd-ngu7 closed: Metal is the default encoder and head device (commits 5b21ec7, 3ceb121, c9a7538 and the default flip). Follow-up gd-eok7 for the encoder overhead. Old ONNX caches data*/states.bin (5-6 GB each) are only needed with --encoder onnx and can be deleted once states.metal.bin exists.
