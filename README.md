# gliner2-doom

[GLiNER2.5](https://huggingface.co/fastino/gliner2.5-multi-v1) (a 287M-parameter zero-shot
classifier, Apache-2.0) plays Doom. Every decision is one encoder pass over a one-line
description of the screen; nothing is generated. The same binary answers "System One" style
business questions (routing, scoring, yes/no) over the same model.

Rust for everything new. The Doom engine is [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)
(pip wheel, bundles Freedoom), driven through a 100-line Python glue file. The direct policy runs
GLiNER2.5's encoder on the Apple GPU (a vendored DeBERTa-v2 port on candle/Metal, `src/debertav2.rs`);
the keyword policy still runs the full model through [gliner25-rs](https://github.com/dariofinardi/gliner25-rs)
on ONNX Runtime. No system packages.

```
 ViZDoom (python glue) ──JSON lines──▶ describe ──▶ "Nearest enemy: imp. Side: RIGHT. Offset: SMALL. ..."
        ▲                                                       │
        │                                     gliner25-rs ◀─────┘   side:   LEFT  RIGHT  CENTER  NONE
        │                                     (ONNX, CPU)           offset: SMALL LARGE  NONE
        └──── buttons ◀── policy ◀── probabilities ◀────┘           ahead:  OPEN  WALL
```

## In plain terms

**What it is.** A small AI model is playing Doom. Not a model built for games: a text model, the
kind normally used to pull names and dates out of documents. It is frozen. A tiny extra piece,
trained here, turns its reading of the situation into a button press.

**One decision.** Thirty-five times a second the game reports what it knows: health, ammo, how far
the walls are, where the nearest enemies and pickups are. That is written as one short line of
text. The frozen model reads the line. The trained head (under a million numbers) looks at that
reading and picks one of ten moves: attack, turn, step, strafe, use. The button is pressed.
The model never sees pixels, and there are no hand-written rules at play time.

**How it learned, with a teacher.** A simple scripted player, the oracle, played first and every
situation was recorded together with what it did. The head learned to copy it. Then the head
played on its own while the oracle kept grading each situation, and the head learned from its own
mistakes (DAgger). That is `collect` followed by `train`.

**How it learned, without a teacher.** In a further round the head played with a little random
exploration, and each of its own moves was weighted by what the game did next: kills and pickups
count for, damage taken and ammo spent count against. No oracle in the loop; it is only the
bootstrap. That is `train --rwr`. Both modes share the same recorder, rows, cache and head; only
the training signal differs.

**The state line, and where it comes from.** The model never sees pixels. ViZDoom exposes the
engine's own knowledge each frame: health and ammo, kill and item counts, position, a depth buffer,
and a list of every visible object with its class. The Python bridge writes those facts out once per
step; `state.rs` turns them into one canonical JSON line, same keys in the same order every time:

```
{"hp":104,"ammo":39,"moved":24,"depth":{"l":17,"c":10,"r":9},"enemies":[{"t":"shotgun guy","d":449,"a":1}],"items":[{"t":"clip","d":79,"a":4}]}
```

Health, ammo, distance moved since the last decision, wall distance left / centre / right, the nearest
live enemies and pickups with distance and angle off the crosshair. About 45 tokens. That line is
the model's entire input; a fact not in it cannot be used. Compact fields, not prose: the model reads
keyed fields reliably and confused left and right in sentences a third of the time.

**Who plays what role.** GLiNER2 is the reader, not the labeler: only its frozen encoder is loaded,
and its own extraction heads are never run. The oracle is the labeler in the ordinary supervised
sense: every training row is a state line plus the index of the option the oracle would pick for it,
and the head is trained to agree. In the DAgger round the head presses the buttons and the oracle
only annotates; offline accuracy therefore means agreement with the oracle, which is why closed-loop
play is the judge. A head trained this way can be more robust than the oracle but cannot learn a
strategy the oracle never expresses; the teacher-free round exists to break that ceiling. In one
sentence: a scripted bot writes the answer key, the frozen text model reads the question, and a tiny
head learns to fill in the answer.

**What we found.** On maps it had never seen, the teacher-free head collected items as well as the
scripted player and killed more. It also picked up a habit: walking forward into fights, because
most of its training came from exploring large maps rather than holding ground. So the steadier
earlier head stays the default. The teacher-free stage has run once, with untuned reward weights.

**Speed.** The frozen text model was the slow part, on the CPU. It now runs on the Mac's GPU,
checked to produce the same numbers to within rounding, and with its repeated setup work removed.
A decision went from about 30 ms to 12 ms. The bot decides once per game frame, and its score on
the standard arena went from 14 kills to 20 with no change to the brain.

**What is still hand-written.** What the model is allowed to see (the state line), what it is
allowed to press (the option table), and the oracle that bootstraps it. Everything between the
line of text and the button is learned.

## Setup (macOS arm64; ~2.7 GB: original weights 1.15 GB for the Metal encoder, ONNX weights 580 MB, ONNX Runtime 120 MB, venv 175 MB, build 500 MB)

```sh
just setup            # fetch ONNX Runtime into vendor/, create ~/virtualenvs/gliner2-doom-vizdoom, cargo build --release
```

Needs cargo, uv and the Xcode command line tools. Weights are fetched into the Hugging Face cache on
first run: `fastino/gliner2.5-multi-v1` (safetensors, only the `encoder.*` tensors are loaded) for the
Metal encoder and the jugaadsrl fp16 ONNX export for the keyword policy. See `AGENTS.md` for the
venv layout and the gotchas.

## Run

```sh
just live                                  # real-time Doom in the terminal, defend_the_center
just live --scenario level                 # a real Freedoom map (map01), 8 buttons, explores and shoots
just play --episodes 5 --record run.jsonl  # synchronous: engine waits for each decision; JSONL per decision
just triage                                # the business twin on three support tickets
just serve                                 # POST /v1/systemone on :8000 (TypeSafe System One contract)
just bench                                 # latency / RSS gate
```

`just` is optional (`cargo install just`): every recipe is one line, e.g. `just live` is
`ORT_DYLIB_PATH=$PWD/vendor/onnxruntime/lib/libonnxruntime.dylib cargo run --release -- live`.

## Interface

`live` draws two things. `--window` opens ViZDoom's own game window. The terminal view (needs a real
terminal; `--no-tui` turns it off) shows the Doom frame on the left as colored half-block cells,
`--cols` wide, and a panel on the right:

- title: encoder backend, scenario or map, policy
- episode, tic, reward, kills, health, ammo
- engine tics/s, decisions/s, decision latency p50, `[PAUSED]` / `[MANUAL]`
- `model read:` the JSON context the encoder saw for the latest decision (the model's entire input)
- `action:` the chosen option, the buttons it presses, the decision time
- one bar per option: the head's probabilities; the argmax is pressed

Keys: `q` quit, `p` pause, `space` one decision while paused, `m` manual (`a`/`d` turn, `f` fire,
`w`/`s` move). In manual the model keeps scoring and the bars keep moving, but your keys are pressed,
so you can steer into a situation and watch what it would do. On exit one summary line prints per-episode
rewards, kills, items, ammo spent and path length, then the rate line.


## Numbers (M4, 16 GB; 2026-09-17 on the CPU encoder, 2026-09-18 on Metal)

Metal encoder (default since 2026-09-18, `--encoder metal --device auto`; `--encoder onnx --device cpu` is the old path):

| | |
|---|---|
| parity vs the ONNX fp32 export, 500 contexts | max abs diff 5.4e-4, mean 1.1e-6 (f32); heads score identically through either encoder |
| encoder latency, idle, real contexts, length-bucketed | single p50 11.6 ms (CPU 41.5); batch 8: 10.7 ms/context (CPU 18.3); batch 32: 12.4 (CPU 17.6). Before the gd-eok7 overhead work: 27.5 / 13.2 / 13.9 |
| f16 weights (`GLINER2_DOOM_ENCODER_DTYPE=f16`) | within 10% of f32 speed, ~6x noisier vs fp32: f32 stays the default |
| defend_the_center, real-time, v3 | mean reward 19.8 (24 / 18 / 17 / 18 / 22) at 34.6 tics/s, 34.6 decisions/s = one per engine tic, p50 15 ms (CPU encoder: 14.0, 26/s, 39 ms) |
| head training, 77k rows | 53 s/epoch on Metal vs 108 s on CPU; one-time state cache ~12 ms/context (length-sorted batches) |
| memory | encoder 1.1 GB f32 in unified memory (0.56 GB f16); peak RSS 1.9 GB vs 1.4 GB on ONNX |

Held-out maps (ticket pro-zekx, 2026-09-18; sync play, 2 episodes each on freedoom2 map09/map10/map12 and
freedoom1 e2m1/e2m2, never seen in training): mean kills / items / survived 60 s: oracle 2.9 / 4.7 / 60%,
v3 3.2 / 1.2 / 60%, v5 (DAgger on 12 maps) 2.6 / 1.7 / 60%, v6 (reward-weighted) 3.5 / 4.7 / 80%. v6 matches the
oracle on items and leads on kills and survival but scores 6.8 on defend_the_center real-time (v5 8.8, v3 14.0):
both learned `move forward` from the maps, which outweigh the scenario 5:1 in the rows. The default head stays v3.

Direct policy (default, `heads/v3.safetensors`; no rules at runtime; CPU encoder numbers from 2026-09-17):

| | |
|---|---|
| training data | 45,350 rows: oracle + epsilon-random rollouts, then two DAgger rounds (head drives, oracle labels) |
| offline, test split (3,977 rows) | top-1 95.6% vs oracle, ECE 0.014, shuffled-context control 28.5% (majority label 40.6%) |
| ablation: contexts permuted at training | top-1 43.6% = majority baseline; closed loop dies with 0 kills every episode |
| defend_the_center, sync | mean reward 18.0 (v2 20.0); oracle 15-17, keyword policy 17, random 1 |
| defend_the_center, real-time | mean reward 14.4 (17 / 13 / 16 / 14 / 12) at 34.6 tics/s, 30 decisions/s |
| decision latency | p50 33 ms (encoder 41 tokens, head ~0.1 ms) |
| map01 (Freedoom), real-time, 120 s | path length 14,865 map units, 0 kills |
| head | 0.8M params, 3.9 MB, 25 epochs x 39 s on CPU after a one-time 10 min state cache |

Keyword policy (`--policy keyword`, the zero-shot schema with hand-written binding):

| | |
|---|---|
| decision latency | p50 62-78 ms, p95 ~90 ms (8-9 labels in the prompt) |
| real-time play | 35 tics/s engine, 12-14 decisions/s |
| defend_the_center, sync | mean reward 17-20 over 3-5 episodes (scripted oracle 15-17, random 1) |
| defend_the_center, real-time | mean reward 11 (12 / 15 / 7); needs the stuck-recovery controller on maps |
| label accuracy on game state | 100% side, 100% offset over 859 decisions |
| RSS | ~1.0 GB right after load, 0.4-0.6 GB steady; ViZDoom ~200 MB |
| load | 2.4 s warm |

## Direct policy (no rules): state JSON in, action out

The keyword policy above still hand-writes what the model may notice and what its answers mean.
The direct policy removes that: the engine state is serialized to one JSON line, the frozen
GLiNER2.5 encoder turns it into token states, and a small option-attention head (jevlike's shape,
0.8M parameters, trained and run with candle) scores the action names directly. The argmax is
pressed. No describer, no label schema, no thresholds, no stuck-recovery controller.

```
{"hp":72,"ammo":18,"moved":0,"depth":{"l":37,"c":49,"r":19},"enemies":[{"t":"imp","d":210,"a":-14}],"items":[]}
      │ frozen encoder (Metal; ONNX with --encoder onnx)  options: attack | turn left | turn right | turn left a little | ... (encoded once)
      ▼                                                  │
   H [S×768] ──▶ option-attention head ◀────────────────┘ ──▶ softmax ──▶ press argmax (buttons + hold from options.json)
```

```sh
just collect --scenario defend_the_center --episodes 40 --epsilon 0.2   # oracle rollouts -> data/
just collect --scenario level --episodes 24 --epsilon 0.2 --timeout-tics 2100
just train --out heads/v1.safetensors --epochs 25                         # state cache first (~13 ms/context on Metal), then ~53 s/epoch on Metal for 77k rows
just eval --head heads/v1.safetensors --data data/test.jsonl              # top-1, NLL, ECE, shuffled-context control
just live                                                                 # default: direct policy with heads/v3.safetensors
just live --policy keyword                                                # the zero-shot keyword schema instead
just collect --episodes 20 --epsilon 0 --head heads/v1.safetensors        # DAgger: head drives, oracle labels; then train again
just train --shuffle-contexts --out heads/ablation.safetensors            # ablation: contexts permuted, only the prior is learnable
```

The scripted oracle exists only to label rows. What remains hand-written: the serializer (which
fields, in what units), `options.json` (name → buttons, hold tics), and the oracle. Results are in
the ticket `.tickets/pro-yjme.md` and summarised in the Numbers table.

## What the keyword policy decides

The screen is described from ViZDoom's labels and depth buffers as keyword fields. The model
classifies that line against small label sets, several questions in one pass. Rust computes the
numeric thresholds (health, ammo, wall distance), binds answers to buttons (LEFT → turn left,
CENTER → attack, NONE + OPEN → explore), pulses turns so a stale verdict cannot overshoot the
±3° hit window, and recovers when stuck. Natural-language descriptions were tried first and lost:
this model confuses LEFT and RIGHT in sentences 30-70% of the time and reads them at 97-100% as
`Side: RIGHT`. Details and every measurement are in the ticket (`.tickets/pro-2tst.md`).

## System One endpoint

```sh
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "I ordered size 10 shoes but received size 8. Please send the right size.",
  "questions": {
    "department":   {"type": "choice", "criteria": {"returns": "Returns and exchanges", "shipping": "Delivery issues", "billing": "Charges and refunds"}},
    "severity":     {"type": "score",  "criteria": ["minor", "moderate: wrong item", "major: safety or financial loss"]},
    "wants_refund": {"type": "noul",   "instructions": "Is the customer asking for a refund to their card?"}
  }}'
```

Each question is one classification pass over the bare state (about 65 ms). `choice` labels are
`name: description`; `score` is the probability-weighted level index; `noul` scores the statement
(`criteria.true`, else the instructions) against `no`. `confidence` is 1 minus normalised entropy.

## Layout

```
src/main.rs        CLI: play | live | collect | train | eval | encoder-check | encoder-parity | encoder-bench | bench | probe | smoke | triage | serve
src/bridge.rs      client for bridge/vizdoom_bridge.py (JSON lines over stdio)
src/describe.rs    observation -> keyword situation line
src/policy.rs      label schema, answer -> buttons, pulse holds, stuck recovery
src/scorer.rs      gliner25-rs wrapper (classification only, Standard execution mode)
src/run_loop.rs    real-time loop: engine on the main thread, scorer on a worker, newest observation wins
src/tui.rs         terminal frame + probability bars (crossterm)
src/systemone.rs   System One request/response over the scorer
src/options.rs     option table (name -> buttons, hold)
src/state.rs       state -> canonical JSON; scripted oracle (training labels only)
src/encoder.rs     frozen GLiNER2.5 encoder, one enum over two backends: text -> token states
src/encoder_metal.rs  the Metal backend: original safetensors (encoder.* prefix) through src/debertav2.rs
src/debertav2.rs   DeBERTa-v2 vendored from candle-transformers 0.11.0 with the f16 dtype fix
src/head.rs        option-attention head (candle, Metal or CPU): states + option vectors -> logits
src/direct.rs      collect / train / eval / DirectBrain
heads/             trained heads (safetensors + json meta)
data*/             rollouts (gitignored) and the encoder state caches (states.metal.bin; states.bin for --encoder onnx)
src/serve.rs       axum server
bridge/            the one Python file, plus nothing else
vendor/            ONNX Runtime (fetched, gitignored)
```

## Credits

Model: Fastino (GLiNER2.5, arXiv:2507.18546). ONNX export and Rust engine: Jugaad s.r.l. / Dario
Finardi. DeBERTa-v2 in Rust: the candle-transformers authors (Hugging Face), vendored under MIT/Apache-2.0. Engine: ViZDoom (Farama). The hypothesis-phrasing and real-time loop ideas come from the
openjev reproductions by mikesmullin and daseinlabs; TypeSafe's Jev defined the contract.
