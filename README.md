# gliner2-doom

A frozen text model plays Doom. The model is the encoder inside
[GLiNER2.5](https://huggingface.co/fastino/gliner2.5-multi-v1), a 287M-parameter document-extraction
model. It never sees pixels, nothing is generated, and it was never trained on games. A small head
trained here reads the model's output and presses a button, 35 times a second, on a base Mac mini M4 with 16 GB.

## How one decision works

```
ViZDoom engine ──▶ one line of JSON ──▶ frozen encoder ──▶ trained head ──▶ button
   (Python glue)      (state.rs)         (Metal, 12 ms)    (0.8M params)    (options.rs)
```

1. **The engine reports what it knows.** Health, ammo, wall distances, and every visible object with
   its class, distance and angle. A 170-line Python file passes that to the Rust binary each tic.
2. **Rust writes one line.** Same keys, same order, every time. About 45 tokens:
   `{"hp":104,"ammo":39,"moved":24,"depth":{"l":17,"c":10,"r":9},"enemies":[{"t":"shotgun guy","d":449,"a":1}],"items":[{"t":"clip","d":79,"a":4}]}`
   This line is the model's entire input. A fact not in it cannot be used.
3. **The frozen encoder reads it.** Only GLiNER2.5's encoder is loaded, run on the GPU through a
   vendored Rust port of DeBERTa. Its own extraction heads are never used. Out come 768 numbers per token.
4. **The trained head picks a move.** Ten options: attack, turn, turn a little, step, strafe, use.
   The head scores each against the encoder's output and the winner is pressed for a few tics.

## What is learned and what is written by hand

Learned: everything between the line of text and the button, in the head.

Written by hand: the state line (what the model may see), the option table (what it may press), and
a scripted player called the oracle. The oracle exists only to make training data: it plays, its
choices are recorded as labels, and the head learns to agree with it (`collect`, then `train`).
A second round lets the head play while the oracle grades it. A third round (`train --rwr`) drops the
oracle and weights the head's own moves by what the game did next.

## Where it stands

- **Default head: v3**, trained on one arena and one map. Real time on defend_the_center: about 20
  kills per round, deciding once per engine tic.
- **v5 and v6** were trained on twelve maps and tested on five maps they had never seen. v6, the
  teacher-free one, collects items as well as the oracle and kills more, but both learned to walk into
  fights and lose the arena test. The training rows favour maps over the arena five to one; fixing
  that is the next round (ticket gd-u0r3).
- **The encoder swap matters.** Through stock mDeBERTa-v3-base weights instead of GLiNER2.5's, every
  head drops to chance. GLiNER2.5's fine-tuning moved the representation the heads rely on (gd-kmv0).
- **Speed.** Moving the encoder to the GPU and removing its per-call setup work cut a decision from
  about 30 ms to 12 ms. Play is now bound by the engine's 35 tics per second.

Tickets live in `.tickets/` (`tk ls`). `docs/HANDOFF.md` points at the handoff ticket.

## Run

```sh
just setup                       # ONNX Runtime into vendor/, the ViZDoom venv, cargo build --release
just live --window               # real-time, game window plus the terminal panel
just live --wad $W/freedoom1.wad --map e2m1 --head heads/v6.safetensors
just play --episodes 5           # synchronous: the engine waits for each decision
just collect --episodes 40 --epsilon 0.2 --out-dir data      # oracle rollouts -> rows
just train --data-dir data --out heads/new.safetensors --epochs 25
just eval --head heads/new.safetensors --data data/test.jsonl
```

`W` is the vizdoom package directory inside the venv. Weights download into the Hugging Face cache on
first run. `just live --policy keyword` runs the first version instead: the full GLiNER2.5 classifier
answering three multiple-choice questions, bound to buttons by hand-written rules. `just serve` exposes
the same classifier as a System One style endpoint (`POST /v1/systemone`).

In the terminal panel: `q` quit, `p` pause, `space` one decision, `m` manual (`a`/`d` turn, `f` fire,
`w`/`s` move). The panel shows the exact line the model read, the option probabilities, and the rates.

## Layout

```
bridge/vizdoom_bridge.py   the one Python file: engine in, JSON out
src/state.rs               observation -> the state line; the oracle
src/encoder.rs             the frozen encoder, Metal (default) or ONNX backend
src/encoder_metal.rs       loads the encoder tensors from the GLiNER2.5 safetensors
src/debertav2.rs           DeBERTa-v2 vendored from candle-transformers, with fixes
src/head.rs                the option-attention head
src/direct.rs              collect / train / eval and the play-time brain
src/options.rs             option name -> buttons, hold tics
src/run_loop.rs, tui.rs    real-time loop and terminal panel
src/policy.rs, describe.rs, scorer.rs, systemone.rs, serve.rs   the keyword policy and the endpoint
heads/                     trained heads (v1-v6, ablation) with json metadata
scripts/pipeline.sh        the resumable training pipeline used for v4-v6
```

Rust throughout, no system packages, one Python file for the engine. Developed and measured on a base Mac mini M4, 16 GB; macOS arm64 only for now.

## Credits

Model: Fastino (GLiNER2.5, arXiv:2507.18546). ONNX export and Rust engine: Jugaad s.r.l. / Dario
Finardi. DeBERTa-v2 in Rust: the candle-transformers authors, vendored under MIT/Apache-2.0.
Engine: ViZDoom (Farama). The world-statement idea and the real-time loop come from the openjev
reproductions; TypeSafe's Jev defined the contract the keyword policy mirrors.
