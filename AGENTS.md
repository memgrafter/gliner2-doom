# gliner2-doom, agent notes

Work is tracked in `.tickets/` with `tk` from this directory. Read the open tickets first; `docs/HANDOFF.md` names the handoff one.

## Rules

- No system packages. Only cargo, uv, the Xcode command line tools, crates, and pip wheels inside a venv.
- New code is Rust. The one Python file is engine glue.
- Maintained libraries only, versions pinned.
- Experiments go in `lab/` and are deleted before the ticket closes.
- Tickets are never written by hand.

## Environment

- Python venvs live in `~/virtualenvs/gliner2-doom-<purpose>`, made with `uv venv --python 3.12`. Never a `.venv` here.
- The engine venv is `gliner2-doom-vizdoom` (`vizdoom==1.3.0`). `GLINER2_DOOM_PYTHON` overrides the interpreter.
- Weights download to the Hugging Face cache: fastino safetensors, plus the ONNX export for the keyword policy.
- `models/` (gitignored) overrides the cache; `GLINER2_DOOM_ENCODER_DIR` points at another encoder.
- ONNX Runtime is fetched into `vendor/` by `just fetch-ort`; only the keyword policy and `--encoder onnx` need it.
- References in `~/clones`: open-jev (no license, do not copy), openjev, gliner25-rs.
- Base Mac mini M4, 16 GB. Build release only.

## Measured, not obvious from the code

- The model reads keyword fields reliably and confuses left and right in prose a third of the time.
- Asked numeric questions, it answers "low" for ammo 26 and health 100. Compute thresholds in Rust.
- Prepending instructions to the text breaks its answers. Labels carry the meaning.
- One softmax over many unrelated statements comes out flat; use several small label sets.
- Each label rides in the prompt, so latency grows with label count. Keep play-time schemas under eight.
- CoreML is slower than CPU for the ONNX path.
- Parity is judged against the ONNX fp32 export only; the fp16 export and the f16 caches are noisy.
- f16 encoder weights save 10% and are six times noisier. Training caches stay f32.
- Stock mDeBERTa weights drop every trained head to chance.
- Offline top-1 means agreement with the oracle. Closed-loop play is the judge.
- Held-out maps need ten episodes per head to separate heads; two cannot.
- Do not run encoder-heavy jobs while a pipeline is on a training stage; real-time scores drop under load.
