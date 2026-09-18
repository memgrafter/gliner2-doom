# gliner2-doom

A frozen text model plays Doom on a base Mac mini M4. The model is the encoder inside
[GLiNER2.5](https://huggingface.co/fastino/gliner2.5-multi-v1); a small head trained here turns its reading of the game into a move.

## One decision

```
ViZDoom ──▶ one line of JSON ──▶ frozen encoder ──▶ trained head ──▶ move
```

- The engine reports health, ammo, walls, enemies and pickups. No pixels.
- Rust writes that as one short line of JSON, the model's only input.
- The encoder runs on the GPU and turns the line into vectors.
- The head scores ten moves against those vectors; the winner's buttons are held for a few tics.

Thirty-five decisions a second, about 12 ms each.

## Hand-written vs learned

Hand-written: the state line, the ten moves, and a scripted player that makes training data.
Learned: the head, first by copying the scripted player, then from game outcomes.

## Where it stands

- Default head v3: about 20 kills a round on defend_the_center, real time.
- v6, trained without a teacher on twelve maps, collects and kills best on unseen maps.
- v5 and v6 walk into fights and lose the arena; the map-heavy training rows are why.
- Stock mDeBERTa weights drop every head to chance; GLiNER2.5's fine-tune matters.

Tickets: `.tickets/` (`tk ls`). Handoff: `docs/HANDOFF.md`.

## Run

```sh
just setup
just live --window
just play --episodes 5
just collect --episodes 40 --epsilon 0.2 --out-dir data
just train --data-dir data --out heads/new.safetensors --epochs 25
just eval --head heads/new.safetensors --data data/test.jsonl
```

Terminal panel keys: `q` quit, `p` pause, `space` step, `m` manual.

## Credits

Fastino (GLiNER2.5), Jugaad s.r.l. (ONNX export, gliner25-rs), Hugging Face (candle, DeBERTa port),
Farama (ViZDoom), the openjev reproductions, and TypeSafe's Jev for the contract.
