---
id: gd-efsi
status: open
open: true
deps: []
links: []
created: 2026-09-18T20:02:00Z
type: task
priority: 2
assignee: memgrafter
tags: [gliner2, doom, jev, planning]
---
# Finding the boundaries of Jev Doom Demo

Verbatim transcript of TypeSafe's Jev Doom demo video (supplied by the maintainer 2026-09-18), kept here to plan against.

00:02 — Jev is so fast it can play "Doom." Every decision, where to move, where to aim, whether to hold down the trigger, Jev

00:11 — is making those decisions inside the software loop, responding at about 100 milliseconds per battery of questions. It's able to make 10 decisions a second, which is fast enough to track enemies and

00:24 — fire effectively, dodge projectiles. Our System 1 model is fed a structured description of the situation, health, nearby enemies, incoming projectiles, available pickups, everything a player needs to make

00:38 — informed decisions about what to do next in the game. With that game state as input, the model is asked to make small typed judgments,

00:46 — such as should the player's trigger be held down right now? What is the player's highest priority goal right now? Should the player be dodging?

00:52 — And others. We compose their answers into the player's next action and keep repeating that loop as the game changes.

00:58 — This is what we call a composition of AI primitives. The model has a default game strategy, generally telling it how to play the game

01:10 — somewhat effectively. But of course, because that's fed in as text, we can change that at any time. Let's try here, do not fire, simply dodge.

01:18 — We can see that immediately the player's behavior changes because the model is making the same judgments, but now in the context of that new instruction.

01:28 — Not the best strategy, but hey, that's on us. Now, this composition handles the moment-to-moment play. Add a second composition that decides where to explore next, and they can work

01:40 — together to make it through a level.

## Notes

**2026-09-18T20:10:33Z**

CRITICAL READ of typesafe.ai/blog/introducing-system-one-models-and-jev (2026-09-18). CLAIMED: a new model class (System One), a model Jev: 'unstructured state in, typed probabilistic decisions out'; three question types (choice / score / noul = is-this-true) mixed per call, answered in parallel and in isolation against one state, 'adding questions barely changes the response time'; 70-500 ms end to end; 'similar levels of intelligence on System One tasks' to frontier LLMs at 40-200x the speed; calibrated confidence; 'can't hallucinate' (0% type errors); training method named RLCD; $0.042/MTok input, output free. SUPPORTED: the API contract (three primitives, parallel independent questions) is concrete and matches what we mirrored in src/systemone.rs; the type-safety claim is definitional (outputs are drawn from a schema, so a malformed output is impossible; that says nothing about being right); the Doom demo is real-time on structured state, 10 queries/s, ~$7/hour. NOT SUPPORTED IN THE POST: no model size, base, or context length; no named benchmark, task list, sample size, or accuracy number for the 'frontier intelligence' claim (the only method described uses the average of two frontier LLMs' predictions as the reference, so 'intelligence' means agreement with those LLMs, and the authors admit bias); calibration asserted with no measurement; the 193.6x / 444.6x workflow numbers are on the authors' own test data ('some bias could exist'); no Doom score, kills, survival, or map; no failure modes or out-of-distribution statement anywhere. WHAT IT MEANS FOR US: the contract is the reusable idea, not the model. 'Questions in isolation, compose in code' is the keyword policy's shape (typed answers + hand-written composition) and is what makes their demo steerable by text. Their decision quality is unevidenced; ours is measured but weak. Our speed (12 ms, 35 decisions/s local) beats their 100 ms per battery. The honest comparison to run: our keyword policy already implements their exact contract locally over GLiNER2 (serve / triage subcommands); putting a Jev-style typed battery (hold trigger? dodge? top goal?) behind it on our state line, with an instruction field, is a one-day experiment that would show whether typed-question composition or a trained head plays better on the same maps.
