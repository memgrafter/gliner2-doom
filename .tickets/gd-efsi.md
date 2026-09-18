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

Verbatim transcript of TypeSafe's Jev Doom demo video (supplied by Trent 2026-09-18), kept here to plan against.

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
