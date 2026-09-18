---
id: pro-zekx
status: closed
open: false
deps: []
links: [pro-yjme, gd-ngu7, gd-lee5, gd-u0r3]
created: 2026-09-18T10:47:57Z
type: feature
priority: 1
assignee: memgrafter
tags: [gliner2, doom, rust, direct-policy, holdout, maps]
---
# gliner2-doom direct policy v4/v5: more maps in training with held-out maps, corpse filtering, item collection and ammo discipline in the oracle

Observed on the map tour (freedoom2 map02/map03, freedoom1 e1m1, head v3 trained on defend_the_center + map01 only): the bot empties its ammo into dead enemies and walks past health/ammo pickups. The maintainer's ask: "we need to include more maps in the training ... but we also need holdback."

## Defects

1. **Corpses.** If ViZDoom's labels buffer keeps dead monsters, the serialized state lists them as enemies and the oracle labels "attack" whenever one is centered; the head imitates that. Verify empirically (kill, then watch the label persist). Fix in the bridge/serializer: drop corpses (by actor flag if exposed; else by tracking per-object kill events / bbox collapse), never in the head.
2. **Items.** The state already carries items (nearest 3 with distance/angle) but the oracle ignores them. Add an oracle rule: no live enemy in view and a pickup in view -> turn toward it, move forward when centered (health pickups when hp < 100, ammo pickups when ammo < cap, otherwise still take them). Priority: enemies > items > explore.
3. **Ammo discipline.** Oracle: attack only when a live enemy is centered and within range; count ammo spent per episode as a metric.

## Data and holdout

- Training maps: defend_the_center + freedoom2 map01-map08 + freedoom1 e1m1-e1m4 (12 maps).
- Held-out maps (never collected, never in DAgger): freedoom2 map09, map10, map12; freedoom1 e2m1, e2m2. Closed-loop only.
- `collect` gains `--wad/--map`; a `collect-maps` recipe loops the training list. Oracle rollouts 12 eps/map, then one DAgger round with the head driving on the training maps only (6 eps/map).
- Bridge exposes ITEMCOUNT and KILLCOUNT; live/play summaries report kills, items picked, ammo spent, path length.

## Pass/fail checks (closed-loop on held-out maps, 120 s episodes, 3 each)

- ammo spent on corpses ~0 (measured as shots fired with no live enemy centered)
- items picked per episode > oracle on the same maps minus 20%
- path length >= 10k units, no episode spinning in place
- defend_the_center regression: real-time mean reward >= 12

## Steps

1. Corpse investigation + bridge fix + serializer field.
2. Oracle: items + ammo discipline. Re-validate oracle closed-loop on map01 (items picked, kills) before any training.
3. collect --wad/--map, map lists, ITEMCOUNT; collect training maps.
4. Train v4 (oracle data), DAgger round on training maps -> v5.
5. Evaluate v5 on held-out maps vs oracle and vs v3; regression on defend_the_center.
6. Docs + numbers + this ticket closed.

## Notes

**2026-09-18T10:55:18Z**

Root cause of 'no kills on maps' (2026-09-18 ~00:20): the bridge's terminal frame carried no game variables, so every end-of-episode kills/items/ammo read as 0 through serde defaults. Fixed: bridge caches last_vars and returns them on the finished frame. True numbers on map01 per 60 s: oracle 2 kills / 4 items, v3 head 2 kills / 2 items / 6 rounds / 14.7k units; deadly_corridor oracle 5 kills. Corpses: ViZDoom removes scenario actors on death, but real-map corpses keep the class name; filtered as monster labels wider than tall (is_alive). Pickups whitelisted by class (Blood, DeadMarine etc. were counted as items). Oracle now seeks pickups when no live enemy is in view and fires only at live, centered enemies within 1500 units. Added: collect --wad/--map + epsilon with a driving head, play --wad/--map, ITEMCOUNT, per-step kills/items/health/ammo in rows, and train --rwr (reward-weighted regression on the head's own actions, warm start) so the final stage has no oracle labels. Pipeline v4 (oracle, 12 training maps + defend) -> v5 (DAgger) -> v6 (RWR) -> held-out maps map09/map10/map12/e2m1/e2m2 launched.

**2026-09-18T11:17:44Z**

STATUS 2026-09-18 04:17 local. WHAT WE ARE DOING: replacing the last hand-written behaviour around the model with training. The direct policy (pro-yjme) already removed the describer, label schema and rule binding: engine state JSON -> frozen GLiNER2.5 encoder -> 0.8M-param option-attention head (Rust, candle ML library) -> action. Its weaknesses seen on the map tour (firing at corpses, ignoring pickups) and its evaluation gap (held-out episodes, not held-out maps) are what this ticket closes. Three stages, one background pipeline (pipeline_v456.sh): (v4) oracle rollouts with 20% random actions on defend_the_center + 12 training maps (freedoom2 map01-map08, freedoom1 e1m1-e1m4), train from scratch, 25 epochs; (v5) DAgger: v4 drives the same maps, oracle labels, warm-start retrain 15 epochs; (v6) reward-weighted regression: v5 drives with 10% exploration, rows are labelled with the action actually taken and weighted by exp(advantage/5) of discounted engine reward-to-go (kills +10, items +3, health +0.05/pt, ammo -0.05/round), warm start from v5, 10 epochs, lr 3e-4. v6 has no oracle in its loop; the oracle is only the bootstrap. Then held-out maps never seen in any training data (freedoom2 map09, map10, map12; freedoom1 e2m1, e2m2), 2 sync episodes each, for oracle vs v3 vs v5 vs v6: kills, items, ammo left, hp, path; plus a real-time defend_the_center regression for v5 and v6. WHERE WE ARE: S1 done at 03:58 (65,913 rows: 48,954 train / 8,334 val / 8,625 test, split by episode 7th->val 8th->test of every 8; defend oracle mean reward 14.2). S2 running: v4 training (encoding ~57k new contexts into data2/states.bin, ~20 min, then 25 epochs at ~60-80 s). Expected: v4 ~04:50, v5 ~05:30, v6 ~06:00, held-out eval ~06:30, defend regression ~06:45. FIXED TONIGHT BEFORE LAUNCH: terminal-frame counters (kills/items/ammo were read as 0 at episode end: reporting bug, not behaviour), corpse filter (monster label wider than tall), pickup whitelist, oracle seeks pickups and fires only at live centered enemies within 1500 units, collect --wad/--map + epsilon while a head drives, play --wad/--map, ITEMCOUNT, per-step kills/items/health/ammo in rows, train --rwr --init-from, split rule. STILL HAND-WRITTEN: state serializer, option table, oracle (bootstrap labels), reward shaping weights for v6. PASS/FAIL CHECKS (from the ticket body): ~0 shots without a live centered enemy, items >= oracle-20% on held-out maps, path >= 10k units, defend real-time mean >= 12. DEFAULT HEAD: heads/v3.safetensors until v5/v6 pass; flip to the best passing head and update README numbers when the pipeline finishes. Artifacts: data2/ (regenerable), data2_rwr/, heads/v4-v6, pipeline log in the session scratchpad; nothing committed.

**2026-09-18T13:12:00Z**

STATUS 2026-09-18 06:12. v5 done 06:08: test top1 0.9483 nll 0.1403 ece 0.0061 (v4 was 0.9397 / 0.1547); defend rows 0.9425 (v4 0.9200). S5 RWR rollouts started 06:09 (v5 drives, eps 0.1). The running job is the previous session's non-resumable copy (pipeline_v456.sh, live log in that session's scratchpad); scripts/pipeline.sh is the resumable one and needs 'touch data2/.dagger_done' before any resume because the running copy never writes that marker. PROJECTIONS from this run's own pace (DAgger 58 eps in 33 min; v4 72 s/epoch on 49k rows): S5 done ~06:45 (60 eps plus a few minutes of Metal encoder bench load), S6 v6 ~07:00 (encode ~25k contexts, 10 epochs on ~28k rows), S7 held-out ~07:25 (40 sync eps), S8 defend real-time ~07:35, ALL DONE ~07:35-07:40. WHEN DONE: 1) read S7 (oracle|v3|v5|v6 per held-out map: kills, items, ammo_left, hp) and S8 (defend real-time mean for v5, v6); 2) check the four pass conditions in this ticket's body; 3) flip the two default_value heads in src/main.rs to the best passing head, rebuild target/release, update README Numbers and AGENTS Status, note results here, close; 4) if v6 < v5 on held-out maps keep v5 and flag the RWR weights (beta 5, kills 10, items 3, health 0.05, ammo -0.05) as untested guesses; 5) only then run the idle-machine Metal latency bench (gd-ngu7 step 4). Metal work so far is in target-metal/ and touches nothing the pipeline reads; training caches stay f32 when we switch.

**2026-09-18T14:17:07Z**

RESULTS 2026-09-18 07:20 (pipeline ALL DONE 07:00; live log appended to scripts/pipeline.log). Offline: v4 test top1 0.9397, v5 0.9483, v6 0.9357 (v6 best epoch 1; RWR labels are not oracle labels so val top1 is not its objective). HELD-OUT MAPS (sync, 2 eps each; oracle baseline run separately because the script's grep for '  ep' only matches every 4th episode and S7 runs 2, so the pipeline log has no oracle rows). Mean kills / items / survived-60s: oracle 2.9 / 4.7 / 60%; v3 3.2 / 1.2 / 60%; v5 2.6 / 1.7 / 60%; v6 3.5 / 4.7 / 80%. Per map: map09 oracle 3.0/6.5, v3 3.0/3.0, v5 2.5/4.5, v6 3.0/4.0; map10 (oracle dies in 12 s) v5 the only survivor; map12 (oracle 0 kills, empties ammo) v5 1.5 kills survives, v6 0 kills 49/50 ammo left both eps = passive; e2m1 (the only map with pickups) oracle 11/17, v3 6.5/3, v5 6/4 dies both eps, v6 12.5/19.5; e2m2 nothing happens for anyone. DEFEND REAL-TIME 5 eps, rerun idle after the pipeline because Xcode was at 220% during S8: v3 14.0 (26 dec/s, p50 39 ms), v5 8.8 (17.7 dec/s, p50 47 ms), v6 6.8 (10.3 dec/s, p50 69 ms). The lower decision rates are policy-driven, not load: v5/v6 pick 'move forward' (hold 4) most of the time (recorded plays: v6 841 of 1896 decisions, v5 419 of 1652, v3 0 of 3728) which on defend_the_center walks into the enemies; 12 maps outweigh defend 5:1 in the training rows. CORPSE/OFF-ORACLE SHOTS (recorded sync plays map09+e2m1, 2 eps each; attack decisions where the oracle would not attack): v3 279 of 353 (79%), v5 278 of 333 (83%), v6 233 of 379 (61%); shots per kill 50 / 33 / 32. PATH >= 10k: not measured (S7 is sync; needs live --no-tui on a map). GATES: items >= oracle-20% (3.76): v6 PASS 4.7, v5 FAIL 1.7, v3 FAIL 1.2. defend real-time >= 12: v3 PASS, v5 FAIL, v6 FAIL. off-oracle shots ~0: all FAIL. path: unmeasured. No head passes everything, so the default stays heads/v3 until the maintainer decides; v6 is the best map policy (matches the oracle on items, beats it on kills and survival) and the worst on the scenario. Likely fix for the next round: rebalance defend vs map rows (or weight them), and revisit RWR's passive behaviour on map12 (beta 5, survival-dominated reward).

**2026-09-18T15:29:06Z**

CLOSED 2026-09-18 08:35 with heads/v3.safetensors kept as the default (maintainer's decision). Delivered: corpse filter, pickup whitelist, item-seeking oracle with ammo discipline, terminal-frame counters, collect/play --wad/--map, ITEMCOUNT, per-step counters in rows, train --rwr/--init-from, split rule, the v4/v5/v6 pipeline, the held-out evaluation and its oracle baseline. Not met: defend_the_center real-time regression for v5/v6 (move-forward bias from the 5:1 map-to-scenario row imbalance), off-oracle shots for every head including v3, path length unmeasured. Next round, if taken: rebalance or weight scenario rows vs map rows, revisit the RWR reward weights that made v6 passive on map12, and measure path length with live --no-tui on a map. v6 remains the best map policy (items = oracle, kills 3.5, 80% survival) and is kept in heads/.
