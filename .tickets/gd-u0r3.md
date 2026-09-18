---
id: gd-u0r3
status: open
open: true
deps: []
links: [pro-zekx, gd-jf3i]
created: 2026-09-18T18:22:19Z
type: feature
priority: 2
assignee: memgrafter
tags: [gliner2, doom, policy, training]
---
# gliner2-doom policy quality: scenario/map row imbalance, off-oracle shooting, RWR passivity, path length

Four open defects from pro-zekx, one round of work. (1) Row imbalance: 12 maps outnumber defend_the_center 5:1 in the training rows; v5 and v6 learned 'move forward' and failed the real-time regression (8.8 / 6.8 vs v3 14.0). Rebalance or weight scenario rows, retrain, re-run the regression. (2) Off-oracle shooting: on recorded map plays 61-83% of attack decisions are ones the oracle would not take (v3 79%, v5 83%, v6 61%); no head has addressed it. (3) RWR passivity: v6 sits still on map12 (0 kills, 49/50 ammo left, both episodes); the reward weights (kills 10, items 3, health 0.05, ammo -0.05, beta 5) were guesses. (4) Path length >= 10k was never measured: S7 was sync play; run live --no-tui per held-out map. Evaluate with 10 episodes per map per head, not 2; 2 cannot separate heads within a few points.
