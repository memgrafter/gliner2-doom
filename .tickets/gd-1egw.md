---
id: gd-1egw
status: open
open: true
deps: []
links: []
created: 2026-09-18T21:07:12Z
type: feature
priority: 2
assignee: memgrafter
tags: [gliner2, doom, jev, policy]
---
# Jev-style typed battery over GLiNER2: typed questions composed in code, steerable by an instruction, vs the trained head on the same maps

The experiment from gd-efsi. Same reader (frozen GLiNER2 encoder via gliner25-rs classification, as the keyword policy uses), same state, same maps; only the decision step differs. NEW POLICY --policy battery: the state line is asked a small set of typed questions in one pass, each a separate small label set answered in isolation: hold trigger (yes/no), dodge (yes/no), top goal (kill nearest / grab pickup / explore / retreat), turn (left / right / none), and an optional instruction string (--instruction 'do not fire, simply dodge') prepended to every question's state. Hand-written composition in code turns the answers into one of the existing ten options with the existing holds; no new buttons. EVALUATION: the same held-out maps as pro-zekx (map09, map10, map12, e2m1, e2m2) and defend_the_center real-time, 10 episodes per map per policy: battery vs battery+instruction vs v3 head vs v6 head. Report kills, items, survival, path, decisions/s, and whether the instruction changes behaviour measurably. Off-oracle shot rate from --record. PASS is not a bar; the point is the comparison. Keep the trained-head path untouched; branch jev-battery.
