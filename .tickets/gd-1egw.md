---
id: gd-1egw
status: closed
open: false
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

## Notes

**2026-09-18T21:55:47Z**

SKIPPED 2026-09-18, not built. GLiNER2 cannot be expected to answer judgment questions, so a Jev-style battery over it would be the keyword policy with extra steps. Evidence: (paper, arXiv 2507.18546) classification is a softmax over per-label token embeddings via an MLP: a similarity judgment between label text and input text; trained on 254k GPT-4o-annotated examples where the classification labels are sentiment, intent and topic (SST-2, IMDB, SNIPS, Banking77, Amazon intent, AG News, 20 Newsgroups; zero-shot avg 0.72 vs GPT-4o 0.84); no decision, consequence, yes/no or instruction data, and no discussion of instruction-like labels. (our runs, pro-2tst notes) keyword fields read back at 97-100%; prose confused left/right 30-70%; one softmax over 12 world statements came out flat; 'is ammo low' answered low for ammo 26; bare yes/no labels carry nothing for a label matcher; prepending instructions to the text broke the System One questions (every support message became 'furious'). Predicted: 'hold trigger' only works when rephrased as a statement that co-occurs with 'Side: CENTER', i.e. a readable field in disguise; 'dodge' has no field to read; 'do not fire' contains 'fire' and would likely raise it. Jev per TypeSafe is trained for typed decisions; GLiNER2 is an extractor with a classifier head. Same API shape, nothing else. Branch jev-battery kept only for the ticket.
