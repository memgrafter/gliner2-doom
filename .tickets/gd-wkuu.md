---
id: gd-wkuu
status: closed
open: false
deps: []
links: []
created: 2026-09-18T18:22:19Z
type: chore
priority: 3
assignee: memgrafter
tags: [gliner2, doom]
---
# gliner2-doom housekeeping: delete target-metal and data, add --window to collect

target-metal/ (956 MB) was the side build used while the Metal port was verified; the real build carries Metal now. data/ (17 MB) is the pro-yjme rollouts, superseded by data2. Both ignored. collect has no --window flag, so the oracle cannot be watched playing.

## Notes

**2026-09-18T18:23:01Z**

Done 2026-09-18: target-metal (956 MB) and data (17 MB) deleted; collect --window added (640x480 visible engine, same as live), smoke-tested with a 1-episode oracle rollout. Commit 38f2956.
