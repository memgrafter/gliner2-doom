---
id: gd-88vd
status: open
open: true
deps: []
links: [gd-eok7]
created: 2026-09-18T18:22:19Z
type: task
priority: 3
assignee: memgrafter
tags: [gliner2, doom, test]
---
# gliner2-doom regression check: parity plus head eval after every build

The crate has no tests and src/debertav2.rs is vendored, so a numerics change there is silent. Add one check that runs encoder-parity --reference onnx-fp32 (bar: max <= 5e-3, mean <= 5e-4) and eval of heads/v3 on data2/test.jsonl (top-1 within 0.3 of 0.7190), and fails the build on either. Needs the ONNX fp32 export and the Metal cache present; skip cleanly when they are not.
