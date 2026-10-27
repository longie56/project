# Frozen revised-source identity

The revised POMO model lock was written on 2026-10-24 after internal paired evaluation and replay audit. The files below are copied byte-for-byte from the frozen compact formal source snapshot. SHA-256 values distinguish them from later working-directory edits.

| File | SHA-256 |
| --- | --- |
| `revision_v3/policy.py` | `eb6d583feacf4243b314d7a022b3aa985c20e00fe2b86c5a036deaa84ae7ef69` |
| `revision_v3/environment.py` | `446d379838b4a5600b76ae3590436d298daa2f53887eac120e74cb0429a310ed` |
| `revision_v3/common_eval.py` | `eac0bcd8b5f659393f908cba3c677c2ecc1e345e0d19dcd2295a9884175e08ff` |
| `revision_v3/selection.py` | `52927fdef5e7419363e8426359c90fc7451d031ec1cbd48c2b5fddf7326518bb` |
| `revision_v3/task_spec.json` | `4e1f1a8071e30eb8cd46ebd9a9be86fdbdd794406d976ddc8523a234b700d8a4` |
| `env/revision_v3_gu_physics.py` | `b214cf6390a64f0b33b8f342ec4e0639b80ba6e2972083fcb5e90c10832a4b79` |
| `formal_runs/20261022/first_dispatch_core.py` | `fd564f9311be962d9c87cf18d80cbc91b5b200754c59ee19b8377b8c50accbd4` |
| `formal_runs/20261022/paired_eval.py` | `145a4266b6b6e13025993c0b7c0854c4b73ac9c0723595b69b09258c30bd08e1` |
| `formal_runs/20261024/pomo_runner_common.py` | `f1c50d775e90db05be8aac61e8252ca9625c98ca5ffcebaf8b26c02ee02b30b0` |
| `formal_runs/20261024/pomo_timing_gate.py` | `7f763ecc979c2111d35570f9ecb520fa405fc0588ab120cdc73046f04a5d78a5` |
| `formal_runs/20261024/manifest_cleanup.py` | `8273e15363a1228d5ad4d5538b7a16da7b64ab22f8fbfa2d0752aaac56a97477` |

`revision_v3/train.py` is the original compact formal base-training source. `formal_runs/20261022/pareto_continue.py` is the original 40-update continuation source. These scripts require checkpoint and instance artifacts recorded by their experiment protocols; none are distributed here.

The later `diagnostics/cleanup_selection_20261027/` source preserves the model and candidate pool, then replays each candidate after load cleanup before selecting S/B. Its 945-setting W05-I comparison is a *post hoc* development analysis, separate from the frozen main-panel evaluation. It must not be presented as a blind test or replacement for the original main table.
