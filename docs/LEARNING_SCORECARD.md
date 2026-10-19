# Track learning, hints and transfer

[Notebook pairs](../notebooks/learning/README.md) · [Blank-notebook playbook](BLANK_NOTEBOOK_PLAYBOOK.md)

Use one copy of this record for each family. A notebook that ran after copying every solution is
a worked example; it is not yet evidence of independent execution.

```text
Family / real practice dataset / date:
Input unit -> output unit:
Exact scorer, direction, output columns and permissions:
Leakage unit and saved fold IDs:
Three inspection observations:
Imbalance diagnosis and why the chosen response matches the score:

Baseline time to first score / checked CSV / reload:
TODO blocks completed independently / total attempted:
Exercise checks passed before revealing a reference:
Hints used: concept / implementation / full reference:
Solutions revealed before attempting a block:
Assistant calls, failed calls, and the function requested each time:
Most costly mistake and its observable symptom:

Advanced idea implemented / what the original winner additionally did:
Same-fold baseline score -> candidate score:
Implementation minutes / extraction seconds / fit seconds / full inference seconds:
Extra data, packages, model assets and memory required:
Keep baseline / keep candidate / uncertain, with evidence:

Next-day weakest-block reconstruction (no notes): pass / needs work
Blank-notebook reconstruction (no repo/notes): time / hints / score / CSV / reload
Unfamiliar-schema or metric transfer: change needed / result
Teammate review: can explain / can reproduce / blockers
Next targeted drill:
```

Judge each skill independently:

| Skill | Evidence of readiness |
|---|---|
| Recognize the task | Explain the prediction unit, output dimension, loss and evaluator before choosing a model. |
| Inspect | Find a label/shape/path problem using examples or a plot, not a training crash. |
| Validate | Justify independent/group/time splitting and hand-check the scorer's boundary cases. |
| Handle imbalance | Compute support using training only and explain why weighting may help or hurt the official score. |
| Implement | Fill the missing functions, run their checks and explain each operation. |
| Diagnose | Explain why a failing toy check points to a shape, metric, fit boundary or update error. |
| Improve | Connect one error pattern to one change and compare score versus total runtime on the same folds. |
| Deliver | Restore the saved candidate and produce a correctly ordered file from a fresh runtime. |
| Transfer | Rebuild under a new schema with no saved code; use the 2k assistant for a bounded gap and survive its failure. |

For a 3–4-hour session, spend about 15 minutes retrieving yesterday's weak block, 60–90 minutes on
the baseline or advanced implementation, 45–60 minutes inspecting errors and one experiment,
then 30–45 minutes on reload, delivery and this record. If a correctness gate fails, the next
session repairs that gate before moving to another architecture.

Do two assistant-free assessments near the end. On a separate tool-permitted repeat, ask DeepSeek
for only a small function with exact arguments/shapes, then test it. Record whether the call saved
time. The goal is familiarity that transfers to a new problem, not completing all reveals quickly.
