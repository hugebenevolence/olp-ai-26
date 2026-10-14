# Preparation guide

Start with **[your first practical session](PRACTICE_QUICKSTART.md)**. It creates offline practice
data, walks through a complete baseline, and shows what to save and measure.

## Choose your next action

| You want to… | Read this |
|---|---|
| Run a baseline today | [Practice quickstart](PRACTICE_QUICKSTART.md) |
| Follow the 15-day training | [Daily program and readiness gates](15_DAY_PREPARATION.md) |
| Choose a notebook for an unfamiliar output | [Task and notebook catalog](BASELINE_CATALOG.md) |
| Change a split, metric, model, loss, or inference path | [Customization cookbook](CUSTOMIZATION_GUIDE.md) |
| Set up Colab, move data, or recover a session | [Colab runbook](COLAB_RUNBOOK.md) |
| Find accepted model names | [Model catalog](MODEL_CATALOG.md) |
| Understand what was repaired and verified | [Current implementation status](IMPLEMENTATION_STATUS.md) |
| Understand the original review findings | [Historical readiness audit](READINESS_AUDIT_2026-10-14.md) |

## Recommended practice order

1. Text TF-IDF and tabular learning: complete workflows with CPU notebook execution tests.
2. Image classification: inspect a batch, fit a tiny subset, then validate a realistic split.
3. Binary segmentation: inspect original masks and verify metric/encoding conventions.
4. One unfamiliar output family: NER, retrieval, generation, or detection.
5. Restart recovery and team rehearsal.

The catalog states each template's exact default contract. A completed practice notebook still
needs the organizer's data mapping, permission checks, evaluator, and output specification.

## Archived Task 2 material

These pages describe a particular normal-only anomaly task and its recorded experiments. Use
them as a case study after the basic workflow works; do not import their permissions or thresholds
into another task.

- [Task 2 operational runbook](TASK2_ANOMALY_DETECTION.md)
- [0.577 baseline audit](TASK2_0577_AUDIT.md)
- [AnomalyDINO research comparison](TASK2_ANOMALYDINO_RESEARCH.md)

[Return to the repository overview](../README.md).
