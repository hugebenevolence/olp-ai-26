# Current implementation status

Updated 19 October 2026. [Guide index](README.md) · [Practice notebooks](../notebooks/learning/README.md)

## What is available

| Route | Scope | Repository needed at execution? |
|---|---|---|
| Seven learning pairs | Home Credit, House Prices, Jigsaw, Cassava, TGS, Wheat, Shopee | No; all 14 notebooks are self-contained |
| Competition kit | 16 task templates, shared trainer, adapters, CLI and validation tools | Yes |
| Reconstruction drills | Text/table answer key, image exercise and blank-notebook assessments | No project imports in the answer key |

Each learning notebook has 11 competition stages plus optional Kaggle acquisition (12 code cells).
Baseline exercises have a closed reference cell immediately after each code cell: 84 references
across seven baselines. Advanced notebooks are reduced, evidence-backed studies; real-dataset
validation must establish whether their additional cost is justified.

All notebooks are generated from adjacent `*_template.py` sources. Make a personal notebook copy
before studying; rebuilding replaces generated templates. The shared kit and learning lessons
deliberately have separate runtime implementations so a learning notebook can run by itself.

## Implemented safeguards

- Training-fold preprocessing, group/time split guards, inspection plots and imbalance diagnosis.
- Explicit metric conventions, including log-price RMSE, TGS masks, Wheat boundary rules and
  Shopee set matching. Generic detection AP50 remains a practice scorer, not COCO mAP.
- ID-aligned submissions with duplicate/missing/extra-key, order, label and finite-value checks.
  Literal CSV IDs are preserved. Empty text/RLE fields require task-specific permission.
- Hardware-aware precision, delivery reserves, atomic best checkpoints and best-state reload.
  The small inline learning loop supports AdamW, one update per batch and no scheduler;
  the shared trainer also supports accumulation and scheduling. Detection uses a visible FP32 loop.
- Saved preprocessing, configuration, predictions and reports, with unique durable learning-run
  directories. Reloading for inference does not promise exact optimizer/scaler/random-state resume.
- Pretrained defaults disabled, with explicit permitted local assets. HF loading is local-only.
  Archived Task 2 profiles have their own permission and require staged weight files.
- Optional Kaggle download cells, disabled by default: cache reuse, safe ZIP extraction, layout
  checks and preservation of existing data. Client output is withheld from saved notebook output.
  The cells do not accept rules, submit predictions or download model weights.
- Colab data staging on runtime storage and constrained installation of missing small dependencies
  in kit notebooks. Learning notebooks install nothing automatically.

See [dataset acquisition](LEARNING_DATASETS.md), [Colab operation](COLAB_RUNBOOK.md) and
[adaptation recipes](CUSTOMIZATION_GUIDE.md) for instructions rather than implementation history.

## Cleanup

Removed 28 unused helper definitions and 32 unused imports from the learning sources, retaining
their self-contained execution and the one-batch training exercise. The CLI replaces a duplicate
builder script. Old reconciliation snapshots and an empty accidental self-submodule pointer were
removed; committed versions remain recoverable from Git history. Historical audits and Task 2
research are in [the archive](archive/README.md). Repeated setup text and obsolete verification
logs were consolidated. No dataset or personal run directory was removed; an edited Home Credit
template was preserved as an ignored personal copy before regeneration.

## Verification

Commands below use the installed `uv.exe` at `C:/Users/DELL/.local/bin/uv.exe` because uv is not
on this machine's PATH. `--no-sync` preserves the existing optional dependency groups.

| Exact command | Cleanup result |
|---|---|
| `uv --cache-dir .uv-cache run --no-sync ruff format .` | Passed; 108 files unchanged on final run |
| `uv --cache-dir .uv-cache run --no-sync ruff check .` | Passed |
| `uv --cache-dir .uv-cache run --no-sync olp-ai build-notebooks` | Generated all 30 notebooks |
| `uv --cache-dir .uv-cache run --no-sync pytest --basetemp .pytest_cache/cleanup-resumed --tb=short` | 113 passed, 1 warning in 223.17 seconds; exit status 0 |
| `uv --cache-dir .uv-cache run --no-sync python -m compileall -q src tests scripts notebooks` | Passed |
| `uv --cache-dir .uv-cache lock --check` | Passed; 261 locked packages, no dependency changes |
| `git diff --check` and `git diff --cached --check` | Passed |

Additional inspection checked 21 Markdown documents, all 114 local links/heading targets and
balanced code fences. All 30 maintained notebook artifacts have empty outputs/execution counts.
The earlier test session's final output was unavailable after interruption, so the resumed run
saves its log and exit status under the ignored `.pytest_cache/` directory.
Joblib emitted its Windows physical-core discovery fallback warning. Optional wandb also printed
ignored temporary-folder cleanup permission errors at process exit in the Windows sandbox;
these did not fail the tests or change the zero exit status.

The suite exercises completed learning solutions outside the repository with project imports and
network access blocked; separate Python processes cover table/image routes. It checks notebook
generation and exercise placement, inline metrics, split boundaries, data/download safety,
checkpoints, durable artifacts and full synthetic delivery. Download tests simulate the CLI and
all seven dataset layouts; they do not access a Kaggle account.

These checks establish local CPU correctness on small synthetic data. They do not establish
leaderboard quality, real-data throughput, clean Colab GPU execution, Drive restart recovery,
organizer uploads or live DeepSeek behavior. Exact COCO/CIDEr and other task-specific metrics
still require the official adapter. Complete the [scorecard](LEARNING_SCORECARD.md) and a timed
team rehearsal in the organizer's permitted environment before treating yourself as contest-ready.
