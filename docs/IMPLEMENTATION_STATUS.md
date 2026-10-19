# Current implementation status

Updated 19 October 2026. [Guide index](README.md) · [Practice notebooks](../notebooks/learning/README.md)

## What is available

| Route | Scope | Repository needed at execution? |
|---|---|---|
| Seven learning pairs | Home Credit, House Prices, Jigsaw, Cassava, TGS, Wheat, Shopee | No; all 14 notebooks are self-contained |
| Competition kit | 16 task templates, shared trainer, adapters, CLI and validation tools | Yes |
| Reconstruction drills | Text/table answer key, image exercise and blank-notebook assessments | No project imports in the answer key |

The 11 competition stages plus optional Kaggle acquisition now contain short, focused cells.
Each baseline has eight named exercises: **56 exercises across seven families**. Contracts,
two hint levels, small executable checks and interpretation questions support each lesson.
Every baseline code cell retains a closed full reference directly afterward. Advanced notebooks
use explained worked components, checks and controlled comparisons. Real-dataset validation must
establish whether their extra cost is justified.

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

## Guided lesson design

The first code cells now contain 22–26 lines of imports. Runtime helpers are separate; storage,
prediction, metric, training and delivery definitions appear in their relevant stages. Supplied
definitions use standard Jupyter source-fold metadata. Configuration stays visible and omits
unrelated model settings. HTML hints and full references start closed.

Small behavioral checks cover unknown categories, log-price conventions, independent multilabel
scores, one optimizer update, rectangular mask geometry, best-checkpoint preservation, self and
non-self matching, and shuffled literal submission IDs. Neural batches are checked before full
training. Metric exercises are introduced before fitting and reused on real validation outputs.
Each pair ends with a task-specific transfer exercise, reconstruction drill and measured ablation.

The [design contract](LEARNING_NOTEBOOK_DESIGN.md) explains authoring and study conventions.
Historical audits and Task 2 research remain in [the archive](archive/README.md). Your edited
Home Credit notebook was preserved as an ignored personal copy before redesign; local datasets
and personal run directories were retained.

## Verification

Commands below use the installed `uv.exe` at `C:/Users/DELL/.local/bin/uv.exe` because uv is not
on this machine's PATH. `--no-sync` preserves the existing optional dependency groups.

| Exact command | Redesign result |
|---|---|
| `uv --cache-dir .uv-cache run --no-sync ruff format .` | Passed |
| `uv --cache-dir .uv-cache run --no-sync ruff check .` | Passed |
| `uv --cache-dir .uv-cache run --no-sync olp-ai build-notebooks` | Generated all 30 notebooks |
| `uv --cache-dir .uv-cache run --no-sync pytest --basetemp .pytest_cache/redesign-final --tb=short` | 120 passed, 1 warning in 212.48 seconds; exit status 0 |
| `uv --cache-dir .uv-cache run --no-sync pytest tests/test_notebooks.py tests/test_learning_notebooks.py -k 'not execute_offline and not copied_notebook and not durable_bundles' --basetemp .pytest_cache/redesign-presentation-recheck --tb=short` | 30 passed, 18 deselected in 18.54 seconds after presentation-only changes |
| `uv --cache-dir .uv-cache run --no-sync python -m compileall -q src tests scripts notebooks` | Passed |
| `uv --cache-dir .uv-cache lock --check` | Passed; 261 locked packages, no dependency changes |
| `git diff --check` and `git diff --cached --check` | Passed |

All 14 notebooks passed Jupyter schema validation: 56 exercises, 167 check cells and 532 closed
baseline references. Credit and Salt HTML exports retained balanced closed details for hints,
references and reasoning prompts. Twenty-two Markdown documents and 118 local links/heading
targets passed validation. All 30 maintained notebooks have empty outputs and execution counts.

The final run saves output and exit status under the ignored `.pytest_cache/` directory.
Individual cells are compiled; completed baseline answers and advanced components
execute through all non-setup cells individually with repository imports and network blocked.
Seven additional regression cases inject plausible broken student functions and verify that
the embedded feedback fails, including a no-op optimizer update and self-only retrieval.
The final visibility/introductory-text adjustments preserved identical executable ASTs. One
supplemental test attempt encountered a Windows permission error while renaming a temporary TGS
dataset folder; the full runs and the fresh-directory follow-up passed that case without changes
to the downloader. It remains an observed intermittent local filesystem limitation.
Joblib emitted its Windows physical-core discovery fallback warning. Optional wandb also printed
ignored temporary-folder cleanup permission errors at process exit in the Windows sandbox;
these did not fail the tests or change the zero exit status.

Separate Python processes cover table/image routes. The suite checks notebook
generation and exercise placement, inline metrics, split boundaries, data/download safety,
checkpoints, durable artifacts and full synthetic delivery. Download tests simulate the CLI and
all seven dataset layouts; they do not access a Kaggle account.

These checks establish local CPU correctness on small synthetic data. They do not establish
leaderboard quality, real-data throughput, clean Colab GPU execution, Drive restart recovery,
organizer uploads or live DeepSeek behavior. Exact COCO/CIDEr and other task-specific metrics
still require the official adapter. Complete the [scorecard](LEARNING_SCORECARD.md) and a timed
team rehearsal in the organizer's permitted environment before treating yourself as contest-ready.
