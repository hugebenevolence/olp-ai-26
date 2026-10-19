# Learn seven competition problem families

Make a personal copy of a generated notebook (for example `01_credit_attempt.ipynb`) before
filling it in. Keep maintained templates unchanged; save your edited attempt with its run bundle.

**Start with the baseline column.** Each baseline has **eight focused exercises**, two levels
of expandable hints, nearby executable checks, and interpretation questions. The eleven competition
stages remain, plus optional dataset acquisition; each stage now contains several small cells.
Every baseline code cell still has a closed **Reveal expected code** reference immediately after it.
Attempt an exercise, run its checks, explain the result, then consult the reference if needed.
Equivalent correct implementations are welcome.

| Cell type | What you do |
|---|---|
| Supplied helper | Run it; its code is folded where supported. Storage/safety plumbing is supplied. |
| Configuration | Edit paths, resources, model permission and task settings. |
| Exercise | Complete the named function or marked operation using its input/output contract. |
| Check | Predict the outcome, run the assertions and fix the first failure. |
| Worked component | Follow an implemented step, then explain why it fits the task. |

The learning rhythm is **understand → implement → check → interpret → reuse**. Small checks cover
unseen categories, log-price conventions, multilabel averaging, optimizer updates, mask geometry,
checkpoint selection, set matching and shuffled submission IDs. Actual batch checks precede neural
training. After each stage, write your answer to **Stop and explain** before opening its checkpoint.

Every pair includes inspection plots, a leakage-aware split, a training-fold imbalance audit and
response, model/loss reasoning, validation errors, score/runtime records, reload, inference and a
checked submission. Advanced notebooks use explained worked components, checks and a controlled
comparison task with the same data contract. They remain complete implementations.

**All 14 learning notebooks are self-contained.** Copy just one `.ipynb` into a new folder or
upload it directly to Colab. Its code uses only installed third-party libraries and your data or
explicitly permitted local model assets. It does not import the project, read another Python file,
modify the Python search path, install the project, or fetch code or weights. Data acquisition
uses the optional online cell; it is disabled during ordinary offline study. The 16 older
competition-kit notebooks in the parent folder remain repository-assisted references.

| Family and representative dataset | Fill-in baseline | Advanced / SOTA study |
|---|---|---|
| Imbalanced mixed tables — Home Credit | [Logistic regression](01_credit_baseline_template.ipynb) | [LightGBM + history aggregates](01_credit_advanced_template.ipynb) |
| Long-tailed regression — House Prices | [Ridge in log space](02_prices_baseline_template.ipynb) | [OOF stacking](02_prices_advanced_template.ipynb) |
| Multilabel text — Jigsaw Toxic Comments | [TF-IDF + one-vs-rest](03_toxic_baseline_template.ipynb) | [Two-layer BiGRU](03_toxic_advanced_template.ipynb) |
| Multiclass images — Cassava | [Small scratch CNN](04_leaves_baseline_template.ipynb) | [Single EfficientNet-B4](04_leaves_advanced_template.ipynb) |
| Binary masks — TGS Salt | [Dense mask predictor](05_salt_baseline_template.ipynb) | [ResNet34 U-Net + threshold selection](05_salt_advanced_template.ipynb) |
| Multiple boxes — Global Wheat | [MobileNet Faster R-CNN](06_wheat_baseline_template.ipynb) | [ResNet FPN + confidence selection](06_wheat_advanced_template.ipynb) |
| Multimodal matching — Shopee | [Title TF-IDF matching](07_matching_baseline_template.ipynb) | [Local image/text embedding fusion](07_matching_advanced_template.ipynb) |

## What “SOTA study” means here

The advanced column studies methods supported by documented Kaggle solutions. It **does not claim
that these reduced notebooks reproduce a winning score, are current global SOTA, or are the fastest
possible implementation**. A leaderboard placement establishes a score in that competition;
efficiency additionally needs measured implementation, training and inference cost on your runtime.

Each notebook links primary evidence and states exactly which component is implemented and what
was omitted. Read the [evidence ledger](../../docs/KAGGLE_METHOD_EVIDENCE.md) before interpreting a
rank or carrying a technique into OlpAI. Several historical winners used resources or test fitting
that cannot be assumed legal in your contest. This curriculum keeps those components out.

## Setup and data

For local preparation:

```powershell
uv sync --locked
uv run olp-ai build-notebooks
```

These commands maintain the repository; **a copied notebook needs neither uv nor a checkout**.
See [local setup](../../README.md#local-setup) for dependency groups. In another environment,
install the needed libraries before
timed offline practice: NumPy, pandas, scikit-learn, matplotlib, PyTorch and Pillow; add torchvision
for image/mask/box lessons, timm for advanced Cassava or matching, LightGBM for advanced credit,
and transformers for matching with local text encoders. Add `kaggle>=2.2,<3` only if using the
optional online download cell. The feature-cache matching path does not
need transformers or timm. The notebooks perform no automatic installation.

For Colab, upload the notebook directly and follow the
[learning-notebook setup](../../docs/COLAB_RUNBOOK.md#choose-the-notebook-route).
Set `DATA_DIR` to runtime storage and `PERSISTENT_DIR` to durable storage in stage 1.
Keep your edited notebook with its unique run bundle. For manual ZIP staging, files must be
at the ZIP root; otherwise point `DATA_DIR` at the extracted subfolder.

For real data, follow the [dataset and authentication guide](../../docs/LEARNING_DATASETS.md).
**Logging in does not start a download:** choose a dedicated empty `DATA_DIR`, then enable
`DOWNLOAD_DATA=True` in stage 1a. Existing fixtures/manual folders are preserved; completed
downloads and cached ZIPs can be reused. This optional online stage downloads competition data,
never model weights. Leave it disabled for offline practice.

To create small offline schema fixtures:

```powershell
uv run olp-ai practice-data data/learning --task learning
```

Locally, `DATA_DIR` defaults to the matching `Path.cwd() / "data" / "learning" / FAMILY` folder.
In Colab it defaults to local runtime storage. Reuse existing fixture folders; creation deliberately
refuses to overwrite them. Fixture scores test plumbing only. Keep their
`grading/answers.csv` closed during the exercise and never mix fixtures into contest training.
To use original Kaggle data, change `DATA_DIR` to its staged folder instead. The configuration
limits the training manifest for short practice; set `MAX_ROWS=None` only when you budget a full run.

Pretrained initialization defaults to **False**. An advanced image model with that setting is a
randomly initialized architecture study, not a transfer-learning result. If a practice task permits
the exact checkpoint, stage it locally and set both permission and provenance explicitly.
The advanced matching notebook requires permitted local encoders or a provenance-checked feature
archive; it fails clearly until these are supplied. A feature-only archive can infer only its
recorded IDs. For new private inputs, retain the actual encoders/tokenizer/configuration too.

## How to learn from each pair

1. **Attempt:** read the task and inspection output. Write your metric, split and imbalance decision.
   Implement one exercise and run its checks before moving on. Use Hint 1, then Hint 2 before the
   full reference. The first guided study may take longer; aim for 45–60 minutes to a complete
   small-pack route during the later reconstruction drill.
2. **Explain:** after each reveal, describe why the reference works and why a plausible alternative
   would fail. Investigate differing predictions rather than copying until the cell turns green.
3. **Compare:** run the advanced notebook on the same source pack and folds. Keep its new candidate
   directory. Compare the official score, minority/group errors and total extraction/fit/inference
   time using `comparison_row.csv`. An advanced model is not automatically the selected model.
4. **Retrieve:** the following day, reproduce your weakest block without notes. Then close both
   notebooks and rebuild the minimal workflow in a new notebook. Use the
   [blank-notebook playbook](../../docs/BLANK_NOTEBOOK_PLAYBOOK.md) for the no-repository assessment.
5. **Transfer:** change one aspect: unfamiliar column names, shuffled sample IDs, a rare class,
   acquisition groups, a new metric, or a smaller runtime budget. Explain the required code changes.

Use the [learning scorecard](../../docs/LEARNING_SCORECARD.md) to track hints, errors and transfer,
not just notebook completion. For the supplied 2k assistant, request one missing function with its
signature/shapes; do not paste the notebook. Hide solutions again before a repeat. HTML details
sections start closed; actual appearance depends on the notebook viewer. Use a renderer that
supports HTML details, such as JupyterLab, if another viewer displays the solution text expanded.
Solution cells are additionally tagged `solution`; they are study aids, not hidden-test security.

## The eleven competition stages

1. Bootstrap, local paths, explicit resources and time budget.
2. Load the original task schema and check prediction-unit IDs.
3. Inspect distributions, missingness and visual examples.
4. Freeze the split and assert identity/group/time boundaries.
5. Diagnose imbalance and choose a score-appropriate response.
6. Construct train-only features or consistent datasets/loaders.
7. Match model outputs and loss to the prediction contract.
8. Train, measure cost and preserve a reloadable candidate.
9. Validate globally, visualize errors and compare justified changes.
10. Reload and infer the complete test manifest.
11. Align IDs, validate the actual CSV, save the recipe/report and synchronize artifacts.

The additional stage **1a** acquires data before stage 2 and is skipped by default. It is supplied
scaffolding rather than another timed algorithm exercise; baseline copies still have a collapsed
reference immediately after this code cell.

Baseline notebooks intentionally stop at an unfilled block when run unchanged. Automated checks run
the expected code from their solution cells; advanced notebooks run their actual worked cells.
Sources are the adjacent `*_template.py` files. Regenerate artifacts after editing them with
`uv run olp-ai build-notebooks`; never hand-edit generated notebooks as the maintained source.
All runtime helpers are visible inside those sources and generated cells. Inline definitions are
intentional for these lessons; changing a package helper does not update the standalone copies.
See the [lesson design contract](../../docs/LEARNING_NOTEBOOK_DESIGN.md) before editing the sources.
Regression checks execute every completed lesson outside the repository working directory with
project imports and network connections blocked. Fresh-process checks cover table and image
routes as well. Baselines still require you to fill their exercises before ordinary Run All works.

Self-contained does not mean you should memorize the entire storage/report scaffold. Reconstruct
the data contract, split, model/loss, optimizer update, exact metric, inference and ID-aligned CSV
first. Then add checkpoint recovery and reporting. The inline training loop supports AdamW and
one optimizer step per batch with no scheduler; its small configuration deliberately rejects
unsupported settings. For a 2k assistant request, provide one function signature, shapes and a
failing example, then implement that function in the appropriate cell. Final closed-book drills
start in an empty notebook with none of these files available.

NER, generation, video and normal-only anomaly detection remain extension families in the
[existing notebook catalog](../../docs/BASELINE_CATALOG.md). After these seven routes work, choose
an extension based on the released contest task rather than memorizing another architecture list.
