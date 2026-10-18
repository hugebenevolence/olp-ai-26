# Build a baseline with no saved code

[Guide index](README.md) · [15-day program](15_DAY_PREPARATION.md) ·
[Standalone study answer key](../scripts/blank_baseline.py) ·
[Image reconstruction exercise](BLANK_NOTEBOOK_CV.md)

For guided learning first, use the [seven paired notebook exercises](../notebooks/learning/README.md).
This page describes the subsequent assessment with those notebooks closed.

**Train to write a small, complete solution from a blank notebook.** Treat internet, this repo,
personal notes, and prepared weights as unavailable. Use the supplied DeepSeek assistant only
within the task's permitted use. The **2k window is your stated deployment limit**, not a claim
about every deployment of deepseek-r1-distill-qwen-32b.

The earlier template path teaches adaptation, but it does not prove you can reconstruct a solution.
This path starts on Day 1. Read an answer key once, close it, reconstruct, run, and compare afterward.
On contest day, the code and notes below are available only if the rules explicitly permit them.

## What you need to remember

Remember the sequence **contract → inspect → split → fit → score → save/reload → predict → check**.
Memorize a few APIs through repeated use, rather than the whole answer key. The model constructor
usually takes fewer lines than the data and submission plumbing.

Write this task card in the new notebook before writing a model:

```text
One input / one prediction unit:
Train files, target, test files, ID keys:
Allowed packages / data / exact weights / assistant use:
Leakage unit and split (person, specimen, video, time, near duplicate):
Exact scorer, label universe, averaging, higher/lower is better:
Output columns, ordering, encoding, archive format, inference limit:
Submission quota, freeze deadline, delivery reserve:
```

Inspect the installed runtime before choosing libraries. Do not spend the first hour rebuilding a
familiar environment, downloading dependencies, or looking for unapproved weights. In Colab keep
its numerical/CUDA stack; on an organizer machine use its actual paths and persistence mechanism.
An unavailable package means choosing a supported baseline, not silently installing from the internet.

## Your first 60 minutes

These are practice targets for familiar classification formats, not guaranteed training times for
all tasks. Start the clock after receiving the files; include inspection and coding.

| Time | Produce | If blocked |
|---|---|---|
| 0–8 min | Task card; print shapes, columns, two examples, class counts, sample submission. | Resolve prediction unit and scorer before choosing the model. |
| 8–15 min | One fixed legal split; inspect overlapping IDs/groups/duplicates. | Use groups/time when required; do not replace a difficult split with an easier leaking one. |
| 15–25 min | Tiny data path and a first fit. For CV inspect a batch and fit a tiny subset. | Reduce data/resolution/batch size; fix labels/shapes before changing architecture. |
| 25–35 min | Validation predictions and the exact score; compare with a dummy baseline. | Hand-check a small fixture. Never average per-batch F1. |
| 35–45 min | Test predictions with IDs, validated output, saved model and recipe. | Debug output alignment independently of training. |
| 45–60 min | Reload check; first upload when permitted and useful; preserve its matching bundle. | Protect the working baseline; postpone tuning. |

A dummy output can test the serializer locally. Spending a limited submission slot on it is a
separate decision: prefer a learned baseline, respect quotas, and confirm actual acceptance when
uploading. A valid CSV alone proves neither useful predictions nor accepted delivery.

## The minimum code repertoire

| Output contract | First learned route | First evidence-driven change |
|---|---|---|
| One text label | TF-IDF → logistic regression on CPU. | Character features or class weighting if errors justify them. |
| One table label | Train-fitted numeric/categorical preprocessing → logistic regression. | Available tree model or one useful feature on the same split. |
| One numerical value | Train-fitted preprocessing → Ridge or an available regression tree. | Inspect residuals; match MAE/RMSE/other official score. |
| One image label | Small CNN from scratch; permitted local weights only if explicitly provided/allowed. | Learning rate, resolution, modest task-valid augmentation. |
| Binary masks | Small dense predictor with image/mask transforms kept together. | Correct resolution, foreground/ignore convention and validation-only threshold selection. |
| Retrieval ranking | Training-corpus TF-IDF similarity, if suitable for the task's corpus rules. | Word/character features; validate exact ranking metric and exclusions. |

These are starting routes, not equal-strength solutions. Detection, NER, generation, normal-only
anomaly detection, and multimodal tasks need their own output adapter and evaluator. Learn one
additional family in depth after the common workflow works. Do not convert an unfamiliar task
into classification merely because that is the code you remember. If generative assets are not
available, a task-legal retrieval baseline may be worth testing; it is not a substitute for checking
the statement or a guarantee of a useful score.

### Study answer key: text and tabular classification

[scripts/blank_baseline.py](../scripts/blank_baseline.py) is intentionally standalone: no project
imports, downloads, pretrained assets, or framework trainer. This exception to the normal shared-kit
workflow exists specifically to train reconstruction without repository access. It uses pandas,
NumPy, and sklearn; check that those are installed in the contest runtime.

Its six marked cells cover input integrity, splits, replaceable models, scoring, delivery, and wiring.
The command-line wrapper and defensive checks are an answer key, not a line-count memorization goal.
Recreate the short happy path first, then add checks before submission. Learn the reason for each
check so you can recreate it when the schema changes.

For text, the part you should soon type without help is just this core, after defining a legal
split `a, b` as row positions. It is a fitting fragment, not a complete submission:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=500))
model.fit(train.iloc[a]["text"].fillna(""), train.iloc[a]["label"])
validation_predictions = model.predict(train.iloc[b]["text"].fillna(""))
test_predictions = model.predict(test["text"].fillna(""))
```

The other stages supply trustworthy validation, persistence and delivery. Keeping those stages
separate lets you replace the model without rewriting the ID handling or changing the folds.

Supported practice contract: single-label classification; `id` and `label`; a `text` column or tabular
features; `train.csv`, `test.csv`, `sample_submission.csv`. IDs and labels are literal strings.
Sample IDs can be permuted; prediction alignment is by ID. The scorer supports **accuracy** and
**macro F1 over the full training label vocabulary**, with zero division returning zero. This is an
explicit practice convention: change it if the official scorer uses a different label universe,
averaging, sample weights, probabilities, or prediction format. Multilabel, probability columns,
regression, composite IDs, or non-CSV output require deliberate adaptation.

Before the timer, create a pack using the [quickstart](PRACTICE_QUICKSTART.md). For an untimed
answer-key check on your local machine:

```powershell
uv run python scripts/blank_baseline.py data/practice/text/data outputs/blank-text-01 --task text --metric macro_f1
uv run python scripts/blank_baseline.py data/practice/tabular/data outputs/blank-table-01 --task tabular --metric accuracy
```

Use a **new output directory** for each candidate. For a pack with the appropriate metadata, choose
`--split group --split-column person_id` or `--split time --split-column timestamp`. The split column
is excluded from tabular features. Chronological splitting keeps equal timestamps together; add
task-specific purge gaps or entity constraints yourself. Stratification is only appropriate when
examples can reasonably be treated as independent. Detect repeated content/identities during
inspection; the script cannot infer the correct leakage unit for you.

Outputs: `pipeline.pkl`, `submission.csv`, and `run.json` with split IDs, labels, metric and elapsed
time. The model stays fitted on the training fold; no test fitting or all-training refit occurs.
The script reloads its saved pipeline and checks predictions in-process. Your recovery drill must
also restart Python and restore features/preprocessing from the saved recipe. Add the task's
technical report and permitted-weight provenance to the final bundle; these files alone are not
every contest's complete deliverables. Load only your own trusted pickle files.

During the timed exercise, **close the repo and answer key**. Open a fresh notebook outside the repo,
give yourself only the pack's data directory and task card, and type the cells. The installed repo
must not become a hidden dependency. Keep `grading/answers.csv` closed until delivery ends.

## Use the 2k assistant as a function helper

You are the integrator. Keep the current task card, variable names, shapes and next function in
your notebook. Ask for **one small unit** with a fixed interface; test it immediately; then start a
fresh assistant conversation for the next unit. Carry only the contract needed for that unit.
Do not repeatedly paste the notebook or ask the model to remember earlier code.

Budget conservatively until you measure the actual interface. If 2k is shared between prompt and
generation, try roughly **300–500 tokens of input and a 20–35-line answer** initially, leaving room
for output and overhead. Line count is not token count. Check whether reasoning and chat history
consume the allowance; if output truncates, halve the unit. Asking for code only may shorten the
visible answer but does not guarantee the reasoning model fits. Use an output limit if the supplied
interface exposes one. Time a few calls in practice: a slow or unreliable call may cost more than
writing the known function yourself.

Use a prompt card like this, adjusted to the actual task:

```text
Write one Python function only; no tutorial or alternative implementations.
Installed: pandas, NumPy. No internet, project imports or extra dependencies.
Signature: align(sample, test_ids, pred) -> pandas.DataFrame
sample has exactly id,label. IDs are nonempty unique strings, possibly '001'/'NA'.
test_ids is a Series in prediction order; pred is a 1D array of string labels.
Require identical ID sets and one prediction per ID; otherwise raise ValueError.
Return a copy of sample with labels aligned by ID, preserving its exact row/column order.
No file writes. Aim for <=25 lines. Include necessary imports.
```

Call sequence for an uncertain data format:

1. You write inspection and the task card. Ask for a **single parser or dataset class** with stated
   input/output types. Run it on two examples and inspect values, not just shapes.
2. Ask for **one model or preprocessing constructor** using installed packages and no downloads.
   Keep label encoding, dimensions, and feature order explicit.
3. Ask for **one training or scoring function**, with exact logits/target shapes, devices and metric
   semantics. A scorer must implement the provided definition; the assistant must not invent it.
4. Ask for **one serializer or targeted patch**. Test shuffled IDs, malformed inputs and a reload.

For a CV training unit, constrain it as tightly as the alignment function:

```text
Write train_epoch(model, loader, optimizer, device) -> float, with imports.
PyTorch is installed. No project imports, downloads, AMP or scheduler.
loader yields float32 images [B,3,H,W] and int64 class indices [B].
model returns raw logits [B,K]. Use cross_entropy; zero_grad, backward, step.
Call model.train(), move both tensors to device. Return sample-weighted mean loss.
Fail clearly on an empty loader. No validation, saving or prediction in this function.
Return only the function, aiming for <=25 lines.
```

Test it on one batch: finite loss, correct shapes, gradients and changed parameters. Then request
evaluation in a separate call, explicitly requiring `eval()`, inference mode, no shuffled output,
and concatenated predictions. You retain the scorer and checkpoint-selection logic in your notebook.

Many familiar text baselines need no assistant calls. Spend the assistant's limited context on the
part you cannot safely reconstruct, especially a novel parser/decoder. Never paste held-out answers
to choose a model or tune thresholds. Treat generated code as an unverified candidate.

For debugging, send the exception, the relevant function (or 10–20 lines), and a tiny input example.
Request a replacement for that function, preserving its signature. If the assistant proposes a new
dependency or a large rewrite, narrow the request. After two unsuccessful attempts at one unit,
shrink the failing example and switch to the simpler working route. Avoid spending your first hour
in an assistant repair loop.

## Tune without losing the baseline

Freeze the first working model, folds, output and score. Record each candidate in a new run directory:

```text
run | exact score / direction | one change | fit + inference time | error evidence | keep/reject
```

For text, compare word bigrams with character features, `C`, or balanced class weights **one at a
time**, with no blanket assumption that balancing helps macro F1. For tables, diagnose missingness,
unseen categories and leakage before trying another estimator. For CV, verify tiny-subset fitting
and inspect mistakes before adding augmentation or a larger model. Tune a threshold only for an
appropriate score/output and only on validation data; probability metrics require scores, not
hard labels. Spend time where estimated **contest points per total minute** is highest.

Before final inference, reload the selected checkpoint and its exact preprocessing/label order.
An all-training refit, if legal and justified, is a new model; its score is not the old held-out score.
Reserve measured inference and upload time plus a safety margin. Preserve the original working
candidate until the replacement passes every delivery check.

## Closed-book drill protocol

Use the [daily program](15_DAY_PREPARATION.md) for task coverage. Every day starts with 15 minutes of
retrieval: write yesterday's sequence or weakest function without opening notes. Compare only
after attempting it; fix the mistake, close the answer key, and retry the next day.

| Stage | Materials during timed work | Pass evidence |
|---|---|---|
| Days 1–3 | One untimed answer-key study, then blank notebook; short assistant calls allowed. | Text and table baselines independently reconstructed, correct split and CSV; record lookup/call counts. |
| Days 4–7 | No repo, notes or internet; assistant only for one function per call. | Text target 30–40 min; image target 60 min on a small prepared pack, including score and reload. |
| Days 8–11 | Closed book; one unfamiliar schema/metric; at least one assistant-free repeat. | Correct adapter, hand-checked metric, shuffled IDs and restarted inference. |
| Day 12 | Matching baseline/advanced study; repeat a weak stage without notes. | Correct gallery/group contract, score and checked output. |
| Day 13 | First half of the two-task mock, **assistant disabled**. | Complete baselines before tuning; report every external hint needed. |
| Day 14 | Finish the rehearsal with actual permitted tools; force a short/truncated assistant response. | Working fallback without the failed call; delivery reserve maintained. |
| Day 15 | Fresh 45–60-minute **assistant-free** reconstruction, then team review. | Another person checks score/IDs/reload; you explain each stage without the answer key. |

Use a new pack/schema or seed for repeats; do not rehearse only the same toy rows. Synthetic packs
test mechanics, not contest quality. Real practice data must follow next. Judge improvement by
time to **correct score + checked output + reload**, not typing speed or notebook length.

For team transfer, teach this reconstruction sequence before sharing the larger kit. Have each
teammate rebuild one family and explain one failure (metric mismatch, leakage, label order or
wrong ID alignment). Assign primary and backup owners using demonstrated execution. Being able to
run your prepared notebook is a separate skill from being able to replace it when access is denied.
