# 15-day practical preparation for OlpAI'26

[Guide index](README.md) · [First practical session](PRACTICE_QUICKSTART.md) ·
[Current implementation status](IMPLEMENTATION_STATUS.md)

Prepared 14 October 2026 for a fastai Practical Deep Learning learner, **3–4 focused hours a day**,
preparing personally first and then distributing working solutions to a team.

**Target: turn an unfamiliar problem into a legal, correctly evaluated, reproducible baseline
within 45–60 minutes; improve it using evidence; deliver the complete solution on time.**
Fifteen days can materially improve competition execution. It cannot guarantee a medal or mastery
of every task family. The largest immediate gains should come from trustworthy validation,
debugging, economical experiments, and dependable submission workflows.

Read the [current implementation status](IMPLEMENTATION_STATUS.md) and
[historical audit](READINESS_AUDIT_2026-10-14.md) before trusting a template's title.
This document is a training plan; implemented repairs do not imply its exercises are complete.

## Contest facts that shape the plan

As checked on 14 October 2026, OlpAI'26 is a university contest for teams of 2–3 students, with
ML, CV, and NLP scope and six-hour rounds. Scores are summed across tasks; task-level scoring and
submission limits are specified separately. Deliverables include code, weights, and a short
technical report for the highest-scoring submission. External real training data, test-set training,
and manual test labeling are forbidden; supplied-data augmentation and synthetic data are allowed.
The final uses two organizer PCs and one submission account, and prohibits bringing documents
and electronic devices. These facts favor broad baseline coverage and independent reconstruction
practice. [Official rules](https://www.olp.vn/olympic-ai-sinh-vi%C3%AAn/quy-ch%E1%BA%BF-k%E1%BA%BF-ho%E1%BA%A1ch).

The regional contest is listed for **1 November 2026**, following practice on **31 October**;
the final is listed for **8 December**, within the 7–9 December event. The schedule contains
different regional start times, so follow the organizer's latest account/venue notice.
[Official schedule](https://www.olp.vn/olympic-ai-sinh-vi%C3%AAn/ch%C6%B0%C6%A1ng-tr%C3%ACnh).

The environment page still shows an announcement placeholder and describes **2025** FPT Smart
Cloud/VNOJ use. This does not establish the 2026 runtime. It links IOAI 2026 tasks for practice.
[Official environment and past-paper page](https://www.olp.vn/olympic-ai-sinh-vi%C3%AAn/m%C3%B4i-tr%C6%B0%E1%BB%9Dng-%C4%91%E1%BB%81-thi).

Have your coach clarify permitted prepared-code access, documentation/internet access, AI-assistant
use, installed packages, hardware quotas, model allowlists, exact deliverables, and private-test
procedure. This is a checklist for your team; no message has been sent. Do not assume this repo,
Drive, personal Colab Pro, or downloaded weights will be accessible in the room.

**Training priorities below are recommendations, not official topic weights or predictions of
the number of tasks.** ML can include more than tabular learning. Multimodal practice is useful
for adapting across inputs; it is not a separately confirmed OlpAI'26 section.

## Time budget and working method

Use 3.5 hours on Days 1–12, three hours on each of Days 13–14, and three hours on Day 15:
**51 focused hours**. If starting 14 October, Day 15 falls on 28 October, leaving 29–30 October
for logistics, retrieval practice, and rest before the listed practice session.

An ordinary day has four blocks:

| Minutes | Work |
|---|---|
| 0–15 | Reconstruct yesterday's key steps without notes; choose one measurable objective. |
| 15–105 | Implement or run the smallest complete baseline. |
| 105–180 | Analyze errors and run at most two purposeful changes. |
| 180–210 | Reload or infer, validate artifacts, record the result and next hypothesis. |

With three hours, shorten the experiment block. With four, use the extra 30 minutes for a cold
rerun or a second split. Avoid spending that extra time searching for more architectures.
Reading is capped at roughly 20–30 minutes daily and must resolve a concrete coding question.
You have already watched fastai; another full course has lower value than completing these drills.

Before timed practice, download a small practice dataset and make an immutable practice pack:
training data, a sample submission, and a held-out test manifest. Keep test labels out of the
working notebook. Use a training-derived validation split for decisions; open the held-out labels
only once after the exercise. Have a teammate create the pack when possible. Solo label separation
is useful discipline but is weaker than a genuinely unseen task.

Practice datasets are for preparation only. They must never be mixed into official contest training.

## Daily program

| Day | Focus and concrete exercise | End-of-day deliverable and pass condition |
|---|---|---|
| 1 | Establish your starting level. Run the text baseline on a prepared small pack; inspect IDs, labels, split, metric, and output. Perform one clean Colab bootstrap. | A timed baseline, validated CSV, fitted pipeline, configuration, and one-page report. Record all blockers and time to first valid output. Aim for 60 minutes; missing it is diagnostic. |
| 2 | Validation and metrics. Hand-score tiny F1/Dice fixtures, test shuffled IDs, and compare random versus group splits on a deliberately duplicated practice fixture. | Correct metric fixtures, zero group overlap, and a prediction-to-ID assertion. Explain why global Dice and per-image Dice can differ. |
| 3 | Cover tabular ML. Build a sklearn preprocessing pipeline and compare a dummy, linear model, and histogram gradient boosting on Adult; use built-in diabetes for a short regression drill. | One reusable tabular workflow with train-only preprocessing, saved pipeline, classification/regression metric selection, and validated output. Stay within a 30-minute first baseline. |
| 4 | Vietnamese text classification. Run word+character TF-IDF on UIT-VSFC sentiment; compare balanced versus unweighted classification and inspect minority-class errors. | Same-split comparison of at most three variants, per-class F1/confusion matrix, 20 annotated validation errors, and a reloadable best pipeline. |
| 5 | Transformer decision drill. If a checkpoint is permitted for the practice task, stage it locally, compare short maximum sequence lengths and one learning-rate change. Otherwise strengthen the sparse fallback. | Record macro F1, truncation rate, epoch time, memory, inference time, and permitted-weight provenance. Decide whether GPU use is justified. |
| 6 | Image classification from scratch. Adapt the CV template to a CIFAR-10 subset or prepared image pack. Overfit 32 examples before the real run. | Tiny-subset fitting works; real validation beats the declared dummy baseline; label mapping and test IDs are correct; checkpoint reload reproduces predictions. |
| 7 | Improve CV using errors. Inspect 30 validation mistakes and compare resolution, conservative augmentation, and one loss/learning-rate change. | A controlled experiment table. Validate any TTA with the same prediction path used at test time; keep it only if measured gain and runtime justify it. |
| 8 | Semantic segmentation. Use a small Oxford Pets mask pack, define foreground/boundary handling, and train a small model. | Visual image/mask overlays, exact selected Dice/IoU convention, and correct RLE round trips. Produce and validate the complete mask submission. |
| 9 | Normal-only anomaly detection. Use an authorized archived Task 2 pack, or normal training images plus separately held-out real defects. Fit memory on training normals only. | A frozen encoder/memory/threshold bundle, normal false-positive rates per category, and a clear distinction between synthetic proxy and real-defect validation. |
| 10 | Unfamiliar-task adaptation. Choose one linked IOAI task, preferably Ghost of the Machine, and read its statement, baseline, and grader. Alternatively do retrieval or NER. | Map the task's input/output and evaluator within 20 minutes. Make a legal scored baseline within 60 minutes, or record the exact missing dependency and a fallback. |
| 11 | Recovery and delivery. Restart the runtime; reload a saved solution and predict without training. Simulate a smaller batch size, shuffled manifest, missing path, and limited disk. | Reproduce predictions after restart, recover within 15 minutes excluding initial data transfer, and verify a complete durable solution bundle. |
| 12 | Compressed multi-task mock. Solve one text/tabular task and one image task in 180 minutes; use 30 minutes for review. | Baselines for both before tuning, a shared experiment ledger, valid artifacts, and a postmortem naming the biggest avoidable time loss. |
| 13 | Six-hour rehearsal, first half. Start from fresh packs and the permitted preparation materials; triage, inspect, validate, and establish baselines. | Usable outputs for every mock task and a written compute/submission budget. Prefer running Days 13–14 as one continuous six-hour session. |
| 14 | Six-hour rehearsal, second half. Improve selectively, release the mock private phase late, and finish delivery. | No private-phase fitting where prohibited; complete code/weights/reports/output; all selected prediction paths validated. Last 30 minutes reserved for delivery. |
| 15 | Team transfer and memory test. Let a teammate run one bundle without your verbal instructions; rebuild a minimal baseline from a blank permitted notebook. | Teammate reproduces output, roles and backup owners are assigned, and remaining failures are explicit. Spend the last hour reviewing mistakes and first-hour procedure. |

If a continuous six-hour session is impossible, Days 13–14 can be two three-hour blocks with frozen
state and no off-clock experiments. That tests workflow and persistence, **not six-hour endurance
or realistic deadline pressure**. Budget one continuous team rehearsal later if feasible.

## Details that make the exercises productive

### Days 1–2: measure execution before model quality

Use the audit's P0 findings as failure drills. Reconcile generated-notebook edits through canonical
sources before regenerating; generic practice copies must explicitly deny unauthorized weights.
Keep one fixed baseline configuration and timestamp stages: data usable, first fit, validation,
test prediction, submission accepted locally, and artifact saved.

For metric fixtures, specify label set, averaging, ignored values, ties, threshold, and empty cases.
Check the organizer's implementation when available. Test IDs with a permutation that preserves
row count; the validator alone will not detect a misattached prediction. For groups, assert train
and validation IDs are disjoint. Fit imputers, scalers, feature selectors, and vectorizers only on
the training fold; sklearn pipelines help enforce that separation.
[Sklearn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

Use fixed folds when comparing candidates. Changing a model and split simultaneously destroys
the comparison. If a split contains few examples of a class, report that limitation instead of
claiming a tiny score difference is decisive.

### Day 3: add a small ML route, not a framework

Start with `ColumnTransformer` + imputation + one-hot encoding + logistic regression. Compare with
`HistGradientBoostingClassifier` using an appropriate numeric/categorical representation. For a
regression drill, compare dummy mean, Ridge, and `HistGradientBoostingRegressor`. Scaling is useful
for the linear path; it is not automatically required by the tree path. Never feed an enormous
dense one-hot matrix to a model without checking memory.

Try only a shallow versus moderately deep tree setting, a learning-rate change, or a carefully
motivated feature. Save preprocessing with the estimator. CatBoost/LightGBM are optional follow-up
choices only if supported by the task environment; adding them is not necessary to complete this
plan. [Sklearn gradient boosting reference](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting).

### Days 4–5: earn the GPU decision

The existing TF-IDF class already supplies word bigrams and character 3–5 grams. First compare its
class weighting with no weighting on the same split. Inspect missing values, contradictory labels,
duplicate texts, and normalization before tuning `C` or adding a larger model. Preserve Vietnamese
diacritics and negation; removing them needs evidence. Character features are a useful experiment
for spelling variation, but no configuration is universally best.
[Text-feature reference](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction).

For transformer practice, stage the exact allowed model and tokenizer, inspect sequence-length
percentiles, and choose a small comparison such as lengths 128 versus 256 and learning rates
`2e-5` versus `5e-5` for fine-tuning. These are starting experiments, not guaranteed optimal values.
Dynamic padding can reduce wasted work. A randomly initialized transformer is not a cheap
replacement for forbidden pretrained weights. Keep the sparse CPU route available.

### Days 6–8: debug the data-model contract

Inspect one batch's shapes, value ranges, channel order, labels, and decoded output before training.
When overfitting 32 examples, temporarily disable augmentation and unnecessary regularization.
If the model cannot fit that subset, investigate labels, normalization, loss, gradient flow, and
decoding before increasing model size. Restore proper validation and augmentation afterward.

For CIFAR-sized images, do not blindly upscale to 224 and expect efficiency; either use a small
appropriate model/adapter or measure the cost of the template's larger-resolution path. For
pretrained practice on other images, use the checkpoint's documented normalization and legal
weight policy. Unfreeze/fine-tune only after a simple baseline works.

For masks, resize labels with nearest-neighbor interpolation, synchronize image/mask geometry,
and inspect boundary conventions. Evaluate at the required output resolution. Compare a
conservative threshold grid on validation only and freeze the selected threshold. Do not assume
Dice+BCE training loss means your evaluation reproduces official Dice.

### Day 9: learn from the repository's strongest case study

Read `TASK2_0577_AUDIT.md` and the runbook's contrast between high synthetic proxy scores and
weaker reported public scores. Reproduce a cheap existing configuration before stacking synthetic
heads, patch fusion, memory augmentation, or TTA. A normal-only training set cannot honestly
support ordinary supervised normal/anomaly classification without an authorized positive source.

Check normal false positives on held-out training normals. If actual anomaly labels are available
in the practice pack, evaluate on a separate real-defect validation set. If they are absent,
state that anomaly sensitivity is unmeasured. A synthetic score or predicted anomaly count does
not prove hidden-test accuracy. Freeze training-derived memory, calibration, and thresholds before
any private phase that forbids refitting. Old task permission for DINO weights is not general permission.

### Day 10: practice mapping an unfamiliar task

Write five lines before touching the architecture: input unit, prediction unit, allowed resources,
scoring function, and artifact format. Read the starter/grader and identify the actual callable
contract. Finish one scored path rather than browsing five task solutions.

The organizer-linked [IOAI 2026 task collection](https://github.com/IOAI-official/IOAI-2026/tree/main/Individual-Contest)
includes ordering, instruction interpretation, semantic search, adversarial disagreement, text-edit
localization, and generation. It illustrates why learning input/output adaptation matters. Its rules,
hardware, and inference time limits are separate from OlpAI'26. If a task's protected runtime is
unavailable, use an accessible task instead of spending the whole day reconstructing it.

For retrieval, verify whether MRR, Recall@K, or another metric is required and how multiple relevant
documents are handled. For NER, inspect word-to-subword alignment and evaluate entities rather
than ordinary token accuracy. The existing NER template needs full inference and serialization
before it counts as a complete contest route.

### Days 11–15: transfer an executable solution

Recovery means reconstructing inference from a saved bundle, not merely seeing a checkpoint file.
The current trainer saves model/optimizer state but does not establish an exact training-resume
path with scheduler, scaler, and random state. Test whichever promise you actually need: identical
inference, or full training continuation. Do not claim the latter based on the former.

Include model, preprocessing/tokenizer, label mapping, metric configuration, thresholds, fold IDs,
seed, permitted-weight provenance, notebook/source, report, and submission. Use a unique run
directory; the currently shared output directory can overwrite another task's `best.pt` or CSV.
Verify the whole bundle survives persistence; suffix-based synchronization is not an inventory.
[PyTorch save/reload guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

Colab practice must read media under `/content/olp_runtime`, preserve the CUDA-matched stack, and
select precision/batch size from `gpu_report()`. Colab resource allocation is variable.
[Colab resource guidance](https://research.google.com/colaboratory/faq.html).
On an organizer platform, adapt paths/storage and use its permitted packages; do not assume Drive
mounting or `/content` exists there.

## Highest-return competition habits

| Order | Action | Reason to prioritize it |
|---|---|---|
| 1 | Establish legal inputs, exact evaluator, correct split, and output contract. | A high local score is worthless if it measures the wrong thing or uses forbidden resources. |
| 2 | Produce a complete cheap baseline and preserve its bundle. | Gives a fallback and an actual benchmark for every experiment. |
| 3 | Fit a tiny subset and inspect decoded outputs. | Quickly exposes data, loss, and label bugs. |
| 4 | Analyze 20–30 validation errors by type/class/group. | Turns model changes into testable hypotheses. |
| 5 | Measure epoch time, peak memory, and full inference time. | Enables useful compute allocation and deadline decisions. |
| 6 | Tune the likely bottleneck: resolution, truncation, loss, weighting, or threshold. | Often cheaper than changing the entire model. |
| 7 | Change one conceptual axis per experiment. | Makes success and failure interpretable. |
| 8 | Confirm a consequential small gain on a second suitable split or seed when affordable. | Reduces the chance of selecting noise. |
| 9 | Try TTA or a small ensemble only with same-split evidence and enough inference budget. | These add cost and can reduce score. |
| 10 | Stop experimenting early enough to reproduce and deliver the chosen solution. | Prevents deadline failure and mismatched weights/reports. |

Use the official scoring scale when allocating work. An extra 0.01 raw F1 on one task is not
necessarily worth the same total-contest points as 0.01 on another. Once task scoring is known,
estimate gain in **contest points per total minute**, including implementation, training, validation,
inference, and failure risk. Prefer improving weak tasks over polishing a strong one when the
expected total-point return is higher.

Avoid broad hyperparameter searches, training large models from scratch without a reason,
complex stacking, test pseudo-labeling, unapproved external data, and copying a public solution
without understanding its grader and weight policy. Broad architectures remain optional tools;
none replaces a trusted evaluator.

## Six-hour operating procedure

This is a template for task count and scoring revealed on the day; it does not assume two tasks.

| Elapsed time | Team outcome |
|---|---|
| 0:00–0:15 | Read every statement and scoring rule. Record permissions, time limits, output format, and submission budget. |
| 0:15–0:30 | Assign one owner per task and a delivery reviewer. Inspect data; freeze a justified validation scheme. |
| 0:30–1:00 | Get a cheap valid baseline for each feasible task. If a complex task cannot train that fast, produce a legal starter-output route and record the remaining work. |
| 1:00–3:30 | Run short, evidence-driven experiments. Keep CPU tasks progressing while GPU work runs. |
| 3:30–4:30 | Improve the best bottleneck, compare candidates fairly, and measure complete inference. |
| 4:30–5:00 | Select and freeze the solution bundle according to the task's private-test rules. |
| 5:00–5:30 | Run final inference and inspect files, evaluator outputs, reports, and reload instructions. |
| 5:30–6:00 | Delivery only: validate/upload and confirm acceptance. Start no new training. |

Use a larger delivery reserve if measured inference or upload requires it. Start final inference
before the deadline minus measured inference time, transfer time, and a safety margin. Freeze or
submission deadlines stated by the task override this generic timetable.

Do not assume two PCs mean two available GPUs. Assign one compute coordinator; avoid overlapping
memory-heavy jobs unless quotas and memory support it. One teammate can run CPU baselines and
evaluation while another trains. Reserve the last available submission slots for the selected
solution and a necessary correction; if a task permits only one submission, do local validation
first rather than following a generic early-submit habit.

Tie every submission ID to its immutable run bundle. The official rules reference the highest-scoring
submission, so preserve its matching source/weights/report even if a later experiment loses.

## Personal preparation and team handoff

During Days 1–11, build your own competence and save short task recipes. Tell the team early which
families you are covering so their preparation can complement yours; distributing working recipes
later is fine, but leaving the first team execution until contest day is risky.

Suggested three-person roles, adapted to actual strengths and released tasks:

- **You:** shared validation, metric and submission review; primary owner of your strongest family.
- **Teammate A:** primary CV owner and inference/runtime measurement.
- **Teammate B:** primary NLP/ML owner and packaging/report backup.

Each task needs a primary and backup owner. If you have two people, combine review/report work
with the lighter CPU route. On Day 15, have the backup run the notebook from a fresh runtime
using only its written recipe. Log every question they need answered; fix the recipe, then retry.

Keep preparation resources for learning and handoff. In a restricted room, reproduce the necessary
workflow from memory or organizer-provided starters unless access to prepared material is permitted.

## Readiness gates after Day 15

These are operational goals, not medal predictions or universal score thresholds.

- [ ] Explain legal weights/data and prediction/output requirements within 10 minutes of reading a task.
- [ ] Produce a complete baseline in 45–60 minutes on an unseen or independently prepared task pack.
- [ ] Match the exact selected evaluator on small fixtures and declare the split/averaging conventions.
- [ ] Assert no prohibited train/validation overlap and no fitting on the held-out/private phase.
- [ ] Fit a tiny sample, map labels correctly, and preserve prediction IDs through shuffled manifests.
- [ ] Justify at least one experiment from validation errors and retain only measured improvements.
- [ ] Benchmark full inference and deliver within a realistic deadline reserve.
- [ ] Reload the full bundle in a clean runtime without training or unauthorized downloads.
- [ ] Have a teammate reproduce the output and identify the matching submitted run/report.
- [ ] Complete the rehearsal; explicitly mark split-session endurance and target-runtime gaps if untested.

Missing a correctness, legal-policy, or delivery gate takes priority over squeezing out another
small validation gain. If the core gates pass, choose one specialization for the remaining days
and maintain the other families' cheap fallbacks. Use the official practice session to test actual
account access, quota, package inventory, evaluator, and upload mechanics.

## Daily experiment record

Copy this into a local Markdown/JSONL record for each meaningful run. Reports must describe what
actually ran, including failures and uncertainty.

```text
Day / task / run ID:
Dataset source and practice/contest permission:
Input/output contract and official metric:
Split IDs, groups/time, seed, and leakage checks:
Weights/model/tokenizer provenance and hashes:
Baseline score and elapsed time:
Error pattern and hypothesis:
One conceptual change:
Candidate score, per-class/group metrics, training and inference time:
Decision: keep / reject / uncertain, with reason:
Saved artifact inventory and submission ID:
Cold-reload result:
Next action:
```

## Small practice resource set

Use these selectively, following their access and licensing conditions; prepare packs before the
timer so download friction does not become the whole exercise.

- [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult): mixed numeric/categorical classification practice.
- [UIT-VSFC author's repository](https://github.com/kietnv/uit-vsfc): Vietnamese sentiment/topic practice;
  preserve its provided splits and inspect access instructions.
- [CIFAR-10](https://cave.cs.toronto.edu/kriz/cifar.html): small-image classification from scratch.
- [Oxford-IIIT Pet](https://robots.ox.ac.uk/~vgg/data/pets/): image/mask practice using a manageable subset.
- [Organizer environment and past-paper page](https://www.olp.vn/olympic-ai-sinh-vi%C3%AAn/m%C3%B4i-tr%C6%B0%E1%BB%9Dng-%C4%91%E1%BB%81-thi):
  select actual past statements where accessible; do not confuse VOAI school papers with university OlpAI rules.

Use the existing `COLAB_RUNBOOK.md`, `BASELINE_CATALOG.md`, `CUSTOMIZATION_GUIDE.md`, and
Task 2 audit as lookup references, not as extra reading assignments. After changing implementation,
follow `AGENTS.md`'s required checks; after changing canonical templates, regenerate `.ipynb` files.
