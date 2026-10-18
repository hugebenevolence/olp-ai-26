# 15-day practical preparation for OlpAI'26

[Guide index](README.md) · [Blank-notebook playbook](BLANK_NOTEBOOK_PLAYBOOK.md) ·
[Repo-assisted session](PRACTICE_QUICKSTART.md) ·
[Current implementation status](IMPLEMENTATION_STATUS.md)

Prepared 14 October 2026; paired curriculum revised 18 October. For a fastai learner,
**3–4 focused hours a day**,
preparing personally first and then distributing working solutions to a team.

**Target: turn an unfamiliar problem into a legal, correctly evaluated, reproducible baseline
within 45–60 minutes; improve it using evidence; deliver the complete solution on time.**
Fifteen days can materially improve competition execution. It cannot guarantee a medal or mastery
of every task family. The largest immediate gains should come from trustworthy validation,
debugging, economical experiments, and dependable submission workflows.

Start with the [seven paired problem exercises](../notebooks/learning/README.md): fill a baseline,
reveal solutions after attempting each block, and compare the explained advanced method on the
same data and folds. Use the [learning scorecard](LEARNING_SCORECARD.md) to record hints and
transfer. The following day retrieve a weak block without notes. The final assessments close the
repo, notes and internet; at least two disable the assistant completely. A permitted DeepSeek
assistant is used for one small function per fresh conversation within the stated 2k window.
The 14 learning notebooks are self-contained: all task code, metrics and training loops are
visible in their cells. Study them individually without a checkout, then close them for timed
reconstruction. Installed libraries and explicitly configured data/assets are still required.

Implemented notebooks and repairs do not imply you have completed the training. See the
[implementation status](IMPLEMENTATION_STATUS.md) for actual verification boundaries.

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

Use 3.5 hours on Days 1–12 and three hours on Days 13–15: **51 focused hours**.
Day numbers are training sessions, not a renewed contest schedule. If starting on 18 October and
the previously listed 1 November contest date holds, Day 15 would be contest day. Compress two
advanced study sessions and preserve baseline coverage, delivery and the mock; confirm actual
dates with the organizer. The contest facts above were checked on 14 October.

An ordinary session: 15 minutes reconstructing yesterday's weakest block, 60–90 minutes attempting
the day's baseline or advanced method, 45–60 minutes inspecting errors and one justified change,
then 30–45 minutes on reload, checked output and the scorecard. At three hours shorten experiments;
at four hours add a cold rerun or second suitable split. Do not spend the extra hour collecting
architectures. Reading should resolve a specific implementation or metric question.

Download and stage real practice data before the timed coding drill using the
[data guide](LEARNING_DATASETS.md). Keep held-out answers closed until delivery. A synthetic fixture
checks execution; a real pack tests data/model judgment. Neither grants additional contest data
permission. Every baseline is followed by an advanced **study** recipe; the
[evidence ledger](KAGGLE_METHOD_EVIDENCE.md) separates documented competition methods from the
reduced implementation and its unmeasured performance.

## Daily program

| Day | Notebook work | Concrete pass condition |
|---|---|---|
| 1 | Home Credit fill-in baseline. Inspect missingness/prevalence, train-only preprocessing, probability scoring and sample IDs. | First complete small-pack route in 45–60 minutes, saved pipeline, checked CSV and hint count. Set up the target runtime before the timed drill. |
| 2 | Home Credit advanced. Applicant aggregates and one LightGBM candidate; redo yesterday's weakest block without notes. | Same-fold AUC and total-runtime comparison. Explain which source features and resources were used; keep the baseline unless the evidence supports the candidate. |
| 3 | House Prices baseline and advanced on the small table. Learn log target, residual plots and inner-fold-safe stacking. | Hand-check the log-price metric, persist both candidates, and explain why preprocessing belongs inside each OOF estimator. Reduce scope if core correctness is still weak. |
| 4 | Jigsaw multilabel baseline. Inspect text lengths, duplicates, co-occurrence and rare-label support. | Six independent probability columns, duplicate-group split, defined AUC for every target, saved TF-IDF/classifiers and checked output. |
| 5 | Jigsaw BiGRU study. Reconstruct loss/decoding, compare the sparse candidate and measure truncation/runtime. | Distinguish random from permitted pretrained embeddings; save vocabulary with the own best checkpoint. If assets/time are unsuitable, strengthen the sparse route and retain its baseline. |
| 6 | Cassava scratch-CNN baseline. Inspect image/label examples and fit 16–32 examples on a separate debug model first. | Inspect a batch, verify loss/gradients, reinitialize, score a real split, save/reload and preserve class order. Target a first small-pack complete route within 60 minutes. |
| 7 | Cassava EfficientNet study. Use permitted local initialization only when available; otherwise study the scratch architecture honestly. | Accuracy selects the candidate; minority recall/macro F1 diagnose errors. Retain only a measured gain whose memory and inference time fit. |
| 8 | TGS Salt fill-in baseline. Inspect images/masks, empty tiles and foreground coverage. | Exact TGS boundary fixtures, paired geometry, original-size evaluation, RLE round trip and full submission. |
| 9 | TGS small-U-Net study. Compare skip connections, Dice/BCE and a short validation threshold grid. | Same-fold score/runtime table; threshold frozen before test; distinguish surrogate loss/Dice from the task score. |
| 10 | Global Wheat fill-in baseline. Inspect boxes/source distributions and split images rather than annotation rows. | Source-group holdout, correct variable-box batching, one validation-selected checkpoint and correctly scaled/serialized test boxes. |
| 11 | Global Wheat ResNet-FPN study plus recovery drill. Compare cost/confidence threshold and restart inference. | Correct evaluator conventions and every output ID; reload without retraining. This is a single-model component study, not the original winning ensemble. |
| 12 | Shopee baseline, then fusion study if permitted encoders/cache were staged beforehand. Reconstruct matching in batches. | Product-group holdout, per-query set F1 with self, train-only vocabulary, bounded similarity memory and honest feature-extraction cost. Cache-only inference cannot embed new private examples. |
| 13 | Three-hour first half of a two-task mock, no repository/notes/internet or assistant. Choose unfamiliar table/text and CV packs. | Complete baselines before tuning, fixed folds and experiment ledger. Record every external hint needed. Prefer combining Days 13–14 into a continuous six-hour rehearsal. |
| 14 | Three-hour second half: selective improvement and delivery. Use only the actual permitted tools; simulate a truncated/failed assistant response. | Preserve a fallback, freeze the chosen bundles, finish inference/checks and reserve the last 30 minutes for delivery. No off-clock experiments between split sessions. |
| 15 | Fresh 45–60-minute assistant-free reconstruction, then team transfer and mistake review. | Correct score, IDs and reload without hints; teammate can explain/reproduce the bundle and primary/backup owners are clear. |

If Days 13–14 must be split, freeze state between sessions. This checks persistence and workflow;
it does not reproduce six-hour endurance or continuous deadline pressure. Budget a continuous
team rehearsal later if feasible. Missing required local assets is a reason to keep a working
fallback and study the relevant component, not to download unapproved weights during a contest.

## What to retain from each family

| Family | Memorize through repeated implementation | Understand before copying the advanced method |
|---|---|---|
| Tables / classification | Pipeline, train-only fit, unknown categories, predict_proba, exact AUC and ID alignment. | Why applicant aggregates are available at inference; weighting may change ranking/calibration. |
| Regression | Positive target, exact score, target transform/inverse, residuals and pipeline reload. | OOF meta-features and inner-fold preprocessing; an ordinary random OOF stack does not automatically suit chronology. |
| Text | Vocabulary/TF-IDF, label support, duplicate groups, single-label versus multilabel loss/decoding. | Token length, pretrained vocabulary coverage, padding and whether GPU cost beats the sparse route. |
| Image classes | Dataset, shape/range, label map, train/eval modes, optimizer step and best checkpoint. | Legal initialization, normalization, resolution, conservative augmentation and actual error types. |
| Masks | Paired transforms, foreground/ignore policy, original size, threshold and RLE. | Skip connections and surrogate loss; empty-image and threshold conventions in the real evaluator. |
| Boxes | Prediction unit, xywh/xyxy, variable-length collate, background class, eval mode and serialization. | Source/domain holdout, small-object resolution and confidence/IoU conventions; COCO AP is not Wheat scoring. |
| Matching | Train-only text features, product-group split, batched similarity, self match and set F1. | Modality normalization, validation-only fusion selection and the difference between cached features and reusable encoders. |

## Tuning order and return on time

First establish legal inputs, the exact scorer, a justified split and correct output. Then fit a
tiny subset and inspect decoded predictions. Preserve the working candidate before tuning.
Analyze 20–30 validation mistakes and choose one likely bottleneck: missingness, text truncation,
minority support, image resolution, mask coverage, box scale or matching threshold. Change one
conceptual axis at a time. Keep extraction, training and full inference time in the ledger.

Compare with a suitable trivial reference: prior probabilities for classification, a constant
log-price predictor for regression, majority class for image labels, empty masks for segmentation,
no boxes for detection, and self-only matches for matching. A valid output with an uninformative
score is still a pipeline drill, not a competitive model. Synthetic fixture accuracy cannot show
real-dataset superiority.

Allocate by expected **contest points per total minute**, including implementation, GPU time,
inference, delivery and failure risk. Different task scoring scales are not directly comparable.
Confirm a consequential small gain on another appropriate split when affordable. Ensembles,
large searches and extra architectures follow evidence and a realistic inference budget. Never
train on test/pseudo-labels, add outside data or enable weights without exact task permission.

## Six-hour operating procedure

This schedule adapts to the task count and scoring released on the day.

| Elapsed | Outcome |
|---|---|
| 0:00–0:15 | Read all statements. Record allowed resources, scorer, output, freeze deadline and submission quota. |
| 0:15–0:30 | Assign task owners and reviewer; inspect data and freeze justified splits. |
| 0:30–1:00 | Establish a cheap complete baseline for each feasible task. For a difficult output, prove its legal serialization route and identify the remaining work. |
| 1:00–3:30 | Run short evidence-driven experiments; keep CPU work progressing while GPU jobs run. |
| 3:30–4:30 | Compare candidates fairly and measure complete inference. |
| 4:30–5:00 | Freeze selected code/weights/preprocessing/configuration according to task rules. |
| 5:00–5:30 | Run final inference; inspect files, scores and matching reports. |
| 5:30–6:00 | Delivery only: validate, upload and confirm acceptance. |

Increase the reserve when measured inference/transfer needs it. Task-specific deadlines override
this generic schedule. Respect submission quotas; do not spend the only slot on a generic dummy.
Keep the immutable source/weights/report corresponding to each submitted ID. Do not assume two
PCs mean two GPUs; coordinate memory-heavy jobs explicitly.

## Team transfer and final readiness gates

Prepare independently, but tell teammates early which families you are covering. Have a primary
and backup per task family. Suggested responsibilities: you own metric/submission review and your
strongest family; another teammate owns CV/runtime; another owns NLP/ML and delivery backup.
Adjust to demonstrated strengths and actual team size. No messages are sent by this plan.

Have the backup reconstruct a baseline and run a saved bundle with no verbal instructions. Log
questions and repair the recipe. Running a prepared notebook and replacing it when access is
unavailable are separate abilities.

- [ ] Explain the prediction unit, exact scorer, legal data/weights and output within ten minutes.
- [ ] Fill the baseline blocks and explain the model/loss/label mapping without reading the solution.
- [ ] Rebuild a complete small-pack route within 45–60 minutes with no saved code or assistant.
- [ ] Hand-check metric boundaries; assert no prohibited identity/group/time overlap.
- [ ] Use a short assistant call for a bounded gap and continue after its failure.
- [ ] Justify an improvement from validation errors and compare score versus total cost.
- [ ] Restore preprocessing, labels, thresholds and best weights in a fresh runtime.
- [ ] Deliver correctly ordered outputs and preserve matching code/report/bundle.
- [ ] Have a teammate explain/reproduce the route; mark actual target-runtime gaps honestly.

Missing correctness, permission or delivery gates take priority over marginal model gains. Keep
the [learning scorecard](LEARNING_SCORECARD.md), notebook error tables and comparison rows as
feedback for the next drill. Use the official practice session to verify actual packages, quotas,
model allowlists, evaluator, account access and upload mechanics. Local CPU checks do not certify
live Colab GPU execution or the organizer's platform.
