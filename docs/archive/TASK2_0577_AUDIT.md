# Task 2: 0.577 baseline audit and controlled improvement plan

This audit uses the executed outputs in `image_anomaly_detection_template.ipynb` supplied after the
public submission. The reported public score is `0.577`. Synthetic proxy accuracy is not treated
as hidden-label evidence.

## Supplied v2 log: 0.590

The later executed `image_anomaly_detection_template (1).ipynb` is still the original
single-backbone implementation: one `wide_resnet50_2`, 256-pixel input, identity-only normal
memory, 4,096 patches, top-3 scoring, and CutPaste-only calibration. It does **not** execute the
new category-specific models or augmentation policies.

Its public output predicted 33 anomalies out of 480, distributed `3, 10, 1, 3, 7, 9` across
categories 01-06. The user reports an official score of `0.590`. Numerically, 0.590 is higher than
the earlier reported 0.577; if it is considered worse, the comparison target must be a different
unrecorded run. Either way, the v2 result cannot diagnose the newer augmentation design because
that code was absent from the executed notebook.

## Reproduction gap table

| Component | Executed 0.577 run | Evidence | Gap | Revision type |
|---|---|---|---|---|
| Encoder | One `wide_resnet50_2` for all six categories | Executed configuration cell | Categories have different scale, texture, and layout | Pragmatic category specialization |
| Normal memory | Identity images only | Executed configuration and memory-building cells | Legitimate orientation or mild lighting variation can appear anomalous | Limited category-specific normal invariance |
| Synthetic positives | CutPaste only | Executed calibration cell | One corruption family does not represent mixed/ghosted content, short dark curves, or thin white lines | Evidence-driven positive-evidence extension |
| Threshold | Synthetic balanced-accuracy optimum, with normal q99 reported but not selected | Executed calibration cell | Proxy balanced accuracy of 0.966-1.000 did not transfer to the 0.577 public result | Separate threshold-policy experiment |
| Public predictions | 39 anomalies out of 480: `4, 9, 2, 6, 8, 10` by category | Executed public-inference output | Only 8.1% of samples were positive; the baseline may be too conservative | Test with aggregate score only |
| Private protocol | Frozen model, memory, and thresholds | Notebook bundle/reload cells | No methodological gap | Preserve unchanged |

Executed calibration evidence from the submitted run:

| Category | Synthetic threshold | Normal q99 | Proxy balanced accuracy | Predicted anomalies / 80 |
|---|---:|---:|---:|---:|
| category_01 | 0.845497 | 0.891015 | 0.966165 | 4 |
| category_02 | 0.728061 | 0.722405 | 0.984962 | 9 |
| category_03 | 0.793006 | 0.764922 | 1.000000 | 2 |
| category_04 | 0.757215 | 0.727953 | 0.988636 | 6 |
| category_05 | 0.805764 | 0.795583 | 0.973485 | 8 |
| category_06 | 0.717555 | 0.653850 | 0.988095 | 10 |

The large category-06 threshold gap is a concrete reason to test a more sensitive policy, but it
does not prove the public labels or the optimal threshold.

## Aggregate acquisition inspection

Train and public images were inspected only at category/acquisition level. No public sample was
manually labeled. Median train/public brightness, contrast, saturation, and aspect ratio were
close for every category, so there is no strong acquisition-shift justification for aggressive
color augmentation.

| Category | Stable visual structure | Normal-memory invariance to test |
|---|---|---|
| category_01 | Board appears in two natural 180-degree orientations | 180-degree rotation and mild contrast |
| category_02 | Small PCB with fine components and stable orientation | Mild brightness/contrast only |
| category_03 | Many capsules with variable layout/orientation | Horizontal/vertical flips and 90-degree rotations |
| category_04 | Fixed 2-by-2 layout with subtle surface details | Mild brightness/contrast; preserve geometry |
| category_05 | Four pieces with variable orientation/layout | Horizontal/vertical flips and 90-degree rotations |
| category_06 | One object on black background with mild color variation | Small brightness, contrast, and warm/cool changes |

Follow-up inspection corrected blur's role: mild blur is legitimate normal variation, so
`blur_mild` now expands the normal memory bank. Low-opacity CutMix, short internal dark curves, and
short thin white lines are synthetic positives. They train a small auxiliary evidence head whose
clamped output can only increase the open-set PatchCore score. Absence of these known patterns adds
zero and cannot make another kind of anomaly look more normal. Held-out normal scores, rather than
synthetic examples, set the final threshold. These remain configurable hypotheses, not manually
assigned public labels.

## Controlled public experiments

Use the presets in this order:

1. `baseline_0577`: exact architecture, normal-memory, and CutPaste calibration design from the
   submitted run. Keep the recorded 0.577 result; do not resubmit it.
2. `category_models_only`: change category backbone/resolution/memory capacity while preserving
   identity-only memory and CutPaste calibration. This isolates model specialization.
3. `category_augmented`: keep the specialized models, add limited normal invariance and the
   one-way learned positive evidence, then use the stated per-category normal quantiles.
4. Only for the better preset, compare the generated global threshold scales.

Do not compare two runs as a model ablation if model, normal augmentation, synthetic calibration,
and threshold policy changed simultaneously. The presets exist to keep those comparison breaks
visible.

## Verification boundary

The implementation can be syntax-, unit-, notebook-build-, and model-forward-tested locally. A
new public score requires Colab training plus an official submission. Until that is run, the
category-specialized configuration is a justified candidate, not a claimed improvement over
0.577.
