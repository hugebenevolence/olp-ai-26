# Choose a notebook by its required output

[Guide index](README.md) · [First practice session](PRACTICE_QUICKSTART.md) · [Implementation status](IMPLEMENTATION_STATUS.md)

Use the cheapest complete route first. Every row below states its **default practice contract**;
replace it with the released task's data columns, permissions, evaluator, and serialization.
The official metric always overrides the template default.

## Primary preparation routes

| Output | Notebook | Inputs and default output | Metric/prediction notes |
|---|---|---|---|
| Table class/value | `tabular_template.ipynb` | train/test/sample CSV; `id,label` by default | Explicit classification/regression and label/probability output; macro F1, AUROC, log loss, RMSE, or MAE |
| Text class | `nlp_classification_template.ipynb` | `id,text,label`; sample `id,label` | Word+character TF-IDF, configurable weighting/C, saved CPU pipeline |
| Image class | `cv_classification_template.ipynb` | `id,image,label`; sample `id,label` | Shared trainer; validate identical TTA before test inference |
| Binary semantic mask | `semantic_segmentation_template.ipynb` | `id,image,mask`; sample `id,rle` | Original image resolution; column-major one-indexed RLE; explicit foreground values and global/mean-image Dice |

Text and tabular notebooks have full synthetic CPU execution tests. Other notebooks have syntax
checks and targeted adapter tests; complete live GPU execution remains a preparation exercise.

## Additional output families

| Output | Notebook | Default contract and remaining task-specific choice |
|---|---|---|
| Transformer text class | `transformer_text_classification_template.ipynb` | Same class CSV; permitted local model/tokenizer directory required |
| Video/activity class | `video_classification_template.ipynb` | `id,video,label`; record frame count; group by speaker/person/source video |
| Class with auxiliary training mask | `classification_segmentation_template.ipynb` | `id,image,label,mask`; submits only image labels, not masks |
| Named entities/token tags | `token_classification_template.ipynb` | `id,tokens,ner_tags` with list cells; output JSON list of integer tags or configured tag names |
| Seq2seq text | `seq2seq_template.ipynb` | `id,source,target`; sample `id,target`; same prefix/decoding at validation and test |
| Caption text | `image_captioning_template.ipynb` | `id,image,caption`; sample `id,caption`; stage local image processor and tokenizer |
| Query/document ranks | `retrieval_template.ipynb` | documents, labeled queries, test_queries; sample `query_id,rank,document_id`; align composite query/rank IDs |
| Bounding boxes | `object_detection_template.ipynb` | one training row per box, with `id,image,label_index,xmin,ymin,xmax,ymax`; test/sample one row per image; output JSON in `predictions` |
| Instance masks | `instance_segmentation_template.ipynb` | one training row per instance, with `id,image,label_index,mask`; JSON boxes/scores plus original-size RLE per instance |

Detection and instance templates save the best evaluated checkpoint and produce full test output.
Their scorer is named **practice integral AP50**, with exclusive xyxy coordinates, stable confidence
ranking, and one match per ground truth. It omits crowd/difficult/area/maxDet rules and COCO's
101-point averaging. Replace `EVALUATOR` and `METRIC_NAME` with the exact official implementation
before contest optimization. See the [VOC development-kit description](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2010/htmldoc/)
for the all-points AP convention; this helper is not a full VOC devkit replica.

For NER, entity F1 is not token accuracy. Training fails on truncated words so labeled chunking is
an explicit decision; inference uses overflow windows to retain one prediction per original word.
For captioning, CIDEr or multiple references require the official metric adapter rather than a
single-reference BLEU substitution.

## Specialized case studies and drills

- `image_anomaly_detection_template.ipynb`: specific normal-only Task 2, six categories, frozen
  memory/calibration, output `sample_id,category,label`. See its dedicated runbook before use.
- `image_anomaly_detection_dino_augmented_template.ipynb`: advanced Task 2 modes and frozen bundles.
  Stage each selected backbone checkpoint; old task permissions are not generic permissions.
- `adversarial_robustness_template.ipynb`: held-out robustness drill using the CV baseline's saved
  model and fold. It produces an evaluation report, not a universal attack submission.

## Important boundaries

A semantic mask merges touching objects of one class; an instance output separates them. An
auxiliary segmentation head still submits classes. Multilabel image classification requires BCE,
sigmoid decoding, class-specific thresholds, and the official output mapping; it is not a switch
that turns the single-label notebook into a completed multilabel route.

Audio, OCR, keypoints, depth, super-resolution, extractive QA, multiclass masks, unusual generation,
and custom attack tasks need additional adapters. Reuse the core inspection/split/report/submission
layer while implementing the exact task contract. Do not replace an absent official metric with
an approximately similar one.
