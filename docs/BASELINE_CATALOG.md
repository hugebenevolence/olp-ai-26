# Baseline and notebook catalog

Choose the row matching the **required output**, not the input modality alone. Run the cheapest
valid baseline first, reproduce the official metric, and create a valid submission before tuning.

| Problem/output | Start with | Main code | Typical official metric |
|---|---|---|---|
| Normal-only image anomaly label | `image_anomaly_detection_template.ipynb` | `cv.anomaly_detection` | macro category balanced accuracy |
| Single-label image class | `cv_classification_template.ipynb` | `cv.classification`, `cv.tta` | accuracy, macro F1 |
| Multi-label image tags | CV classification; set BCE loss and sigmoid thresholds | `classification_loss(multilabel=True)` | micro/macro F1, mAP |
| Video/activity class | `video_classification_template.ipynb` | `cv.video_classification` | accuracy, macro F1 |
| One semantic mask per image | `semantic_segmentation_template.ipynb` | `cv.segmentation`, `cv.tta` | Dice, IoU |
| Image class with training masks | `classification_segmentation_template.ipynb` | `cv.multitask` | classification metric |
| Bounding boxes | `object_detection_template.ipynb` | `cv.detection` | mAP, IoU |
| Separate mask per object | `instance_segmentation_template.ipynb` | `cv.instance_segmentation` | mask mAP |
| Robustness/adversarial examples | CV classification + attack block | `cv.adversarial` | robust accuracy |
| Text class, very fast CPU | `nlp_classification_template.ipynb` | `nlp.classification.TfidfTextClassifier` | accuracy, macro F1 |
| Transformer text class | `transformer_text_classification_template.ipynb` | `build_hf_text_classifier` | accuracy, macro F1 |
| Named entities/token tags | `token_classification_template.ipynb` | `nlp.token_classification` | entity F1 |
| Translation/summarization/generation | `seq2seq_template.ipynb` | `nlp.seq2seq` | BLEU, chrF, ROUGE-L |
| Query-document matching/ranking | `retrieval_template.ipynb` | `nlp.retrieval` | MRR, Recall@K |
| Image-to-text/captioning | `image_captioning_template.ipynb` | `multimodal.image_to_text` | BLEU, CIDEr, ROUGE |

## Semantic vs instance segmentation

- **Semantic segmentation** returns one class for each pixel. Touching objects of the same class
  merge. Use U-Net/FPN/DeepLab and Dice/IoU.
- **Instance segmentation** returns a separate mask, class, and confidence for each object. Use
  Mask R-CNN and the official mask-mAP evaluator.
- **Classification plus segmentation head** still submits image classes. Its mask head is an
  auxiliary training signal and is useful only when reliable training masks exist.

## Problems that still need task-specific adaptation

Audio, OCR, keypoint/pose, depth, super-resolution, and extractive QA are not turnkey notebooks in
this kit. The shared split, configuration, reporting, and submission utilities remain reusable,
but the released data format and official metric must drive those adapters. Do not pretend a
nearby metric or output format is equivalent.
