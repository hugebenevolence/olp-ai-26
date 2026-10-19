# Task 2 representation audit and AnomalyDINO adoption

## Evidence from the executed 0.590 run

The supplied `image_anomaly_detection_template (3).ipynb` executed the
`category_augmented` preset on an A100. It predicted 57 anomalies out of 480, distributed as
`3, 10, 5, 6, 19, 14` across categories 01-06. The user reports an official score of `0.590`.

The learned synthetic proxy did not cleanly separate held-out normal and generated examples:
per-category proxy balanced accuracy ranged from `0.6990` to `0.9116`. Normal-only thresholds were
also much higher than the proxy-optimal thresholds in categories 01, 02, 04, and 06. This means the
run changed both the score representation and decision boundary without improving the official
result. The next experiment therefore isolates representation and scoring.

## Primary-source comparison

| Method | Relevant result/design | Fit for this contest | Decision |
|---|---|---|---|
| [PatchCore](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html) | Normal-only mid-level patch memory with representative coreset; up to 99.6 image AUROC on MVTec AD | Already approximated, but the current random projection and tiny random memory are not the paper's full feature/coreset recipe | Preserve only as historical baseline |
| [AnomalyDINO](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_With_DINOv2_WACV_2025_paper.html) and [official code](https://github.com/dammsi/AnomalyDINO) | Training-free DINOv2 patch tokens, cosine 1-NN, mean of highest 1% patch distances; 96.6 one-shot AUROC on MVTec AD | Direct match to normal-only sensory defects; simple enough for six hours and one Colab GPU | Adopt as the next controlled run |
| [EfficientAD](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html) | Student-teacher branch plus autoencoder for structural and logical anomalies | Strong follow-up, but requires a carefully matched training loop and calibrated branch combination | Keep as fallback, not first change |
| [Dinomaly](https://arxiv.org/abs/2405.14325) | DINOv2-based Transformer reconstruction; 99.6/98.7 image AUROC on MVTec AD/VisA | Strong but much larger implementation and training-risk surface during a six-hour contest | Do not port before the simpler representation test |

## Gap table for the selected method

| AnomalyDINO contract | Previous notebook | Likely impact | Adopted change |
|---|---|---|---|
| DINOv2-S final patch tokens | ImageNet CNN features from different category backbones | Representation is the main methodological mismatch | `DinoV2PatchFeatureExtractor` with `vit_small_patch14_dinov2.lvd142m` |
| 448-pixel smaller edge; patch size 14 | Square 288-320 inputs and CNN feature strides | Fewer/coarser patches can miss short lines or low-opacity local defects | Controlled 448-by-448 input, producing 32-by-32 patch tokens |
| Cosine distance to one nearest normal patch | Euclidean distance on projected normalized CNN features | Euclidean ranking is related, but its scale and representation are not the paper's setup | Exact cosine distance and 1-NN |
| Mean of most anomalous 1% of patches | Fixed top-5 mean | Fixed count changes effective tail fraction with feature-grid size | `top_fraction=0.01`, giving 11 of 1,024 patches |
| Training-free scoring | Synthetic positive head | Synthetic proxy may overfit known generated artifacts | No synthetic head in `anomalydino_448` |
| Nominal memory bank | 6,144-8,192 random CNN patches | Too few patches can discard normal modes | 32,768 DINOv2 patches per category |
| Optional PCA foreground mask | No patch mask | May reduce background false positives, but the paper documents category failures | Deliberately deferred until the backbone-only result is known |

## Competition adaptation and verification boundary

The preset is named `anomalydino_448`. It is a faithful implementation of the paper's core feature,
distance, and tail-scoring contract, but it is not an exact paper replication:

- This dataset is full-shot; the paper's headline setting is few-shot.
- Images are resized to a square for efficient batched Colab inference rather than preserving the
  smaller edge and variable aspect ratio.
- The full patch bank is bounded by deterministic reservoir sampling. The paper advises coreset
  reduction for larger banks, but its few-shot default does not need one.
- PCA foreground masking is intentionally off because it is category-sensitive and can hide valid
  object patches. It should be tested as a separate experiment only if anomaly maps show background
  false positives.
- With no labeled anomalies, the official hidden balanced accuracy and optimal binary threshold
  cannot be computed locally. Held-out normal quantiles remain a competition adaptation.

## Next Colab run

Use only:

```python
EXPERIMENT_PRESET = "anomalydino_448"
RUN_TRAINING = True
PHASE = "public"
```

Submit the main candidate first. Compare its per-category positive counts and official aggregate
score with the recorded 0.590 run before testing threshold scales. Do not combine DINOv2, masking,
synthetic evidence, and a new threshold in the same first submission.
