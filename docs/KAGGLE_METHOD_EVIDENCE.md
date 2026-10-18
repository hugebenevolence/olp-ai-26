# Evidence behind the advanced notebook pairs

[Learning notebook pairs](../notebooks/learning/README.md) · [Data staging](LEARNING_DATASETS.md)

Primary sources checked 18 October 2026. These are historical competition results or author claims,
not a current SOTA survey. The code here is an educational implementation of selected components.
There is **no evidence that one reduced recipe is universally the most efficient**; measure score,
implementation effort, extraction, fit, inference and failure risk on your actual hardware.

| Problem | Primary evidence | Implemented component | Important difference |
|---|---|---|---|
| Home Credit | [First-place feature-engineering account](https://www.kaggle.com/competitions/home-credit-default-risk/discussion/64821), [author's seventh-place single LightGBM component](https://www.kaggle.com/code/jsaguiar/lightgbm-7th-place-solution) | A few per-applicant history aggregates and a single LightGBM classifier. | Much fewer features and no winning blend. This is not the seventh-place complete ensemble. |
| House Prices | [Author's “Stacked Regressions: Top 4%” notebook](https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard) | A small out-of-fold Ridge/gradient-boosting stack with preprocessing inside each inner fold. | Historical author-reported percentile, not a current leaderboard rank or first-place result. |
| Toxic Comments | [First-place author's write-up](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/writeups/toxic-crusaders-1st-place-solution-overview) | Two-layer BiGRU and optional local pretrained embeddings. | No translations, test pseudo-labels, six-GPU search or large stack. Random embeddings do not replicate pretrained performance. |
| Cassava | [First-place author's write-up](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/discussion/221957) | Single configurable EfficientNet-B4. | The winner used an ensemble including specialized CropNet; B0 is an optional smaller adaptation. No specialized external model is imported here. |
| TGS Salt | [Winning repository](https://github.com/ybabakhin/kaggle_salt_bes_phalanx), [authors' paper](https://arxiv.org/abs/1904.04445) | ResNet34 encoder U-Net, mask-loss comparison and threshold selection. | A smaller residual encoder variant, with optional exact local initialization; no winning ensemble or semi-supervised test fitting. |
| Global Wheat | [First-place source repository](https://github.com/dungnb1333/Global-Wheat-Dection-1st-Place) | Single Faster R-CNN FPN and confidence selection. | A smaller torchvision ResNet model in that family; no EfficientDet ensemble, multi-scale WBF, extra datasets or test pseudo-labeling. |
| Shopee | [First-place author's write-up](https://www.kaggle.com/competitions/shopee-product-matching/writeups/upstage-making-ai-beneficial-1st-place-solution-fr) | Frozen local image/text embeddings, per-modality normalization, fusion and match-threshold comparison. | No ArcFace fitting, large model ensemble or iterative neighborhood blending. Local encoder quality remains a separate variable. |

## Metric traps to learn explicitly

- [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
  defines RMSE between log prices. The helper uses natural logs of strictly positive prices,
  rather than silently substituting log1p.
- [Cassava](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) selects by
  accuracy. Macro F1 is useful to inspect minority errors but is not the official ranking metric.
- [TGS Salt](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/overview) averages
  image-level hits across strict IoU cutoffs .50–.95. Dice and continuous IoU are diagnostics only.
  Empty-mask behavior and original resolution matter; RLE is one-based and column-major.
- [Global Wheat's statement](https://www.kaggle.com/competitions/global-wheat-detection) specifies
  confidence-ordered matching across .50–.75 and gives no-target images zero. The
  [winner's reference evaluator](https://github.com/dungnb1333/Global-Wheat-Dection-1st-Place/blob/main/evaluation.py)
  uses inclusive pixel areas and non-strict cutoff comparisons. The teaching helper exposes the
  pixel convention, follows the statement's strict cutoff and empty-target rule by default, and
  has a non-strict option for reference-code comparisons. Check boundary fixtures against a
  supplied official grader rather than treating two similar implementations as interchangeable.
- [Shopee matching F1](https://www.kaggle.com/c/shopee-product-matching/overview/evaluation) is
  averaged per query over predicted/relevant ID sets. It is not pairwise accuracy or retrieval
  MRR. Queries self-match; the task caps true groups at 50. A top-50 postprocessing candidate is
  a useful validation experiment, rather than an assumption to transfer to other retrieval tasks.
- [Toxic Comments](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/overview/evaluation)
  requires six independent probabilities and their mean column-wise AUC, not the different
  subgroup-bias metric from Jigsaw's separate 2019 competition.

Use each notebook's `comparison_row.csv` and saved fold IDs to judge a practical advance. Gains
from different splits, additional data or different weight permissions are different experiments.
When a smaller baseline gives a trustworthy score and a timely valid submission, retaining it is
a legitimate result of the comparison.
