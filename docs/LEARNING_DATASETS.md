# Stage the representative practice datasets

[Learning notebook pairs](../notebooks/learning/README.md) · [Evidence ledger](KAGGLE_METHOD_EVIDENCE.md)

Use the original dataset page for each problem. Accept applicable terms yourself before the
timed exercise and keep the official sample submission. Every learning notebook has an optional
Kaggle download/extraction cell; manual local staging remains available. No model weights are
fetched. A selected training subset is a practice budget, not a new benchmark.

| Folder / dataset page | Required local contents | What to inspect first |
|---|---|---|
| `01_credit` — [Home Credit](https://www.kaggle.com/competitions/home-credit-default-risk/data) | `application_train.csv`, `application_test.csv`, `sample_submission.csv`; `bureau.csv` for the advanced aggregate experiment | Default prevalence, missingness, repeated applicants/history, feature availability at decision time. |
| `02_prices` — [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) | `train.csv`, `test.csv`, `sample_submission.csv` | Positive prices, long tails, structural versus unknown missing values, numerical/categorical features. |
| `03_toxic` — [Jigsaw Toxic Comments](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) | `train.csv`, `test.csv`, `sample_submission.csv` | Six independent binary columns, rare labels, duplicate comments, length/truncation. Do not use `test_labels.csv` for model selection. |
| `04_leaves` — [Cassava](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data) | `train.csv`, `sample_submission.csv`, `train_images/`, `test_images/` | Imbalance, noisy labels, repeated plants, blur and image dimensions; IDs are complete filenames. |
| `05_salt` — [TGS Salt](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/data) | `train.csv`, `sample_submission.csv`, `train/images/`, `train/masks/`, `test/images/` | Empty masks, foreground coverage, image/mask alignment, original resolution. Extract nested train/test archives into those folders. |
| `06_wheat` — [Global Wheat](https://www.kaggle.com/competitions/global-wheat-detection/data) | `train.csv`, `sample_submission.csv`, `train/`, `test/` | Several box rows per image, `source`, small boxes, xywh coordinates and missing annotations. Original image suffix is `.jpg`. |
| `07_matching` — [Shopee](https://www.kaggle.com/competitions/shopee-product-matching/data) | `train.csv`, `test.csv`, `sample_submission.csv`, `train_images/`, `test_images/` | Product groups, duplicate images/text, long-tail group size and text/image disagreement. |

Use a dedicated folder for each. Set the notebook's configuration `DATA_DIR` to that folder;
use the self-contained notebook's inline folder/ZIP staging and durable-copy functions. No
repository or companion Python source is needed. On Kaggle itself,
point `DATA_DIR` at the actual mounted competition directory. Code competitions may impose a
separate notebook submission contract; a locally checked CSV does not prove platform acceptance.

## Download from a learning notebook

1. On the competition's Kaggle page, review and accept its rules yourself. Some historical
   competitions restrict access; an API token cannot bypass their access requirements.
2. Locally run `uv sync --locked`: the default `kaggle-extra` group includes the official
   `kaggle>=2.2,<3` client. For a copied notebook/Colab runtime, install that small client separately
   only if missing; do not run uv sync in Colab or replace its torch/NumPy/CUDA packages.
3. Configure credentials using Kaggle's environment/configuration support. The current client
   supports `KAGGLE_API_TOKEN`, `~/.kaggle/access_token`, and legacy `~/.kaggle/kaggle.json`.
   In Colab, `PROMPT_FOR_KAGGLE_TOKEN` defaults to true when downloading, providing hidden input;
   blank input keeps already configured credentials. On a local machine, enable it if you want
   that prompt. Never put the token directly in a saved code cell or commit a credential file.
4. In stage 1 set `DATA_DIR` to a dedicated empty destination, then run setup. For example, use
   `paths.data_dir / "kaggle" / FAMILY` instead of a folder already holding synthetic fixtures.
5. In stage **1a**, set `DOWNLOAD_DATA=True` and run the cell. Each notebook already has its correct
   competition slug and expected files/directories. The official command downloads the competition
   ZIP into `DOWNLOAD_CACHE`; the notebook extracts into a temporary sibling folder and publishes
   it only after layout checks pass. TGS train/test ZIPs and individually zipped CSVs are expanded.
6. Confirm the printed dataset path, then restore `DOWNLOAD_DATA=False` for offline work. Training
   time starts after acquisition. A marked complete folder or intact cache is reused on repeats.

The cell never accepts rules or submits anything. Client errors direct you to check credentials,
competition permissions, internet and disk space; captured client output is not copied into the
notebook because it could contain signed URLs. A failed extraction preserves its temporary folder
for inspection and leaves `DATA_DIR` unchanged. Archives plus extracted media can occupy several
GB, so check storage before starting. The seven complete real datasets are not fetched during
repository tests; those tests simulate client responses and exercise real archive extraction.

Primary references: [official Kaggle CLI authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication),
[competition-download command](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md#kaggle-competitions-download),
and [competition access/rules](https://www.kaggle.com/docs/competitions).

For a genuinely unseen repeat, ask a teammate to prepare a smaller pack with intact group
boundaries and shuffled sample IDs. Keep test labels outside the working notebook. Thresholds,
class weights, early stopping, feature selection and ensemble decisions use training-derived
validation, not the held-out answer file or repeated public-leaderboard feedback.

## Advanced assets

The notebooks always default to `PRETRAINED_ALLOWED=False`. If exact local assets are permitted
for your practice task, record their version/hash and provenance in `ASSET_PROVENANCE` before
enabling them. Kaggle's original weight/data permissions do not grant permission in OlpAI.

- **Toxic BiGRU:** optional NPZ with string `tokens` of shape `[V]` and float `vectors` of shape
  `[V, EMBEDDING_DIM]`; `allow_pickle=False`. The training vocabulary is built separately from the
  training fold. A missing embedding file means random embeddings, which changes the experiment.
- **Cassava:** `WEIGHT_PATH` is an exact local original checkpoint supported by the timm model
  named in `MODEL_NAME`. Use its proper normalization. The study default is EfficientNet-B4;
  B0 is an optional smaller family adaptation when cost justifies that separate comparison.
- **Salt U-Net:** optional `WEIGHT_PATH` is the original torchvision ResNet34 state dictionary.
  It initializes only the encoder; the decoder is trained on the training fold. The advanced
  route uses ImageNet mean/std consistently for train/validation/test and needs no SMP package.
- **Detection:** `WEIGHT_PATH` is the original detector state matching the selected torchvision
  architecture. The factory replaces its head for foreground/background. A saved own training
  checkpoint reloads directly without the original initialization asset.
- **Matching encoders:** `TEXT_MODEL_DIR` includes the permitted Hugging Face model and tokenizer;
  loading is local-only. `IMAGE_MODEL_CHECKPOINT` matches the timm `MODEL_NAME`. The notebook
  computes frozen text/image features and saves encoders for future inputs.
- **Matching cache alternative:** NPZ contains string `ids [N]`, float `image [N,Di]`, float
  `text [N,Dt]`, and string `fit_ids [M]`. Every requested train/validation/test ID must be present.
  `fit_ids` declares task-specific fitting rows and must be a subset of the training fold; an
  empty array declares frozen encoders with no task-specific fitting. Audit that claim: metadata
  cannot prove an encoder did not use validation labels. Record exact encoder/source provenance.

A cache can be created offline ahead of a **learning** run when permitted. Its extraction cost
belongs in the comparison. A cache-only bundle cannot embed new private examples; preserve the
actual encoders and preprocessing when the task will supply new inputs. Original winning methods
that fit on test/pseudo-labels are intentionally excluded.

## Offline schema fixtures

`uv run olp-ai practice-data data/learning --task learning` creates seven tiny artificial packs
in the expected schemas, with independent grading files. They include RGB images, masks, boxes,
six-label text and mixed tables. For Wheat only, set `IMAGE_SUFFIX=".png"` when using these fixtures.
They are not the Kaggle datasets and cannot establish whether an advanced model is competitive.
Their purpose is quick debugging, shuffled-ID checks and infrastructure verification before you
spend GPU time on real practice data.
