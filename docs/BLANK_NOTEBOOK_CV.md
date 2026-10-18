# Image baseline reconstruction exercise

[Blank-notebook playbook](BLANK_NOTEBOOK_PLAYBOOK.md) ·
[Text/table study cells](../scripts/blank_baseline.py)

This is the **single-label image classification** extension to the common workflow. Study these
cells, close the page, and rebuild them. They use only PyTorch, Pillow, NumPy, pandas and sklearn.
They do not download weights, use mixed precision, or assume torchvision is available. Full
precision is a deliberate first-baseline simplification; add hardware-aware AMP after correctness.
This small CNN is a plumbing fallback, not a promise of strong performance on an arbitrary dataset.

First recreate `read_tables`, `split_rows`, `score_labels`, and `deliver` from the answer key in
your blank notebook. Keep its imports. Configure `DATA_DIR`, `OUTPUT_DIR` (a new directory),
`METRIC` (`accuracy` or the answer key's explicit `macro_f1` convention), `SPLIT`, and `SPLIT_COLUMN`.
Use the official metric instead when different. The pack schema is `id,image,label` for training,
`id,image` for test, and `id,label` for sample output; image paths are relative to `DATA_DIR`.
For the synthetic image pack, set `DATA_DIR = Path("data/practice/image/data")`,
`OUTPUT_DIR = Path("outputs/blank-image-01")`, `METRIC = "macro_f1"`,
`SPLIT = "stratified"`, and `SPLIT_COLUMN = None` in your own configuration cell.

## 1. Read, split, encode labels and inspect one example

```python
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)
train, test, sample = read_tables(DATA_DIR)
a, b = split_rows(train, SPLIT, SPLIT_COLUMN, 42)
labels = sorted(train["label"].unique().tolist())
if not len(a) or not len(b) or set(train.iloc[a]["label"]) != set(labels):
    raise ValueError("Inspect empty splits or classes missing from the training fold")
output = OUTPUT_DIR
output.mkdir(parents=True, exist_ok=False)


class Images(Dataset):
    def __init__(self, frame, root, labels, size=64):
        self.frame, self.root, self.labels, self.size = frame, root, labels, size

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.root / row["image"]) as image:
            image = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32).copy() / 255.0
        x = torch.from_numpy(pixels).permute(2, 0, 1)
        y = self.labels.index(row["label"]) if "label" in row.index else -1
        return x, y


training = DataLoader(Images(train.iloc[a], DATA_DIR, labels), batch_size=16, shuffle=True)
validation = DataLoader(Images(train.iloc[b], DATA_DIR, labels), batch_size=32, shuffle=False)
testing = DataLoader(Images(test, DATA_DIR, labels), batch_size=32, shuffle=False)
pixels, targets = next(iter(training))
print(pixels.shape, pixels.min().item(), pixels.max().item(), targets)
```

The train and validation transforms are identical here. This deliberately avoids augmentation
bugs in the first route. Inspect actual images and decoded labels as well as tensor shapes.
Repeated specimens/persons/videos need a grouped split; the random practice split cannot detect
that need. On Colab stage images on local runtime storage before the timer; do not repeatedly
decode from mounted Drive. On the contest platform use its permitted local storage.

## 2. Build a small network and an evaluation-mode prediction function

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_cnn(n_classes):
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, n_classes),
    )


@torch.inference_mode()
def predict_ids(model, loader):
    model.eval()
    chunks = [model(x.to(device)).argmax(1).cpu().numpy() for x, _ in loader]
    return np.concatenate(chunks)


model = make_cnn(len(labels)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_function = nn.CrossEntropyLoss()
```

Cross-entropy receives raw logits `[B,K]` and integer class indices `[B]`; do not add softmax before
it. Keep the string label vocabulary for decoding. The prediction function changes the model to
evaluation mode, so every training epoch below explicitly switches back to training mode.

## 3. Train briefly, validate globally, preserve the best checkpoint

```python
best = -float("inf")  # Only the two supported higher-is-better classification metrics here.
EPOCHS = 3  # Measure epoch time and stop early enough for complete inference and delivery.
for epoch in range(EPOCHS):
    model.train()
    for x, y in training:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(x), y)
        loss.backward()
        optimizer.step()
    predicted = np.asarray(labels)[predict_ids(model, validation)]
    value = score_labels(train.iloc[b]["label"], predicted, METRIC, labels)
    print(epoch, METRIC, value)
    if value > best:
        best = value
        temporary = output / "best.pt.tmp"
        torch.save({"model": model.state_dict(), "labels": labels, "size": 64}, temporary)
        temporary.replace(output / "best.pt")
```

Before the timed real-fold run, perform a separate tiny-subset fitting drill: disable augmentation,
try 16–32 training examples, and evaluate on **those same examples** until the model clearly learns.
That is a data/loss/gradient check, not a validation result. Then rebuild the model/optimizer and
train only the chosen training fold. Three epochs on the real task are not a quality target. If
training is slow, shrink resolution/batch/data for debugging and measure useful learning progress.

## 4. Reload the best model, decode and deliver in sample order

```python
checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
labels = checkpoint["labels"]
restored = make_cnn(len(labels)).to(device)
restored.load_state_dict(checkpoint["model"], strict=True)
restored_validation = np.asarray(labels)[predict_ids(restored, validation)]
assert score_labels(train.iloc[b]["label"], restored_validation, METRIC, labels) == best
predictions = np.asarray(labels)[predict_ids(restored, testing)]
deliver(sample, test["id"], predictions, output)
recipe = {
    "metric": METRIC,
    "validation_score": best,
    "labels": labels,
    "size": checkpoint["size"],
    "pixel_scale": "RGB float32 / 255",
    "seed": 42,
    "split": SPLIT,
    "split_column": SPLIT_COLUMN,
    "pretrained_allowed": False,
    "train_ids": train.iloc[a]["id"].tolist(),
    "validation_ids": train.iloc[b]["id"].tolist(),
}
(output / "run.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
```

The test loader must remain unshuffled because `test["id"]` is in manifest order. `deliver` then
aligns to the possibly shuffled sample. For a fresh-process recovery, recreate preprocessing
using the saved `size` and pixel scale, restore label order from the checkpoint, rebuild `make_cnn`,
and load the state dict before inference. A different normalization or class ordering breaks that
promise even if the weights load successfully. Preserve the source and required report too.

This reference uses a fixed three-epoch practice budget; it is not an automatic contest time-budget
manager. Benchmark on the actual hardware and reserve measured inference/upload time. Use a larger
model, task-valid augmentation or allowed local weights only after the baseline is saved and
validation errors justify the additional cost. For other output families, return to the task card
and build the appropriate decoder and scorer.
