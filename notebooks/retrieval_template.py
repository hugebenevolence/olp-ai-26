# %% [markdown]
# # Text retrieval/ranking baseline
# Use this when a query must rank document IDs. This CPU TF-IDF baseline is fast and debuggable.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
exec((PROJECT_ROOT / "notebooks" / "_colab_bootstrap.py").read_text(encoding="utf-8"))

# %%
import pandas as pd

from olp_ai_26.core.colab import (
    ColabPaths,
    gpu_report,
    mount_google_drive,
    stage_data,
    sync_artifacts,
)
from olp_ai_26.nlp.retrieval import TfidfRetriever, reciprocal_rank

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/retrieval")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
DOCUMENTS_CSV, QUERIES_CSV = DATA_DIR / "documents.csv", DATA_DIR / "queries.csv"
DOCUMENT_ID, DOCUMENT_TEXT = "document_id", "text"
QUERY_ID, QUERY_TEXT, RELEVANT_ID = "query_id", "query", "relevant_document_id"
TOP_K = 10
print(gpu_report())

# %% [markdown]
# ## 1. Fit document index and search

# %%
documents, queries = pd.read_csv(DOCUMENTS_CSV), pd.read_csv(QUERIES_CSV)
retriever = TfidfRetriever(ngram_range=(1, 2), max_features=200_000)
retriever.fit(documents[DOCUMENT_TEXT])
results = retriever.search(queries[QUERY_TEXT], top_k=TOP_K)
ranked_ids = [[documents.iloc[index][DOCUMENT_ID] for index, score in row] for row in results]

# %% [markdown]
# ## 2. Local MRR when relevance labels exist

# %%
if RELEVANT_ID in queries:
    id_to_position = {value: index for index, value in enumerate(documents[DOCUMENT_ID])}
    relevant = [{id_to_position[value]} for value in queries[RELEVANT_ID]]
    ranked_positions = [[index for index, score in row] for row in results]
    print({"mrr": reciprocal_rank(relevant, ranked_positions)})

# %% [markdown]
# ## 3. Long-format submission; change to match the sample exactly

# %%
rows = []
for query_id, result_ids in zip(queries[QUERY_ID], ranked_ids, strict=True):
    rows.extend(
        {QUERY_ID: query_id, DOCUMENT_ID: document_id, "rank": rank}
        for rank, document_id in enumerate(result_ids, start=1)
    )
submission = pd.DataFrame(rows)
submission.to_csv(paths.output_dir / "submission.csv", index=False)
sync_artifacts(paths.output_dir, paths.persistent_dir)
submission.head()
