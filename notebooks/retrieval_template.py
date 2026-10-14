# %% [markdown]
# # Text retrieval/ranking baseline
# Use this when a query must rank document IDs. This CPU TF-IDF baseline is fast and debuggable.

# %%
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/olp-ai-26") if "google.colab" in sys.modules else Path.cwd()
REQUIRED_MODULES = ()
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
from olp_ai_26.core.config import CompetitionConfig
from olp_ai_26.core.report import record_run
from olp_ai_26.core.split import make_split
from olp_ai_26.core.submission import build_submission, write_submission
from olp_ai_26.nlp.retrieval import TfidfRetriever, reciprocal_rank

# %% [markdown]
# ## 0. Drag-and-plug configuration

# %%
DRIVE_DATA_ARCHIVE = None  # or Path("/content/drive/MyDrive/olpai26/data.zip")
PERSISTENT_DIR = None  # or Path("/content/drive/MyDrive/olpai26/retrieval")
if DRIVE_DATA_ARCHIVE or PERSISTENT_DIR:
    mount_google_drive()
paths = ColabPaths.create(persistent_dir=PERSISTENT_DIR)
paths.output_dir = paths.output_dir / "retrieval"
paths.output_dir.mkdir(parents=True, exist_ok=True)
DATA_DIR = paths.data_dir
if DRIVE_DATA_ARCHIVE:
    stage_data(DRIVE_DATA_ARCHIVE, DATA_DIR)
DOCUMENTS_CSV, QUERIES_CSV = DATA_DIR / "documents.csv", DATA_DIR / "queries.csv"
DOCUMENT_ID, DOCUMENT_TEXT = "document_id", "text"
QUERY_ID, QUERY_TEXT, RELEVANT_ID = "query_id", "query", "relevant_document_id"
TOP_K = 10
GROUP_COLUMN = None
config = CompetitionConfig(data_dir=DATA_DIR, output_dir=paths.output_dir, device="cpu")
config.prepare()
print(gpu_report())

# %% [markdown]
# ## 1. Fit document index and search

# %%
documents = pd.read_csv(DOCUMENTS_CSV, converters={DOCUMENT_ID: str})
queries = pd.read_csv(QUERIES_CSV, converters={QUERY_ID: str, RELEVANT_ID: str})
if documents[DOCUMENT_ID].duplicated().any():
    raise ValueError("Document IDs must be unique")
if RELEVANT_ID not in queries:
    raise ValueError("Provide labeled validation queries before tuning retrieval")
print(documents.head())
split = make_split(queries, group_column=GROUP_COLUMN, seed=config.seed)
retriever = TfidfRetriever(ngram_range=(1, 2), max_features=200_000)
retriever.fit(documents[DOCUMENT_TEXT])
results = retriever.search(split.valid[QUERY_TEXT], top_k=TOP_K)
checkpoint = retriever.save(paths.output_dir / "index.pkl")
retriever = TfidfRetriever.load(checkpoint)

# %% [markdown]
# ## 2. Local MRR when relevance labels exist

# %%
id_to_position = {value: index for index, value in enumerate(documents[DOCUMENT_ID])}
relevant = [{id_to_position[value]} for value in split.valid[RELEVANT_ID]]
ranked_positions = [[index for index, score in row] for row in results]
score = reciprocal_rank(relevant, ranked_positions)
print({f"mrr_at_{TOP_K}": score})

# %% [markdown]
# ## 3. Full test search and exact query/rank sample alignment
# Practice sample columns: query_id, rank, document_id. Adapt to the official ranked-output format.

# %%
test = pd.read_csv(DATA_DIR / "test_queries.csv", converters={QUERY_ID: str})
sample = pd.read_csv(
    DATA_DIR / "sample_submission.csv", converters={QUERY_ID: str, DOCUMENT_ID: str}
)
results = retriever.search(test[QUERY_TEXT], top_k=TOP_K)
rows = []
for query_id, results_for_query in zip(test[QUERY_ID], results, strict=True):
    rows.extend(
        {QUERY_ID: query_id, DOCUMENT_ID: documents.iloc[index][DOCUMENT_ID], "rank": rank}
        for rank, (index, _) in enumerate(results_for_query, start=1)
    )
prediction_table = pd.DataFrame(rows)
submission = build_submission(
    sample,
    {DOCUMENT_ID: prediction_table[DOCUMENT_ID].tolist()},
    prediction_ids=prediction_table[[QUERY_ID, "rank"]],
    id_columns=[QUERY_ID, "rank"],
)
write_submission(
    submission, paths.output_dir / "submission.csv", sample=sample, id_columns=[QUERY_ID, "rank"]
)
record_run(
    paths.output_dir,
    task="retrieval",
    model="tfidf_cosine",
    configuration=config,
    validation_metrics={f"mrr_at_{TOP_K}": score},
    split=split,
    data_summary={"documents": len(documents), "queries": len(queries)},
    inference_options={"top_k": TOP_K, "document_ids": documents[DOCUMENT_ID].tolist()},
    checkpoint=str(checkpoint),
)
sync_artifacts(paths.output_dir, paths.persistent_dir)
submission.head()
