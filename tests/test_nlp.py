from __future__ import annotations

from olp_ai_26.nlp.classification import TfidfTextClassifier
from olp_ai_26.nlp.retrieval import TfidfRetriever, reciprocal_rank
from olp_ai_26.nlp.text_cleaning import normalize_text


def test_vietnamese_normalization_preserves_diacritics():
    assert normalize_text("  Xin   chào  Việt Nam ") == "Xin chào Việt Nam"


def test_tfidf_classifier_smoke():
    texts = [
        "mèo đáng yêu",
        "mèo con dễ thương",
        "con mèo ngủ",
        "mèo đang ăn",
        "chó chạy nhanh",
        "chó con sủa",
        "con chó ngủ",
        "chó đang ăn",
    ]
    labels = ["cat"] * 4 + ["dog"] * 4
    model = TfidfTextClassifier(max_features=1000).fit(texts, labels)
    predictions = model.predict(["mèo con", "chó sủa"])
    assert predictions.tolist() == ["cat", "dog"]


def test_tfidf_retrieval_and_mrr():
    retriever = TfidfRetriever().fit(["mèo con", "chó chạy", "dịch máy"])
    results = retriever.search(["con mèo"], top_k=2)
    assert results[0][0][0] == 0
    assert reciprocal_rank([{0}], [[0, 2]]) == 1.0
