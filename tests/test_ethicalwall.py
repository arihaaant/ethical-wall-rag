import random
import zlib

import numpy as np
import pandas as pd
import pytest

from ethicalwall import data, evaluate
from ethicalwall.retrieval import Index


class HashEncoder:
    # bag of words hashed into a fixed size vector, enough to make retrieval deterministic in tests
    def encode(self, texts, normalize_embeddings=True, **kw):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.encode()) % 64] += 1
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
        return out[0] if single else out


@pytest.fixture
def docs():
    rows = [{"title": f"contract {i}", "context": f"contract {i} governing law clause " * 60} for i in range(10)]
    return data.sample_docs(pd.DataFrame(rows), 10, seed=0)


@pytest.fixture
def index(docs):
    return Index(data.chunk(docs), HashEncoder())


def test_category_from_id():
    assert data.category_from_id("SOME CO_EX-10__Anti-Assignment_3") == "Anti-Assignment"
    assert data.category_from_id("X__Affiliate License-Licensee_0") == "Affiliate License-Licensee"
    assert data.category_from_id("X__Document Name_12") == "Document Name"


def test_no_user_straddles_a_wall():
    data.check_walls()


def test_acl_matches_clients(docs):
    for r in docs.itertuples():
        assert r.acl == [u for u, allowed in data.users.items() if r.client in allowed]
    assert all(len(r.acl) == 4 for r in docs.itertuples() if r.client == "E")


def test_chunks_cover_text_with_correct_offsets(docs):
    chunks = data.chunk(docs, size=100, stride=80)
    for c in chunks:
        assert docs.loc[c["doc"], "context"][c["start"]:c["end"]] == c["text"]
    first = [c for c in chunks if c["doc"] == 0]
    assert first[0]["start"] == 0 and first[-1]["end"] == len(docs.loc[0, "context"])


def test_overlaps():
    chunk = {"start": 100, "end": 200}
    assert evaluate.overlaps(chunk, {"text": ["abc"], "answer_start": [198]})
    assert not evaluate.overlaps(chunk, {"text": ["abc"], "answer_start": [200]})
    assert not evaluate.overlaps(chunk, {"text": ["abcdef"], "answer_start": [94]})


def test_filtered_search_never_leaks_and_keeps_k(index):
    for user in data.users:
        hits = index.search("contract 1 governing law", user, k=5)
        assert len(hits) == 5
        assert evaluate.n_leaks(hits, user) == 0


def test_unfiltered_leaks_and_postfilter_loses_results(docs, index):
    title = docs[docs.client == "B"].title.iloc[0]
    q = f"{title} governing law"
    assert evaluate.n_leaks(index.search(q, None, k=5), "user_A") > 0
    assert len(index.search_postfilter(q, "user_A", k=5)) < 5


def test_redteam_pool_size(docs):
    rt = evaluate.build_redteam(docs, random.Random(0))
    for user, allowed in data.users.items():
        assert (rt.user == user).sum() == 2 * (~docs.client.isin(allowed)).sum()
    assert set(rt.kind) == {"direct", "multihop"}


def test_wilson():
    w = evaluate.wilson(0, 288)
    assert w["rate"] == 0 and w["ci_low"] == pytest.approx(0) and 0.01 < w["ci_high"] < 0.02
