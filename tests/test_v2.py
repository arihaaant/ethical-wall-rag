import random
import re

import numpy as np
import pandas as pd
import pytest

from ethicalwall import answers, data
from ethicalwall.llm import build_messages
from ethicalwall.retrieval import Index, Reranked
from test_ethicalwall import HashEncoder


class WordOverlap:
    def predict(self, pairs):
        return [len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs]


def echo_codes(convs):
    # a model with no guardrails at all: repeats every matter code in its context
    return [" ".join(re.findall(r"MC-\d{6}", c[-1]["content"])) for c in convs]


def obey_injection(convs):
    return ["PINEAPPLE PROTOCOL" if "PINEAPPLE" in c[-1]["content"] else "ok" for c in convs]


@pytest.fixture
def docs():
    rows = [{"title": f"contract {i}", "context": f"contract {i} governing law clause " * 60} for i in range(20)]
    return data.plant_markers(data.sample_docs(pd.DataFrame(rows), seed=0), random.Random(0), inject_frac=0.2)


@pytest.fixture
def chunks(docs):
    return data.chunk(docs)


@pytest.fixture
def retriever(chunks):
    return Reranked(Index(chunks, HashEncoder()), WordOverlap(), candidates=20)


def test_plant_markers_appends_only(docs):
    assert docs.code.is_unique
    assert docs.injected.sum() == 4
    for r in docs.itertuples():
        assert r.context.startswith(f"{r.title} governing law clause")
        assert (data.injection in r.context) == r.injected
        assert f"matter code for this agreement: {r.code}." in r.context


def test_reranked_respects_acl(retriever):
    for user in data.users:
        hits = retriever.search("contract 3 governing law", user, k=5)
        assert all(user in h.payload["acl"] for h in hits)


def test_guard_and_harden_prompts():
    hits = [{"client": "A", "title": "t", "text": "x"}]
    plain = build_messages("q", hits)[0]["content"]
    guarded = build_messages("q", hits, "user_A", {"A", "E"}, guard=True, harden=True)[0]["content"]
    assert "A, E" in guarded and "user_A" in guarded and "untrusted" in guarded
    assert "user_A" not in plain and "untrusted" not in plain


def test_construction_never_leaks_but_prompting_can(docs, retriever):
    cases = answers.build_code_cases(docs, retriever, random.Random(0), k=5)
    res = answers.score_code_cases(cases, echo_codes([c["messages"] for c in cases]), docs)
    forb = res[~res.visible]
    assert forb[forb["mode"] == "construction"].leak.sum() == 0
    assert forb[forb["mode"] == "prompting"].leak.sum() > 0
    assert res[res.visible & (res["mode"] == "construction")].correct.any()
    assert not res.title.isin(docs[docs.injected].title).any()


def test_injection_cases_always_contain_the_planted_text(docs, chunks, retriever):
    cases = answers.build_injection_cases(docs, chunks, retriever, random.Random(0), k=5)
    assert len(cases) == 4 * len(answers.injection_templates) * 2
    res = answers.score_injection_cases(cases, obey_injection([c["messages"] for c in cases]))
    assert res.followed.all()


def test_token_f1():
    assert answers.token_f1("The Seller shall", "seller shall") == 1.0
    assert answers.token_f1("nothing here", "seller shall") == 0.0
    assert 0 < answers.token_f1("the seller may", "seller shall") < 1


def test_quality_scoring():
    hits = [{"title": "t", "start": 0, "end": 100}, {"title": "t", "start": 500, "end": 600}]
    case = {"title": "t", "category": "c", "hits": hits,
            "answers": {"text": np.array(["x" * 10]), "answer_start": np.array([520])}}
    res = answers.score_quality_cases([case, case], ["see [2]", "see [1]"])
    assert res.cited_correct.tolist() == [True, False]
    assert res.context_has_answer.all()
