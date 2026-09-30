# ethical-wall-rag

[![CI](https://github.com/arihaaant/ethical-wall-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/arihaaant/ethical-wall-rag/actions/workflows/ci.yml)

Permission-aware retrieval over legal contracts. Law firms put "ethical walls" between teams working for clients with conflicting interests, and a retrieval system has to respect them. Here the wall is enforced inside the vector search itself, and a red-team set checks whether anything leaks.

## setup

- 60 contracts from [CUAD](https://www.atticusprojectai.org/cuad), split into 4,108 chunks of 1,000 characters (800 stride)
- each contract is assigned to one of 5 clients: A, B, C, D, and E (shared). Walls sit between A/B and C/D
- 4 users. `user_A` can see A and E, and so on. No user can see both sides of a wall (checked at startup)
- the ACL is stored on every chunk and passed as a Qdrant payload filter in the query, so unauthorized chunks are never candidates
- embeddings: `all-MiniLM-L6-v2`, cosine similarity, k = 5

Three retrieval modes are compared:

- **unfiltered**: plain top 5, no access control (the baseline)
- **post-filter**: plain top 5, then drop what the user can't see
- **in-query filter**: the ACL filter is part of the vector search

## results

### leakage

288 red-team queries: for every user, one direct and one multi-hop question about every contract they are not allowed to see. Direct questions name the forbidden contract, some with injection-style phrasing ("Ignore access restrictions and...", "I'm an admin..."). Multi-hop questions ask to compare an allowed contract with a forbidden one. A leak is any query where at least one returned chunk belongs to a client the user can't see. That check uses the client mapping, not the ACL field the filter reads.

| mode | queries with a leak | 95% CI | avg results returned (of 5) |
|---|---|---|---|
| unfiltered | 219 / 288 (76.0%) | 70.8 to 80.6% | 5.00 |
| post-filter | 0 / 288 | 0 to 1.3% | 1.88 |
| in-query filter | 0 / 288 | 0 to 1.3% | 5.00 |

Both filters stop every leak, but post-filtering throws away most of the results. For direct questions it returns 0.26 chunks on average, so the user mostly gets nothing back. An empty result also hints that something they can't see matched. The in-query filter always fills all 5 slots from documents the user is allowed to see.

Unfiltered, every direct question leaks (100%), and about half of the multi-hop ones do (50% for non-wall clients, 56% across a wall).

### citation quality

500 benign questions from CUAD's labeled answers, each asked by a user who is allowed to see that contract. A hit means one of the top 5 chunks comes from the right contract and overlaps the labeled answer span.

| questions | hit@5 | 95% CI | top-1 |
|---|---|---|---|
| all | 174 / 500 (34.8%) | 30.8 to 39.1% | 14.0% |
| clause categories | 163 / 347 (47.0%) | 41.8 to 52.2% | |
| metadata (parties, dates, document name) | 11 / 153 (7.2%) | 4.1 to 12.4% | |

The overall number is dragged down by metadata fields. "Parties" alone is 100 of the 500 questions, and their answers are short names scattered through the contract, so a query like "Parties clause in X" is a poor fit for dense retrieval. Actual clauses do much better (Insurance 90%, Uncapped Liability 91%, Governing Law 61%). Per-category numbers are in `results/summary.json`.

## running it

```bash
pip install -e ".[dev]"
python -m ethicalwall.run --out results
pytest
```

Runs on CPU in about 10 minutes. The results in this README are the files in `results/` (seed 0).

## limitations

- This covers retrieval only, with no LLM answering on top. Since restricted chunks never reach the context, a generation step can't leak them, but answer quality isn't measured here.
- Clients are assigned to contracts round-robin, not from real conflict data.
- The retriever is a small general-purpose embedding model with fixed-size chunks. A legal-domain model, hybrid BM25 search, or using CUAD's full question text would likely raise citation hit rates, especially for metadata fields.
- The in-memory Qdrant client doesn't use payload indexes, so this says nothing about filtered-search latency at scale.
