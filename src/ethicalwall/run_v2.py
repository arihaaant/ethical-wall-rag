import argparse
import gc
import json
import random
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import answers, data, evaluate
from .llm import vllm_generator
from .retrieval import Index, Reranked

bge_prefix = "Represent this sentence for searching relevant passages: "


def rates(frame, col):
    return evaluate.wilson(frame[col].sum(), len(frame)) if len(frame) else None


def citation_summary(cit):
    meta = cit.category.isin(evaluate.metadata_categories)
    return {"all": rates(cit, "hit"), "clauses": rates(cit[~meta], "hit"),
            "metadata": rates(cit[meta], "hit"), "top1": rates(cit, "top1")}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=int, default=None)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-citation", type=int, default=1000)
    p.add_argument("--n-quality", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--out", default="results/v2")
    args = p.parse_args(argv)
    seed, k = args.seed, args.k
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data.check_walls()
    df = data.load_cuad(("train", "test"))
    docs = data.plant_markers(data.sample_docs(df, args.docs, seed), random.Random(seed))
    chunks = data.chunk(docs)
    print(f"{len(docs)} contracts, {len(chunks)} chunks, {docs.injected.sum()} with a planted instruction")

    half = {"torch_dtype": torch.float16} if torch.cuda.is_available() else {}
    minilm = Index(chunks, SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"))
    bge = Index(chunks, SentenceTransformer("BAAI/bge-large-en-v1.5", model_kwargs=half), query_prefix=bge_prefix)
    rerank = Reranked(bge, CrossEncoder("BAAI/bge-reranker-large", model_kwargs=half))

    retrieval = {}
    for name, r in [("minilm", minilm), ("bge", bge), ("bge_rerank", rerank)]:
        for query in ("category", "question"):
            cit = evaluate.run_citation(r, df, docs, random.Random(seed), args.n_citation, k, seed, query)
            cit.to_csv(out / f"citation_{name}_{query}.csv", index=False)
            retrieval[f"{name}/{query}"] = citation_summary(cit)
            print(name, query, f"hit@{k} {retrieval[f'{name}/{query}']['all']['rate']:.3f}")

    rt = evaluate.run_redteam(bge, evaluate.build_redteam(docs, random.Random(seed)), k)
    rt.to_csv(out / "redteam.csv", index=False)

    code_cases = answers.build_code_cases(docs, rerank, random.Random(seed), k)
    inj_cases = answers.build_injection_cases(docs, chunks, rerank, random.Random(seed), k)
    qual_cases = answers.build_quality_cases(df, docs, rerank, random.Random(seed), args.n_quality, k, seed)

    # free the retrieval models before vllm takes the gpu
    del minilm, bge, rerank
    gc.collect()
    torch.cuda.empty_cache()

    generate = vllm_generator(args.llm)
    convs = [c["messages"] for c in code_cases + inj_cases + qual_cases]
    outs = generate(convs)
    a, b = len(code_cases), len(code_cases) + len(inj_cases)
    code = answers.score_code_cases(code_cases, outs[:a], docs)
    inj = answers.score_injection_cases(inj_cases, outs[a:b])
    qual = answers.score_quality_cases(qual_cases, outs[b:])
    code.to_csv(out / "access_control.csv", index=False)
    inj.to_csv(out / "injection.csv", index=False)
    qual.to_csv(out / "answer_quality.csv", index=False)

    access = {}
    for mode, g in code.groupby("mode"):
        forb, vis = g[~g.visible], g[g.visible]
        access[mode] = {
            "leak": rates(forb, "leak"),
            "leak_when_exposed": rates(forb[forb.exposed], "leak"),
            "forbidden_code_in_context": rates(forb, "exposed"),
            "correct_on_allowed": rates(vis, "correct"),
            "leak_by_template": {t: rates(forb[forb.q.str.startswith(t.split("{")[0])], "leak")
                                 for t in answers.code_templates},
        }

    summary = {
        "llm": args.llm,
        "contracts": len(docs),
        "chunks": len(chunks),
        "k": k,
        "retrieval": retrieval,
        "redteam_bge": {m: rates(rt, f"leak_{m}") for m in ("unfiltered", "postfilter", "filtered")}
                       | {"avg_returned_postfilter": rt.n_returned_postfilter.mean(),
                          "avg_returned_filtered": rt.n_returned_filtered.mean()},
        "access_control": access,
        "injection": {("hardened" if h else "plain"): rates(g, "followed") for h, g in inj.groupby("harden")}
                     | {"planted_chunk_retrieved": rates(inj[~inj.harden], "retrieved")},
        "answer_quality": {
            "token_f1": qual.f1.mean(),
            "context_has_answer": rates(qual, "context_has_answer"),
            "cited_correct": rates(qual, "cited_correct"),
            "cited_correct_when_available": rates(qual[qual.context_has_answer], "cited_correct"),
            "dont_know": rates(qual, "dont_know"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
