import argparse
import gc
import json
import pickle
import random
from pathlib import Path

import pandas as pd
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


# every stage writes its output once. if the run dies, rerunning with the same --out picks up from there

def csv_stage(path, make):
    if path.exists():
        print("resuming from", path.name)
        return pd.read_csv(path)
    frame = make()
    frame.to_csv(path, index=False)
    return frame


def pickle_stage(path, make):
    if path.exists():
        print("resuming from", path.name)
        return pickle.loads(path.read_bytes())
    obj = make()
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(obj))
    tmp.replace(path)
    return obj


def generate_stage(folder, convs, make_generator, batch=512):
    # one file per batch, written to a temp name and renamed, so a kill mid-write never leaves half a batch
    folder.mkdir(exist_ok=True)
    generate = None
    for start in range(0, len(convs), batch):
        path = folder / f"{start:06d}.json"
        if path.exists():
            continue
        if generate is None:
            generate = make_generator()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(generate(convs[start:start + batch])))
        tmp.replace(path)
        print(f"generated {min(start + batch, len(convs))}/{len(convs)}")
    return [t for path in sorted(folder.glob("*.json")) for t in json.loads(path.read_text())]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=int, default=None)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-citation", type=int, default=1000)
    p.add_argument("--n-quality", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--retrievers", default="minilm,bge,bge_rerank")
    p.add_argument("--retrieval-only", action="store_true")
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
    loaded = {}

    def retriever(name):
        # built on first use, so a resumed run doesn't load models it no longer needs
        if name not in loaded:
            if name == "minilm":
                loaded[name] = Index(chunks, SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
                                     cache=out / "emb_minilm.npy")
            elif name == "bge":
                loaded[name] = Index(chunks, SentenceTransformer("BAAI/bge-large-en-v1.5", model_kwargs=half),
                                     query_prefix=bge_prefix, cache=out / "emb_bge.npy")
            else:
                loaded[name] = Reranked(retriever("bge"), CrossEncoder("BAAI/bge-reranker-large", model_kwargs=half))
        return loaded[name]

    retrieval = {}
    for name in args.retrievers.split(","):
        for query in ("category", "question"):
            cit = csv_stage(out / f"citation_{name}_{query}.csv", lambda: evaluate.run_citation(
                retriever(name), df, docs, random.Random(seed), args.n_citation, k, seed, query))
            retrieval[f"{name}/{query}"] = citation_summary(cit)
            print(name, query, f"hit@{k} {retrieval[f'{name}/{query}']['all']['rate']:.3f}")

    if args.retrieval_only:
        (out / "summary_retrieval.json").write_text(json.dumps(retrieval, indent=2, default=float))
        return

    rt = csv_stage(out / "redteam.csv", lambda: evaluate.run_redteam(
        retriever("bge"), evaluate.build_redteam(docs, random.Random(seed)), k))

    code_cases, inj_cases, qual_cases = pickle_stage(out / "cases.pkl", lambda: (
        answers.build_code_cases(docs, retriever("bge_rerank"), random.Random(seed), k),
        answers.build_injection_cases(docs, chunks, retriever("bge_rerank"), random.Random(seed), k),
        answers.build_quality_cases(df, docs, retriever("bge_rerank"), random.Random(seed), args.n_quality, k, seed),
    ))

    # free the retrieval models before vllm takes the gpu
    loaded.clear()
    gc.collect()
    torch.cuda.empty_cache()

    convs = [c["messages"] for c in code_cases + inj_cases + qual_cases]
    outs = generate_stage(out / "generations", convs, lambda: vllm_generator(args.llm))
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
        "injection": injection_summary(inj),
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


def injection_summary(inj):
    return ({("hardened" if h else "plain"): rates(g, "followed") for h, g in inj.groupby("harden")}
            | {("hardened_quoted" if h else "plain_quoted"): rates(g, "quoted") for h, g in inj.groupby("harden")}
            | {"planted_chunk_retrieved": rates(inj[~inj.harden], "retrieved")})


if __name__ == "__main__":
    main()
