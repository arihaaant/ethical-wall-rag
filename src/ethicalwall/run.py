import argparse
import json
import random
from pathlib import Path

from sentence_transformers import SentenceTransformer

from . import data, evaluate
from .retrieval import Index


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--docs", type=int, default=60)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-citation", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data.check_walls()
    df = data.load_cuad()
    docs = data.sample_docs(df, args.docs, args.seed)
    chunks = data.chunk(docs)
    index = Index(chunks, SentenceTransformer(args.model))
    print(f"{len(docs)} contracts, {len(chunks)} chunks")

    rt = evaluate.run_redteam(index, evaluate.build_redteam(docs, rng), args.k)
    cit = evaluate.run_citation(index, df, docs, rng, args.n_citation, args.k, args.seed)
    rt.to_csv(out / "redteam.csv", index=False)
    cit.to_csv(out / "citation.csv", index=False)

    n = len(rt)
    is_meta = cit.category.isin(evaluate.metadata_categories)
    clauses, meta = cit[~is_meta], cit[is_meta]
    summary = {
        "contracts": len(docs),
        "chunks": len(chunks),
        "k": args.k,
        "redteam_queries": n,
        "leak_unfiltered": evaluate.wilson(rt.leak_unfiltered.sum(), n),
        "leak_postfilter": evaluate.wilson(rt.leak_postfilter.sum(), n),
        "leak_filtered": evaluate.wilson(rt.leak_filtered.sum(), n),
        "avg_returned_postfilter": rt.n_returned_postfilter.mean(),
        "avg_returned_filtered": rt.n_returned_filtered.mean(),
        "leak_by_kind": rt.groupby(["kind", "wall"])[["leak_unfiltered", "leak_filtered"]].mean()
                          .reset_index().to_dict("records"),
        "citation_queries": len(cit),
        "hit_at_k": evaluate.wilson(cit.hit.sum(), len(cit)),
        "top1": evaluate.wilson(cit.top1.sum(), len(cit)),
        "hit_at_k_clauses": evaluate.wilson(clauses.hit.sum(), len(clauses)),
        "hit_at_k_metadata": evaluate.wilson(meta.hit.sum(), len(meta)),
        "hit_by_category": cit.groupby("category").hit.agg(["mean", "count"]).sort_values("mean")
                              .reset_index().to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    for key in ["leak_unfiltered", "leak_postfilter", "leak_filtered", "hit_at_k", "top1",
                "hit_at_k_clauses", "hit_at_k_metadata"]:
        s = summary[key]
        print(f"{key:16s} {s['k']}/{s['n']} = {s['rate']:.3f}  (95% ci {s['ci_low']:.3f}-{s['ci_high']:.3f})")
    print(f"avg results returned: postfilter {summary['avg_returned_postfilter']:.2f}, "
          f"filtered {summary['avg_returned_filtered']:.2f}")


if __name__ == "__main__":
    main()
