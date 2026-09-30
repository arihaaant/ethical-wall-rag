import re

from datasets import load_dataset

clients = ["A", "B", "C", "D", "E"]  # E is shared / neutral
walls = [("A", "B"), ("C", "D")]
partner = {"A": "B", "B": "A", "C": "D", "D": "C"}
users = {f"user_{c}": {c, "E"} for c in "ABCD"}


def category_from_id(qid):
    # ids look like "<title>__Anti-Assignment_3", the trailing number is not part of the category
    return re.sub(r"_\d+$", "", qid.split("__")[-1])


def load_cuad():
    try:
        ds = load_dataset("theatticusproject/cuad-qa", split="train")
    except Exception:
        # newer versions of datasets refuse loading scripts, the parquet branch works
        ds = load_dataset("theatticusproject/cuad-qa", split="train", revision="refs/convert/parquet")
    df = ds.to_pandas()
    df["category"] = df["id"].map(category_from_id)
    return df


def sample_docs(df, n_docs, seed=0):
    docs = df.drop_duplicates("title")[["title", "context"]].sample(n_docs, random_state=seed)
    docs = docs.reset_index(drop=True)
    docs["client"] = [clients[i % len(clients)] for i in range(len(docs))]
    docs["acl"] = docs["client"].map(lambda c: [u for u, allowed in users.items() if c in allowed])
    return docs


def check_walls():
    for a, b in walls:
        for user, allowed in users.items():
            assert not {a, b} <= allowed, f"{user} can see both sides of wall {a}/{b}"


def chunk(docs, size=1000, stride=800):
    chunks = []
    for i, r in docs.iterrows():
        text = r.context
        for s in range(0, len(text), stride):
            chunks.append({"doc": i, "title": r.title, "start": s, "end": min(s + size, len(text)),
                           "text": text[s:s + size], "acl": r.acl, "client": r.client})
    return chunks
