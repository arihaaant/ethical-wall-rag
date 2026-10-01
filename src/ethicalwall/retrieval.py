from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models


class Index:
    def __init__(self, chunks, encoder, batch_size=64, query_prefix="", cache=None):
        self.encoder = encoder
        self.query_prefix = query_prefix  # bge models want an instruction in front of queries
        if cache is not None and Path(cache).exists():
            emb = np.load(cache)
        else:
            emb = encoder.encode([f"{c['title']}\n{c['text']}" for c in chunks],
                                 batch_size=batch_size, normalize_embeddings=True)
            if cache is not None:
                np.save(cache, emb)
        self.qc = QdrantClient(":memory:")
        self.qc.create_collection("contracts", vectors_config=models.VectorParams(
            size=emb.shape[1], distance=models.Distance.COSINE))
        self.qc.upload_points("contracts", points=[
            models.PointStruct(id=i, vector=emb[i].tolist(), payload=c) for i, c in enumerate(chunks)])

    def search(self, q, user=None, k=5):
        # user=None is the unfiltered baseline, otherwise the acl filter is part of the vector query
        flt = None
        if user is not None:
            flt = models.Filter(must=[models.FieldCondition(key="acl", match=models.MatchValue(value=user))])
        qv = self.encoder.encode(self.query_prefix + q, normalize_embeddings=True).tolist()
        return self.qc.query_points("contracts", query=qv, query_filter=flt, limit=k).points

    def search_postfilter(self, q, user, k=5):
        # naive version: global top k, then drop what the user can't see
        return [h for h in self.search(q, None, k) if user in h.payload["acl"]]


class Reranked:
    # dense top n candidates (already acl filtered), reordered by a cross encoder
    def __init__(self, index, cross_encoder, candidates=50):
        self.index = index
        self.ce = cross_encoder
        self.candidates = candidates

    def search(self, q, user=None, k=5):
        hits = self.index.search(q, user, self.candidates)
        if not hits:
            return hits
        # same "title + text" the dense index embeds, otherwise it over-rewards first pages that repeat the title
        scores = self.ce.predict([(q, f"{h.payload['title']}\n{h.payload['text']}") for h in hits])
        return [hits[i] for i in np.argsort(-np.asarray(scores))[:k]]
