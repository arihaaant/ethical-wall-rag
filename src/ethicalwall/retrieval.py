from qdrant_client import QdrantClient, models


class Index:
    def __init__(self, chunks, encoder, batch_size=64):
        self.encoder = encoder
        emb = encoder.encode([f"{c['title']}\n{c['text']}" for c in chunks],
                             batch_size=batch_size, normalize_embeddings=True)
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
        qv = self.encoder.encode(q, normalize_embeddings=True).tolist()
        return self.qc.query_points("contracts", query=qv, query_filter=flt, limit=k).points

    def search_postfilter(self, q, user, k=5):
        # naive version: global top k, then drop what the user can't see
        return [h for h in self.search(q, None, k) if user in h.payload["acl"]]
