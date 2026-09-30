import json, numpy as np, faiss
from rank_bm25 import BM25Okapi
from openai import AzureOpenAI

AOAI_ENDPOINT = <>
AOAI_KEY = <>
aoai = AzureOpenAI(azure_endpoint=AOAI_ENDPOINT, api_key=AOAI_KEY, api_version="2024-10-21")

chunks = [json.loads(l) for l in open("chunks.jsonl")]
texts = [c["content"] for c in chunks]

tokenized = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized)

def embed_batch(batch):
    resp = aoai.embeddings.create(model="text-embedding-3-large", input=batch)
    return [d.embedding for d in resp.data]

print("Embedding all chunks (one-time)...")
all_embeddings = []
for i in range(0, len(texts), 16):
    all_embeddings.extend(embed_batch(texts[i:i+16]))
    print(f"  {min(i+16, len(texts))}/{len(texts)}")

embeddings = np.array(all_embeddings, dtype="float32")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

np.save("embeddings.npy", embeddings)
faiss.write_index(index, "faiss.index")
print("Saved embeddings.npy + faiss.index")

def hybrid_search(query, top_k=25):
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k]

    q_emb = np.array(embed_batch([query]), dtype="float32")
    faiss.normalize_L2(q_emb)
    _, dense_top = index.search(q_emb, top_k)
    dense_top = dense_top[0]

    rrf_scores = {}
    for rank, idx in enumerate(bm25_top):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank)
    for rank, idx in enumerate(dense_top):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{**chunks[idx], "fusion_score": score} for idx, score in ranked]

if __name__ == "__main__":
    results = hybrid_search("What is the Collision deductible for the Tesla Model 3?")
    for r in results[:5]:
        print(f"{r['fusion_score']:.4f}  {r['document_name']} — {r['section']}")
