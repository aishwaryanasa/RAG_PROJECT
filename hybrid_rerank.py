# hybrid_rerank.py — hybrid BM25+dense retrieval, then BGE-Reranker re-ranking.
# Reuses embeddings.npy / faiss.index if already built (from hybrid_retrieval.py run).
import json, os, numpy as np, faiss
from rank_bm25 import BM25Okapi
from openai import AzureOpenAI
from FlagEmbedding import FlagReranker

AOAI_ENDPOINT = <>
AOAI_KEY = <>
AOAI_EMBED_DEPLOYMENT = "text-embedding-3-large"   # from earlier fix

aoai = AzureOpenAI(azure_endpoint=AOAI_ENDPOINT, api_key=AOAI_KEY, api_version="2024-10-21")

chunks = [json.loads(l) for l in open("chunks.jsonl")]
texts = [c["content"] for c in chunks]

# --- BM25 (keyword) index ---
tokenized = [t.lower().split() for t in texts]
bm25 = BM25Okapi(tokenized)

def embed_batch(batch):
    resp = aoai.embeddings.create(model=AOAI_EMBED_DEPLOYMENT, input=batch)
    return [d.embedding for d in resp.data]

# --- Dense (FAISS) index: reuse if already built, else build once ---
if os.path.exists("embeddings.npy") and os.path.exists("faiss.index"):
    print("Reusing existing embeddings.npy + faiss.index")
    embeddings = np.load("embeddings.npy")
    index = faiss.read_index("faiss.index")
else:
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

# --- BGE-Reranker (loads once, reused across queries) ---
print("Loading BGE-Reranker (first run downloads the model, ~1.1GB)...")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)


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


def rerank(query, candidates, keep_top=8):
    if not candidates:
        return []
    passages = [c["content"] for c in candidates]
    scores = reranker.compute_score([[query, p] for p in passages], normalize=True)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = s
    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:keep_top]


def search_and_rerank(query, retrieve_k=25, keep_top=8):
    candidates = hybrid_search(query, top_k=retrieve_k)
    return rerank(query, candidates, keep_top=keep_top)


if __name__ == "__main__":
    query = "What is the Collision deductible for the Tesla Model 3?"
    results = search_and_rerank(query)
    print(f"\nQuery: {query}\n")
    print(f"{'rerank':>7}  {'fusion':>7}  document — section")
    print("-" * 70)
    for r in results:
        print(f"{r['rerank_score']:7.4f}  {r['fusion_score']:7.4f}  {r['document_name']} — {r['section']}")

# --- Phase 6: grounded generation with citations ---
def build_context_block(reranked_chunks):
    return "\n\n".join(
        f"[{i+1}] (Source: {c['document_name']}, {c['section']}, p.{c['page_number']})\n{c['content']}"
        for i, c in enumerate(reranked_chunks)
    )

SYSTEM_PROMPT = """You are a claims-handler assistant. Answer ONLY using the numbered
sources below. Every factual claim must end with a bracketed citation like [2] matching
the source number. If the sources don't contain the answer, say so explicitly — never
guess a coverage limit, deductible, or exclusion."""

def generate_grounded_answer(query, reranked_chunks):
    context = build_context_block(reranked_chunks)
    response = aoai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"SOURCES:\n{context}\n\nQUESTION: {query}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content
