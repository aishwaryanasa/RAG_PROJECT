# -*- coding: utf-8 -*-
"""
Embeds chunks.jsonl (produced by build_chunks.py over the 29-document corpus)
and uploads them to the Azure AI Search index defined in Phase 4 of the guide.

Fill in the placeholders below with your resource values, then run:
    pip install azure-search-documents azure-identity openai
    python upload_to_azure_search.py
"""
import json, os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# --- fill these in ---
SEARCH_ENDPOINT = "https://<your-search-service>.search.windows.net"
SEARCH_KEY = "<SEARCH_ADMIN_KEY>"
SEARCH_INDEX = "claims-policy-index"        # matches the schema in Phase 4
AOAI_ENDPOINT = "https://<your-aoai-resource>.openai.azure.com/"
AOAI_KEY = "<AOAI_KEY>"
AOAI_EMBED_DEPLOYMENT = "text-embedding-3-large"
CHUNKS_PATH = "chunks.jsonl"
# ---------------------

aoai = AzureOpenAI(azure_endpoint=AOAI_ENDPOINT, api_key=AOAI_KEY, api_version="2024-10-21")
search_client = SearchClient(endpoint=SEARCH_ENDPOINT, index_name=SEARCH_INDEX,
                              credential=AzureKeyCredential(SEARCH_KEY))

def embed_batch(texts):
    resp = aoai.embeddings.create(model=AOAI_EMBED_DEPLOYMENT, input=texts)
    return [d.embedding for d in resp.data]

def load_chunks(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def main():
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks")

    batch_size = 16
    docs_to_upload = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeddings = embed_batch([c["content"] for c in batch])
        for c, emb in zip(batch, embeddings):
            docs_to_upload.append({
                "chunk_id": c["chunk_id"],
                "content": c["content"],
                "document_name": c["document_name"],
                "form_number": c.get("form_number") or "",
                "section": c["section"],
                "page_number": c["page_number"],
                "doc_type": c["doc_type"],
                "embedding": emb,
            })
        print(f"Embedded {min(i+batch_size, len(chunks))}/{len(chunks)}")

    # upload in batches of 100 (Azure AI Search limit-friendly)
    for i in range(0, len(docs_to_upload), 100):
        result = search_client.upload_documents(documents=docs_to_upload[i:i+100])
        failed = [r for r in result if not r.succeeded]
        print(f"Uploaded batch {i//100 + 1}: {len(result)-len(failed)} ok, {len(failed)} failed")

    print("Done.")

if __name__ == "__main__":
    main()
