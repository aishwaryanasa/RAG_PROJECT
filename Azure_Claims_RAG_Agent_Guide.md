# Building a Claims-Handler RAG Agent on Azure
### From first login to a RAGAS-evaluated, citation-grounded retrieval agent

**Scope covered:** Azure onboarding → data ingestion → layout-aware parsing/chunking → hybrid retrieval (BM25 + dense) → BGE-Reranker → grounded generation with inline citations → 50-question golden set → RAGAS evaluation.

**Use case framing:** ingest the policy wordings, endorsements, and claims FAQs (e.g., the specimen CA-PAP policy packets, endorsement forms, and SERFF rate/rule pages) so claim handlers get one grounded answer with a citation back to the exact form/page, instead of hunting across PDFs.

---

## Phase 0 — Azure account setup (first-time login)

1. **Create the account.** Go to `portal.azure.com`. If you don't have a tenant, sign up at `azure.microsoft.com/free` — this creates an Entra ID (Azure AD) tenant and a Pay-As-You-Go or free-tier subscription tied to it.
2. **First login.** Sign in with the Microsoft account you used to sign up. Azure will drop you into the **Azure Portal Home**. Confirm your subscription is active: **Subscriptions** (left search bar) → you should see one subscription with a status of *Active*.
3. **Set spending guardrails.** Go to **Cost Management + Billing → Budgets** → create a monthly budget with an alert at 50/80/100%. RAG pipelines (embeddings + reranker + LLM calls during eval) can burn through credits fast during iteration.
4. **Request quota where needed.** Azure OpenAI is a gated service — go to **Azure AI Foundry** (`ai.azure.com`) and apply for access if your subscription doesn't already show it (usually instant for standard usage now, but check region availability for GPT-4o/4.1 and `text-embedding-3-large`).
5. **Install tooling locally:**
   ```bash
   # Azure CLI
   curl -L https://aka.ms/InstallAzureCli | bash
   az login
   az account set --subscription "<your-subscription-id>"
   ```
6. **Create a Resource Group** — everything below lives in one RG so you can tear it down cleanly:
   ```bash
   az group create --name rg-claims-rag --location eastus2
   ```
   (Use a region where Azure AI Search, Azure OpenAI, and Document Intelligence are all available — `eastus2`, `swedencentral`, or `westus3` are safe bets as of 2026.)

---

## Phase 1 — Provision the core resources

| Resource | Purpose | Portal blade |
|---|---|---|
| Storage Account | Raw PDF landing zone | Storage accounts |
| Azure AI Document Intelligence | Layout-aware PDF parsing (tables, headers, reading order) | AI Services → Document Intelligence |
| Azure AI Search | Hybrid BM25 + vector index | AI Search |
| Azure OpenAI (in AI Foundry) | Embeddings + grounded generation LLM | AI Foundry |
| Azure Container Apps (or Azure ML endpoint) | Host BGE-Reranker (not a native Azure PaaS model) | Container Apps |
| Azure AI Foundry Agent Service (or Functions) | Orchestration layer / the "agent" | AI Foundry / Functions |

CLI provisioning:

```bash
# Storage for raw + processed docs
az storage account create --name stclaimsragdocs --resource-group rg-claims-rag \
  --location eastus2 --sku Standard_LRS

az storage container create --account-name stclaimsragdocs --name raw-policies
az storage container create --account-name stclaimsragdocs --name parsed-chunks

# Document Intelligence (layout model)
az cognitiveservices account create --name di-claims-rag --resource-group rg-claims-rag \
  --kind FormRecognizer --sku S0 --location eastus2 --yes

# Azure AI Search (Standard tier — needed for vector + semantic features at volume)
az search service create --name srch-claims-rag --resource-group rg-claims-rag \
  --sku standard --location eastus2

# Azure OpenAI resource
az cognitiveservices account create --name aoai-claims-rag --resource-group rg-claims-rag \
  --kind OpenAI --sku S0 --location eastus2 --yes
```

Then in **AI Foundry Studio** (`ai.azure.com`), attach the AOAI resource to a project and deploy two models:
- `text-embedding-3-large` (dense retrieval)
- `gpt-4o` or `gpt-4.1` (grounded generation)

---

## Phase 2 — Ingestion + layout-aware parsing

**Upload raw documents** (the policy PDFs, endorsement forms, rate pages, FAQs) to the `raw-policies` container:
```bash
az storage blob upload-batch --account-name stclaimsragdocs \
  --destination raw-policies --source ./policy_pdfs/
```

**Parse with Document Intelligence's Layout model** — this is what gives you layout awareness: reading order, section headers, tables (e.g., your Coverage Schedule and Rate/Rule tables) preserved as structured cells rather than flattened text, and paragraph roles (`title`, `sectionHeading`, `pageFooter`, etc.).

```python
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

client = DocumentIntelligenceClient(
    endpoint="https://di-claims-rag.cognitiveservices.azure.com/",
    credential=AzureKeyCredential("<DI_KEY>")
)

with open("Full_03_2025_Tesla_Model3.pdf", "rb") as f:
    poller = client.begin_analyze_document("prebuilt-layout", body=f)
result = poller.result()

# result.paragraphs -> each has .role (title/sectionHeading/pageFooter/...),
#                      .content, and .bounding_regions (page number)
# result.tables     -> each has .cells with row_index/column_index — use this
#                      for Coverage Schedule, Rate/Rule, and Form Schedule tables
```

**Why this matters for your corpus specifically:** policy documents have three structurally distinct zones that a naive PDF-to-text dump destroys — (1) the SPECIMEN watermark/banner running through every page, which you must **strip**, not chunk; (2) tables (coverage schedules, rate factors) that must stay row-intact; (3) named sections (`Part A — Liability`, `Part D — Physical Damage`) that are the natural retrieval unit boundary.

---

## Phase 3 — Chunking strategy

Layout-aware chunking rules for this corpus:

1. **Strip boilerplate first**: regex out the repeating SPECIMEN banner text, footer, and watermark artifacts using the paragraph `role == "pageFooter"` / known banner strings — don't let them pollute embeddings.
2. **Chunk by section, not by fixed token count.** Use `sectionHeading` paragraphs as hard chunk boundaries: `PART A — LIABILITY COVERAGE`, `PART D — COVERAGE FOR DAMAGE TO YOUR AUTO`, each numbered endorsement provision block, each Rate/Rule page's named subsection.
3. **Keep tables atomic.** A Coverage Schedule table or an RR-06 relativity table becomes one chunk (serialized as markdown table), never split mid-table — splitting breaks the semantic unit a handler needs ("what's the Collision deductible for this vehicle").
4. **Target size:** 200–500 tokens per chunk for prose sections; tables are chunked as a whole regardless of token count (cap at ~800 tokens, split only on natural row groups if it exceeds that).
5. **Attach rich metadata to every chunk** — this is what makes inline citation possible later:
   ```json
   {
     "chunk_id": "Full_03_Tesla_Model3__PartD__c2",
     "document_name": "Full_03_2025_Tesla_Model3.pdf",
     "form_number": "CA-PAP-2026",
     "section": "Part D — Coverage for Damage to Your Auto",
     "page_number": 6,
     "doc_type": "policy_wording",           // policy_wording | endorsement | rate_rule | faq
     "vehicle": "2025 Tesla Model 3",
     "content": "...",
     "embedding": [...]
   }
   ```

Reference chunking implementation:
```python
import re

BANNER_PATTERNS = [
    r"SPECIMEN / SAMPLE DOCUMENT.*?REAL POLICY",
    r"SAMPLE\s*[—-]\s*NOT A BINDING CONTRACT",
]

def strip_boilerplate(text: str) -> str:
    for pat in BANNER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()

def chunk_by_section(paragraphs, tables, doc_meta):
    chunks, current, current_section = [], [], "General"
    for p in paragraphs:
        if p.role == "sectionHeading":
            if current:
                chunks.append(make_chunk(current, current_section, doc_meta))
            current_section, current = p.content, []
        elif p.role not in ("pageFooter", "pageHeader"):
            clean = strip_boilerplate(p.content)
            if clean:
                current.append(clean)
    if current:
        chunks.append(make_chunk(current, current_section, doc_meta))
    for t in tables:
        chunks.append(make_table_chunk(t, current_section, doc_meta))
    return chunks
```

---

## Phase 4 — Embed + index (Azure AI Search, hybrid BM25 + dense)

**Create the index** with both a searchable text field (for BM25/keyword) and a vector field (for dense retrieval):

```python
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SemanticConfiguration, SemanticPrioritizedFields, SemanticField, SemanticSearch
)

fields = [
    SimpleField(name="chunk_id", type="Edm.String", key=True),
    SearchableField(name="content", type="Edm.String", analyzer_name="en.microsoft"),
    SimpleField(name="document_name", type="Edm.String", filterable=True, facetable=True),
    SimpleField(name="form_number", type="Edm.String", filterable=True),
    SimpleField(name="section", type="Edm.String", filterable=True),
    SimpleField(name="page_number", type="Edm.Int32", filterable=True),
    SimpleField(name="doc_type", type="Edm.String", filterable=True, facetable=True),
    SearchField(name="embedding", type="Collection(Edm.Single)",
                searchable=True, vector_search_dimensions=3072,
                vector_search_profile_name="hnsw-profile"),
]

vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="hnsw-cfg")],
    profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw-cfg")]
)

semantic_search = SemanticSearch(configurations=[
    SemanticConfiguration(name="semantic-cfg", prioritized_fields=SemanticPrioritizedFields(
        content_fields=[SemanticField(field_name="content")]))
])

index = SearchIndex(name="claims-policy-index", fields=fields,
                     vector_search=vector_search, semantic_search=semantic_search)

SearchIndexClient(endpoint="https://srch-claims-rag.search.windows.net",
                   credential=AzureKeyCredential("<SEARCH_ADMIN_KEY>")).create_or_update_index(index)
```

**Embed and upload chunks** using `text-embedding-3-large` via your Azure OpenAI deployment, then push documents to the index (`SearchClient.upload_documents`).

**Query time — hybrid retrieval** (BM25 keyword + vector, fused by Azure AI Search's built-in RRF):
```python
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

results = search_client.search(
    search_text=user_query,                       # drives BM25/keyword leg
    vector_queries=[VectorizedQuery(vector=query_embedding, k_nearest_neighbors=25,
                                     fields="embedding")],  # dense leg
    query_type="semantic", semantic_configuration_name="semantic-cfg",
    top=25
)
```
This gives you native hybrid (BM25 + dense) fusion out of the box — Azure AI Search does the RRF merge for you.

---

## Phase 5 — BGE-Reranker (not a native Azure PaaS offering — host it yourself)

BGE-Reranker (BAAI `bge-reranker-large` / `bge-reranker-v2-m3`) isn't in the Azure AI model catalog as a managed endpoint by default, so host it as a container:

1. **Package it:**
   ```dockerfile
   FROM python:3.11-slim
   RUN pip install FlagEmbedding fastapi uvicorn torch --extra-index-url https://download.pytorch.org/whl/cpu
   COPY reranker_service.py .
   CMD ["uvicorn", "reranker_service:app", "--host", "0.0.0.0", "--port", "8080"]
   ```
   ```python
   # reranker_service.py
   from fastapi import FastAPI
   from FlagEmbedding import FlagReranker

   app = FastAPI()
   reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

   @app.post("/rerank")
   def rerank(payload: dict):
       query, passages = payload["query"], payload["passages"]  # list[str]
       scores = reranker.compute_score([[query, p] for p in passages], normalize=True)
       return {"scores": scores}
   ```
2. **Deploy to Azure Container Apps** (GPU-backed if you want low latency; CPU is workable for a 25-passage rerank step):
   ```bash
   az acr create --name acrclaimsrag --resource-group rg-claims-rag --sku Basic
   az acr build --registry acrclaimsrag --image bge-reranker:v1 .

   az containerapp env create --name env-claims-rag --resource-group rg-claims-rag --location eastus2

   az containerapp create --name bge-reranker --resource-group rg-claims-rag \
     --environment env-claims-rag --image acrclaimsrag.azurecr.io/bge-reranker:v1 \
     --target-port 8080 --ingress internal --min-replicas 1 --max-replicas 3 \
     --cpu 2 --memory 4Gi
   ```
   (Swap to a GPU-enabled Container Apps workload profile, or an Azure ML Managed Online Endpoint with a GPU SKU, if latency under load matters more than cost.)
3. **Pipeline order:** Azure AI Search hybrid retrieval returns top-25 → call the reranker service with `(query, chunk.content)` pairs → keep top 5–8 by reranker score → pass those to the LLM as grounding context.

---

## Phase 6 — Grounded generation with inline citations

Force citation discipline through the prompt contract and the context format — give the LLM chunks pre-labeled with a citation tag it must reuse verbatim:

```python
def build_context_block(reranked_chunks):
    return "\n\n".join(
        f"[{i+1}] (Source: {c['document_name']}, {c['section']}, p.{c['page_number']})\n{c['content']}"
        for i, c in enumerate(reranked_chunks)
    )

SYSTEM_PROMPT = """You are a claims-handler assistant. Answer ONLY using the numbered
sources below. Every factual claim must end with a bracketed citation like [2] matching
the source number. If the sources don't contain the answer, say so explicitly — never
guess a coverage limit, deductible, or exclusion. Do not merge facts from different
vehicles/policies unless the question asks for a comparison."""

response = aoai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"SOURCES:\n{build_context_block(reranked_chunks)}\n\nQUESTION: {user_query}"}
    ],
    temperature=0
)
```

Post-process the answer to turn `[2]` back into a clickable reference (`document_name`, `section`, `page_number`) in the UI, and **reject/flag** any answer containing an uncited sentence with a simple regex/NLI check before it reaches the handler.

---

## Phase 7 — Build the 50-question golden set

Draw questions directly from your corpus (the 9 vehicle policies, 6 endorsement forms, 8 rate/rule pages) so every question has a ground-truth answer and a known source chunk. Structure:

| # | Question | Expected Answer | Source doc/section | Type |
|---|---|---|---|---|
| 1 | What is the Collision deductible for the 2025 Tesla Model 3 policy? | $500 | Full_03…, Coverage Schedule | Factual lookup |
| 2 | Does the EV Battery endorsement cover gradual battery degradation? | No, explicitly excluded | CA-51 endorsement, Provisions #3 | Exclusion check |
| 3 | What Bodily Injury limits apply to the minimum-limits Civic policy? | $50,000/$100,000 | Full_04…, Coverage Schedule | Factual lookup |
| 4 | Is rideshare/TNC use covered under Part A? | No, excluded unless endorsed | Part A exclusions | Exclusion check |
| 5 | What's the Good Driver Discount factor for Collision? | 0.80 (20%) | RR-04 | Rate lookup |
| ... | ... | ... | ... | ... |

Cover these categories across the 50 (roughly 10 each):
1. **Direct factual lookups** (limits, deductibles, premiums per vehicle)
2. **Exclusion questions** ("is X covered") — these catch hallucinated coverage
3. **Cross-document comparison** ("which vehicles have OEM parts endorsements")
4. **Rate/rule reasoning** (how a factor is applied, e.g., ILF calculation)
5. **Negative/unanswerable questions** (ask about a coverage that doesn't exist in the corpus — tests whether the system correctly says "not found" instead of hallucinating)

Store as JSON/CSV with columns: `question, ground_truth_answer, ground_truth_chunk_ids, category`.

---

## Phase 8 — RAGAS evaluation

Install and wire RAGAS against Azure OpenAI as the judge model:

```bash
pip install ragas datasets langchain-openai
```

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

judge_llm = AzureChatOpenAI(azure_deployment="gpt-4o", api_version="2024-10-21")
judge_embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-large")

# Run your RAG pipeline over all 50 golden questions first, collecting:
# question, answer (generated), contexts (retrieved chunk texts), ground_truth
records = []
for row in golden_set:
    retrieved = run_hybrid_retrieval_and_rerank(row["question"])
    answer = generate_grounded_answer(row["question"], retrieved)
    records.append({
        "question": row["question"],
        "answer": answer,
        "contexts": [c["content"] for c in retrieved],
        "ground_truth": row["ground_truth_answer"],
    })

dataset = Dataset.from_list(records)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=judge_embeddings,
)
print(result)   # per-metric scores, 0–1
result.to_pandas().to_csv("ragas_eval_results.csv")
```

**What each metric tells you for this use case:**
- **Faithfulness** — did the answer only state what the retrieved chunks support? (catches invented coverage/limits)
- **Answer relevancy** — did it actually answer the handler's question, not a tangent?
- **Context precision** — did reranking put the *right* chunk near the top? (validates your BGE-Reranker is earning its keep over raw hybrid search)
- **Context recall** — did retrieval surface *all* chunks needed for a complete answer (important for multi-part questions like "what's covered and what's excluded")?

**Set a bar before go-live**, e.g., faithfulness ≥ 0.90, context recall ≥ 0.85 — anything below triggers a chunking/retrieval fix, not a prompt patch.

---

## Phase 9 — Wrap it as an agent + wire up monitoring

1. **Orchestration**: stand this up as a tool-calling agent in **Azure AI Foundry Agent Service** (or Semantic Kernel / LangGraph if you want more control), with the retrieval+rerank pipeline exposed as a single `search_policy_corpus(query)` tool the agent calls before answering.
2. **Guardrails**: add Azure AI Content Safety on both input and output; add a deterministic "no source found → say so" fallback rather than letting the LLM improvise.
3. **Monitoring**: log every query, retrieved chunk IDs, citations used, and RAGAS-style scores (sampled in production via Azure Monitor + Application Insights custom events) so you can catch drift as new policy editions/endorsements are ingested.
4. **Re-run the 50-question RAGAS suite** as a CI gate any time you change chunking, embeddings model, or the reranker — treat it like a regression test suite, not a one-time report.

---

### Summary of the pipeline shape
```
PDFs → Blob Storage → Document Intelligence (layout) → section/table-aware chunker
     → text-embedding-3-large → Azure AI Search (BM25 + vector index)
     → query: hybrid retrieval (top-25) → BGE-Reranker (top-8)
     → GPT-4o grounded generation with numbered citations
     → 50-question golden set → RAGAS (faithfulness/relevancy/precision/recall)
     → Agent Service wrapper + Content Safety + monitoring
```
