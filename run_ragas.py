import json
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

AOAI_ENDPOINT = "https://<your-aoai-resource>.openai.azure.com/"
AOAI_KEY = "<AOAI_KEY>"

judge_llm = AzureChatOpenAI(
    azure_endpoint=AOAI_ENDPOINT, api_key=AOAI_KEY,
    azure_deployment="gpt-4.1", api_version="2024-10-21"
)
judge_embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AOAI_ENDPOINT, api_key=AOAI_KEY,
    azure_deployment="text-embedding-3-large", api_version="2024-10-21"
)

records = [json.loads(l) for l in open("eval_records.jsonl")]
dataset = Dataset.from_list(records)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=judge_embeddings,
)

print(result)
df = result.to_pandas()
df.to_csv("ragas_eval_results.csv", index=False)
print("\nSaved detailed per-question scores to ragas_eval_results.csv")

print("\n--- Scores by category ---")
print(df.groupby("category")[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean())
