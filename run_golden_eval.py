import csv, json
from hybrid_rerank import search_and_rerank, generate_grounded_answer

records = []
with open("golden_set_50.csv") as f:
    golden_rows = list(csv.DictReader(f))

print(f"Running pipeline over {len(golden_rows)} golden questions...")
for i, row in enumerate(golden_rows, 1):
    question = row["question"]
    ground_truth = row["ground_truth_answer"]

    retrieved = search_and_rerank(question, retrieve_k=25, keep_top=8)
    answer = generate_grounded_answer(question, retrieved)

    records.append({
        "question": question,
        "answer": answer,
        "contexts": [c["content"] for c in retrieved],
        "ground_truth": ground_truth,
        "category": row.get("category", ""),
    })
    print(f"  [{i}/{len(golden_rows)}] {question[:60]}...")

with open("eval_records.jsonl", "w") as out:
    for r in records:
        out.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"\nDone. Saved {len(records)} records to eval_records.jsonl")
