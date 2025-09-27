import pandas as pd
import heapq
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import re

# --- Text cleaning ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Load queries ---
# --- Load queries ---
def load_queries(filepath="own_spoken_queries.tsv"):
    queries = {}
    with open(filepath, encoding="utf-8") as f:
        next(f)  # skip header
        for line_num, line in enumerate(f, start=2):
            parts = line.strip().split("\t")
            if len(parts) >= 2:  # only qid and spoken_text
                qid = parts[0]
                query_text = clean_text(parts[1])  # apply cleaning if you want
                queries[qid] = query_text
            else:
                print(f"⚠️ Skipped line {line_num} due to missing fields: {parts}")
    return queries


# --- Main chunked retrieval + merging ---
def retrieve_with_chunking(collection_path="collection.tsv", chunk_size=100000):
    queries = load_queries()
    final_results = {qid: [] for qid in queries}

    print(f"📥 Loaded {len(queries)} queries.")
    print("🚀 Starting chunked retrieval...\n")

    chunk_iter = pd.read_csv(collection_path, sep="\t", header=None,
                             names=["pid", "passage"], chunksize=chunk_size, encoding="utf-8")

    for chunk_num, chunk in enumerate(tqdm(chunk_iter, desc="🔄 Processing chunks"), start=1):
        chunk["clean_passage"] = chunk["passage"].apply(clean_text) 
        chunk_pids = chunk["pid"].astype(str).tolist()
        corpus = chunk["clean_passage"].tolist()
        tokenized_corpus = [doc.split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        for qid, query in tqdm(queries.items(), desc=f"💬 Ranking queries (Chunk {chunk_num})", leave=False):
            tokenized_query = query.split()
            scores = bm25.get_scores(tokenized_query)
            top_docs = heapq.nlargest(50, enumerate(scores), key=lambda x: x[1])

            for rank, (idx, score) in enumerate(top_docs):
                pid = chunk_pids[idx]
                heapq.heappush(final_results[qid], (-score, pid))

    print("\n🧠 Selecting top 1000 results per query...")
    with open("bm25_own_spoken.trec", "w", encoding="utf-8") as fout:
        for qid in tqdm(queries, desc="✍️ Writing to TREC file"):
            top_1000 = heapq.nsmallest(1000, final_results[qid])
            for rank, (neg_score, docid) in enumerate(top_1000):
                fout.write(f"{qid} Q0 {docid} {rank+1} {-neg_score:.4f} BM25\n")

    print("\n✅ Done! File saved as: bm25_own_spoken.trec")

if __name__ == "__main__":
    retrieve_with_chunking()

