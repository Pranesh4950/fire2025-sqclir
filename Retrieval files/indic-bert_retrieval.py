import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import torch
import os


# File paths
# File paths
corpus_file = "collection_cleaned_full.tsv"
index_file = "output/faiss_index_labse.index"
pid_file = "output/pid_mapping.npy"
checkpoint_file = "output/checkpoint.txt"

# Create output folder if not exists
os.makedirs("output", exist_ok=True)

# Create output folder if not exists
#os.makedirs("/output", exist_ok=True)

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load LaBSE model
model = SentenceTransformer("l3cube-pune/indic-sentence-bert-nli", device=device)

# FAISS index initialization
batch_size = 256
dimension = 768  # LaBSE output dimension
index = faiss.IndexFlatIP(dimension)  # Using cosine similarity (with normalized vectors)
pid_list = []

# Load checkpoint (if exists)
def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return int(f.read().strip())
    return 0

# Save checkpoint
def save_checkpoint(chunk_num):
    with open(checkpoint_file, 'w') as f:
        f.write(str(chunk_num))

# Load existing PID mapping if available (for resume)
if os.path.exists(pid_file):
    pid_list = list(np.load(pid_file))

# Load existing index if available
if os.path.exists(index_file):
    print("🔄 Loading existing index to resume...")
    index = faiss.read_index(index_file)

# Resume from checkpoint
start_chunk = load_checkpoint()

print(f"▶️ Starting from chunk {start_chunk}...")

# Process entire corpus in chunks
with pd.read_csv(corpus_file, sep="\t", chunksize=100_000,
                 names=["pid", "passage", "clean_passage"],
                 skiprows=1, dtype={"pid": str}) as reader:

    for chunk_num, chunk in enumerate(reader):
        if chunk_num < start_chunk:
            continue  # Skip already processed chunks

        print(f"\n🔄 Processing chunk {chunk_num} (100,000 passages)")

        passages = chunk["passage"].tolist()
        pids = chunk["pid"].tolist()

        embeddings = model.encode(
            passages,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
            device=device
        )

        index.add(embeddings)
        pid_list.extend(pids)

        # Save index, pid mapping, and checkpoint
        faiss.write_index(index, index_file)
        np.save(pid_file, np.array(pid_list))
        save_checkpoint(chunk_num + 1)

        print(f"✅ Saved index and checkpoint after chunk {chunk_num}")

print("🎉 Full indexing completed.")








# Load LaBSE model
model = SentenceTransformer("l3cube-pune/indic-sentence-bert-nli")

# Load filtered queries
filtered_queries_df = pd.read_csv("/home/bharathi/Desktop/SqClir/spoken_queries_cleaned.tsv", sep="\t", names=["qid", "spoken", "clean"])

# Load FAISS index and pid mapping
index = faiss.read_index("output/faiss_index_labse.index")
pid_mapping = np.load("output/pid_mapping.npy")

# Encode queries
query_texts = filtered_queries_df["spoken"].tolist()
query_embeddings = model.encode(query_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

# Retrieve top-k results
top_k = 1000
D, I = index.search(query_embeddings, top_k)

# Write .trec file with actual similarity scores (not ranks)
with open("output/given_text.trec", "w") as f:
    for query_index, (query_id, retrieved_indices) in enumerate(zip(filtered_queries_df["qid"], I)):
        for rank, passage_index in enumerate(retrieved_indices):
            passage_pid = pid_mapping[passage_index]
            score = D[query_index][rank]  # actual cosine similarity
            f.write(f"{query_id} Q0 {passage_pid} {rank+1} {score} model-name\n")

print("✅ Retrieval complete.")







