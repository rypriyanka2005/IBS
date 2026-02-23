from Bio import Entrez, SeqIO
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Setup
# ---------------------------------------------------------
Entrez.email = "harinimahalakshmi.rs@gmail.com"

# 2. Search NCBI for Spike protein (Homo sapiens)
# Increased retmax to 1500 as requested
# ---------------------------------------------------------
print("Searching NCBI...")
search_handle = Entrez.esearch(
    db="protein",
    term='SARS-CoV-2 spike protein AND "Homo sapiens"[host]',
    retmax=3000
)
search_results = Entrez.read(search_handle)
protein_ids = search_results["IdList"]

print("Total IDs fetched from NCBI:", len(protein_ids))

# 3. Fetch sequences and save RAW FASTA
# ---------------------------------------------------------
print("Fetching sequences from NCBI...")
fetch_handle = Entrez.efetch(
    db="protein",
    id=protein_ids,
    rettype="fasta",
    retmode="text"
)

raw_fasta_path = "covid_spike_raw.fasta"
with open(raw_fasta_path, "w") as f:
    f.write(fetch_handle.read())

print(f"Raw FASTA saved to {raw_fasta_path}")

# 4. Removing redundant sequences using Cosine Similarity
# ---------------------------------------------------------
print("Removing redundant sequences using cosine similarity...")

def kmers(seq, k=3):
    """Generates k-mers from a sequence to treat it like text."""
    return " ".join([seq[i:i+k] for i in range(len(seq)-k+1)])

# Load the raw records
records = list(SeqIO.parse(raw_fasta_path, "fasta"))
ids = [r.id for r in records]
seq_strings = [str(r.seq) for r in records]

# Convert sequences to k-mer text format for vectorization
kmer_texts = [kmers(seq) for seq in seq_strings]

# Vectorize the k-mer patterns
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(kmer_texts)

# Compute Similarity Matrix
similarity_matrix = cosine_similarity(X)

threshold = 0.99
keep_indices = []
removed = set()

# Iterative filtering based on similarity threshold
for i in range(len(ids)):
    if i in removed:
        continue
    keep_indices.append(i)
    for j in range(i+1, len(ids)):
        if similarity_matrix[i][j] >= threshold:
            removed.add(j)

unique_records = [records[i] for i in keep_indices]

print("--- Redundancy Statistics ---")
print("Raw sequences processed:", len(records))
print("Non-redundant sequences retained:", len(unique_records))
print("Duplicates/Highly similar removed:", len(records) - len(unique_records))

# 5. Save NON-REDUNDANT FASTA
# ---------------------------------------------------------
output_fasta = "covid_spike_nonredundant.fasta"
SeqIO.write(unique_records, output_fasta, "fasta")
print(f"Non-redundant FASTA saved to {output_fasta}")

# 6. Save NON-REDUNDANT CSV (ML-ready)
# ---------------------------------------------------------
data = []
for record in unique_records:
    data.append({
        "ID": record.id,
        "Sequence": str(record.seq),
        "Length": len(record.seq)
    })

df = pd.DataFrame(data)
output_csv = "covid_spike_nonredundant.csv"
df.to_csv(output_csv, index=False)

print(f"Non-redundant CSV saved to {output_csv}")
print("\nFirst 5 rows of final dataset:")
print(df.head())

# 7. Final Sequence Statistics
# ---------------------------------------------------------
metrics = {
    "Total Sequences": len(df),
    "Mean Length": df["Length"].mean(),
    "Max Length": df["Length"].max(),
    "Min Length": df["Length"].min()
}

print("\n--- Final Metrics ---")
for key, value in metrics.items():
    print(f"{key}: {round(value, 2)}")

# ---------------------------------------------------------
# 8. READ BLAST OUTPUT (outfmt 6)


print("\nReading BLAST results...\n")

columns = [
    "qseqid","sseqid","pident","length","mismatch",
    "gapopen","qstart","qend","sstart","send",
    "evalue","bitscore"
]

df_blast = pd.read_csv("spike_all_vs_all.txt", sep="\t", names=columns)

print("Total alignments (including self-hits):", len(df_blast))

# Remove self-hits
df_blast = df_blast[df_blast["qseqid"] != df_blast["sseqid"]]

print("Total alignments (excluding self-hits):", len(df_blast))


# ============================================
# 7. SAVE BLAST ALIGNMENTS TO EXCEL
# ============================================

df_blast.to_excel("spike_blast_alignment_scores.xlsx", index=False)
print("BLAST alignment scores saved to Excel.")


# ============================================
# 8. STATISTICS FROM BLAST OUTPUT
# ============================================

print("\n===== BLAST STATISTICS =====")

metrics = {
    "Mean % Identity": df_blast["pident"].mean(),
    "Median % Identity": df_blast["pident"].median(),
    "Std Dev % Identity": df_blast["pident"].std(),
    "Min % Identity": df_blast["pident"].min(),
    "Max % Identity": df_blast["pident"].max(),

    "Mean Bit Score": df_blast["bitscore"].mean(),
    "Median Bit Score": df_blast["bitscore"].median(),
    "Std Dev Bit Score": df_blast["bitscore"].std(),
    "Min Bit Score": df_blast["bitscore"].min(),
    "Max Bit Score": df_blast["bitscore"].max(),

    "Mean E-value": df_blast["evalue"].mean(),
    "Min E-value": df_blast["evalue"].min(),
    "Max E-value": df_blast["evalue"].max(),
}

for key, value in metrics.items():
    print(f"{key}: {round(value, 4)}")

print("\nPIPELINE COMPLETED SUCCESSFULLY USING BLAST")

# # ============================================
# 9. LOAD PROT-BERT MODEL (GPU OPTIMIZED)
# ============================================

print("\nLoading ProtBERT model...")

from transformers import BertModel, BertTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")

model = model.to(device)
model = model.half()   # Faster inference on GPU
model.eval()

print("ProtBERT loaded successfully.")

# ============================================
# 10. GENERATE BERT EMBEDDINGS
# ============================================

print("\nGenerating BERT embeddings (batched + GPU)...")

def preprocess_sequence(seq):
    seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X")
    return " ".join(list(seq))

sequences = df["Sequence"].tolist()
processed_sequences = [preprocess_sequence(seq) for seq in sequences]

batch_size = 16   # Increase to 32 if memory allows
data_loader = DataLoader(processed_sequences, batch_size=batch_size)

embeddings = []

with torch.no_grad():
    for batch in tqdm(data_loader):

        encoded_input = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512  # Reduced for speed (change to 1024 if needed)
        )

        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        output = model(**encoded_input)

        token_embeddings = output.last_hidden_state

        # Mean pooling
        mean_embedding = torch.mean(token_embeddings, dim=1)

        embeddings.extend(mean_embedding.cpu().numpy())

embeddings = np.array(embeddings)

print("Embedding shape:", embeddings.shape)
# ============================================
# 11. SAVE EMBEDDINGS AS CSV
# ============================================

print("\nSaving ProtBERT embeddings as CSV...")

import pandas as pd

# Create DataFrame
emb_df = pd.DataFrame(embeddings)

# Add sequence ID column (important for tracking)
emb_df.insert(0, "ID", df["ID"].values)

# Save to CSV
emb_df.to_csv("spike_protbert_embeddings.csv", index=False)

print("ProtBERT embeddings saved as spike_protbert_embeddings.csv")
print("File shape:", emb_df.shape)

# ============================================
# 12. COSINE SIMILARITY USING BERT
# ============================================

print("\nComputing cosine similarity (ProtBERT)...")

bert_similarity_matrix = cosine_similarity(embeddings)

np.save("spike_protbert_similarity.npy", bert_similarity_matrix)

print("ProtBERT similarity matrix saved.")

print("\n===== BERT SIMILARITY STATISTICS =====")
print("Mean Similarity:", round(np.mean(bert_similarity_matrix), 4))
print("Max Similarity:", round(np.max(bert_similarity_matrix), 4))
print("Min Similarity:", round(np.min(bert_similarity_matrix), 4))
