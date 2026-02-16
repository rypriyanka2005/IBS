from Bio import Entrez, SeqIO
import pandas as pd


# 1. Setup

Entrez.email = "harinimahalakshmi.rs@gmail.com" 


# 2. Search NCBI for Spike protein (Homo sapiens)

search_handle = Entrez.esearch(
    db="protein",
    term='SARS-CoV-2 spike protein AND "Homo sapiens"[host]',
    retmax=1000
)
search_results = Entrez.read(search_handle)
protein_ids = search_results["IdList"]

print("Total IDs fetched from NCBI:", len(protein_ids))

# 3. Fetch sequences and save RAW FASTA
fetch_handle = Entrez.efetch(
    db="protein",
    id=protein_ids,
    rettype="fasta",
    retmode="text"
)

with open("covid_spike_raw.fasta", "w") as f:
    f.write(fetch_handle.read())

print("Raw FASTA saved")

# 4. Remove redundant sequences(Exact sequence duplicates)

unique_sequences = {}
raw_count = 0

for record in SeqIO.parse("covid_spike_raw.fasta", "fasta"):
    raw_count += 1
    seq = str(record.seq)

    # keep only first occurrence of each unique sequence
    if seq not in unique_sequences:
        unique_sequences[seq] = record

print("Raw sample count:", raw_count)
print("Non-redundant sample count:", len(unique_sequences))

# 5. Save NON-REDUNDANT FASTA

SeqIO.write(
    unique_sequences.values(),
    "covid_spike_nonredundant.fasta",
    "fasta"
)

print("Non-redundant FASTA saved")


# 6. Save NON-REDUNDANT CSV (ML-ready)

data = []

for record in unique_sequences.values():
    data.append({
        "ID": record.id,
        "Sequence": str(record.seq),
        "Length": len(record.seq)
    })

df = pd.DataFrame(data)
df.to_csv("covid_spike_nonredundant.csv", index=False)

print("Non-redundant CSV saved")

#7
input_fasta = "/content/covid_spike_raw.fasta"
records = list(SeqIO.parse(input_fasta, "fasta"))

print("Total sequences in RAW file:", len(records))



unique_sequences = {}
seen = set()

for record in records:
    seq = str(record.seq)
    if seq not in seen:
        seen.add(seq)
        unique_sequences[seq] = record

print("Non-redundant sequences:", len(unique_sequences))
print("Duplicates removed:", len(records) - len(unique_sequences))



SeqIO.write(unique_sequences.values(),
            "/content/covid_spike_nonredundant.fasta",
            "fasta")

print("Non-redundant FASTA saved")




sequences = [str(record.seq) for record in unique_sequences.values()]
ids = [record.id for record in unique_sequences.values()]



# 8. GLOBAL ALIGNMENT


print("\n================ GLOBAL ALIGNMENT (globalxx) ================\n")

alignment_data = []

for i in range(len(sequences)):
    for j in range(i + 1, len(sequences)):

        # Skip identical sequences
        if sequences[i] == sequences[j]:
            continue

        alignments = pairwise2.align.globalxx(sequences[i], sequences[j])

        for alignment in alignments[:1]:   # show only best alignment
            print(f"Alignment between {ids[i]} and {ids[j]}")
            print(format_alignment(*alignment))

            alignment_data.append({
                "Seq1_ID": ids[i],
                "Seq2_ID": ids[j],
                "Global_Score": alignment.score,
                "Local_Score": None
            })


# 9. LOCAL ALIGNMENT
=

print("\n================ LOCAL ALIGNMENT (localxx) ================\n")

for i in range(len(sequences)):
    for j in range(i + 1, len(sequences)):

        if sequences[i] == sequences[j]:
            continue

        alignments = pairwise2.align.localxx(sequences[i], sequences[j])

        for alignment in alignments[:1]:
            print(f"Alignment between {ids[i]} and {ids[j]}")
            print(format_alignment(*alignment))

            # update local score in dataframe
            for row in alignment_data:
                if row["Seq1_ID"] == ids[i] and row["Seq2_ID"] == ids[j]:
                    row["Local_Score"] = alignment.score




df = pd.DataFrame(alignment_data)
df.to_csv("/content/spike_alignment_scores.csv", index=False)

print("Alignment scores saved to CSV")
import pandas as pd
import numpy as np



file_path = "/content/spike_alignment_scores.csv"
df = pd.read_csv(file_path)

print("Total alignment comparisons:", len(df))
print("\nFirst 5 rows:")
print(df.head())



metrics = {}

# Global score statistics
metrics["Global Mean"] = df["Global_Score"].mean()
metrics["Global Median"] = df["Global_Score"].median()
metrics["Global Std Dev"] = df["Global_Score"].std()
metrics["Global Min"] = df["Global_Score"].min()
metrics["Global Max"] = df["Global_Score"].max()
metrics["Global Range"] = metrics["Global Max"] - metrics["Global Min"]

# Local score statistics
metrics["Local Mean"] = df["Local_Score"].mean()
metrics["Local Median"] = df["Local_Score"].median()
metrics["Local Std Dev"] = df["Local_Score"].std()
metrics["Local Min"] = df["Local_Score"].min()
metrics["Local Max"] = df["Local_Score"].max()
metrics["Local Range"] = metrics["Local Max"] - metrics["Local Min"]

for key, value in metrics.items():
    print(f"{key}: {round(value, 4)}")