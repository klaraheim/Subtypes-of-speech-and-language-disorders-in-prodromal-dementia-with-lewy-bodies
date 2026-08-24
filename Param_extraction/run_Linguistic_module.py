import os
import csv
import math

from Linguistic_module import (
    Lex_parameters,
    Semantic_parameters,
    Syntax_parameters,
)
# Cesta ########################################################
base_dir = r"E:/diplomka/DLBNLP"

# CSV nazev ########################################################
output_csv = "Linguistic_parameters.csv"

# Sloupce ########################################################
csv_columns = [
    "Subject",

    # Lex
    "Lex_total_tokens",
    "Lex_unique_types",
    "Lex_TTR",
    "Lex_repetition_ratio",
    "Lex_hapax_ratio",
    "Lex_HDD",

    # Sem
    "Sem_personal_ratio",
    "Sem_vague_ratio",
    "Sem_personal_count",
    "Sem_vague_count",
    "Sem_semantic_category_count",
    "Sem_content_density",
    "Sem_n_content",
    "Sem_n_tokens",

    # Syntax
    "Syn_num_sent",
    "Syn_MLU",
    "Syn_sub_per_sent",
    "Syn_total_sub",
    "Syn_hes",
    "Syn_hes_per_min",

    # transcript
    "Transcript",
]

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()

task_nmbr = "1"  

for prsnl_ID in os.listdir(base_dir):
    subject_dir = os.path.join(base_dir, prsnl_ID)
    if not os.path.isdir(subject_dir):
        continue

    file_name = f"{prsnl_ID}_CZ-AZV-TSK{task_nmbr}_1.wav"
    wav_path = os.path.join(subject_dir, file_name)

    if not os.path.exists(wav_path):
        print(f"[SKIP] Nenalezen soubor pro 🧐 {prsnl_ID}: {wav_path}")
        continue

    print(f"[OK!] zpracovávám 🤓 {prsnl_ID}")

# Lex ########################################################
    (
        total_tokens,
        unique_types,
        ttr,
        repetition_ratio,
        hapax_ratio,
        hdd_value,
        transcript,
    ) = Lex_parameters(prsnl_ID, task_nmbr)

# Sem ########################################################
    (
        personal_ratio,
        vague_ratio,
        personal_count,
        vague_count,
        semantic_category_count,
        content_density,
        n_content,
        n_tokens,
    ) = Semantic_parameters(transcript)

# Syntax ########################################################
    (
        num_sent,
        mlu,
        sub_per_sent,
        total_sub,
        hes,
        hes_per_min,
    ) = Syntax_parameters(transcript, prsnl_ID, task_nmbr)

# CSV architektura ########################################################
    row = {
        "Subject": prsnl_ID,

        "Lex_total_tokens": total_tokens,
        "Lex_unique_types": unique_types,
        "Lex_TTR": ttr,
        "Lex_repetition_ratio": repetition_ratio,
        "Lex_hapax_ratio": hapax_ratio,
        "Lex_HDD": hdd_value,

        "Sem_personal_ratio": personal_ratio,
        "Sem_vague_ratio": vague_ratio,
        "Sem_personal_count": personal_count,
        "Sem_vague_count": vague_count,
        "Sem_semantic_category_count": semantic_category_count,
        "Sem_content_density": content_density,
        "Sem_n_content": n_content,
        "Sem_n_tokens": n_tokens,

        "Syn_num_sent": num_sent,
        "Syn_MLU": mlu,
        "Syn_sub_per_sent": sub_per_sent,
        "Syn_total_sub": total_sub,
        "Syn_hes": hes,
        "Syn_hes_per_min": hes_per_min,

        "Transcript": transcript,
    }

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writerow(row)

print("Hotovo! ✨ výsledky jsou v:", output_csv)
