import os
import csv
import math

import Acoustic_module as ac

# Cesta ########################################################
base_dir = r"E:/diplomka/DLBNLP"

# CSV nazev ########################################################
output_csv = "Acoustic_parameters.csv"

# Sloupce ########################################################
csv_columns = [
    "Subject",

    #Resp
    "RR_TSK2",
    "IPAmean_TSK2",
    "MPT_TSK6",

    #Fon
    "HNR_TSK3", "HNR_TSK4", "HNR_TSK5", "HNR_TSK6",
    "HRF_TSK3", "HRF_TSK4", "HRF_TSK5", "HRF_TSK6",
    "Jitter_TSK3", "Jitter_TSK4", "Jitter_TSK5", "Jitter_TSK6",
    "Shimmer_TSK3", "Shimmer_TSK4", "Shimmer_TSK5", "Shimmer_TSK6",
    "PPQ_TSK3", "PPQ_TSK4", "PPQ_TSK5", "PPQ_TSK6",
    "APQ_TSK3", "APQ_TSK4", "APQ_TSK5", "APQ_TSK6",
    "DF0_TSK3", "DF0_TSK4", "DF0_TSK5", "DF0_TSK6",
    "DDF0_TSK3", "DDF0_TSK4", "DDF0_TSK5", "DDF0_TSK6",
    "DUV_TSK3", "DUV_TSK4", "DUV_TSK5", "DUV_TSK6",

    #Art
    "RelF1SD_TSK1", "RelF1SD_TSK2", "RelF1SD_TSK3",
    "RelF1SD_TSK4", "RelF1SD_TSK5", "RelF1SD_TSK6",
    "RelF2SD_TSK1", "RelF2SD_TSK2", "RelF2SD_TSK3",
    "RelF2SD_TSK4", "RelF2SD_TSK5", "RelF2SD_TSK6",

    "VAI_TSK1", "VSHA_TSK1", "AAVS_TSK1",
    "VAI_TSK2", "VSHA_TSK2", "AAVS_TSK2",

    "PR_TSK7", "COV_TSK7", "PA_TSK7", "RI_TSK7", "RA_TSK7",

    #Proz
    "RelF0SD_TSK1", "RelF0SD_TSK2",
    "RelSE0SD_TSK1", "RelSE0SD_TSK2",
    "DurMED_TSK1", "DurMAD_TSK1", "SPIR_TSK1",
    "pause_n_TSK1", "pause_ratio_TSK1",
    "DurMED_TSK2", "DurMAD_TSK2", "SPIR_TSK2",
    "pause_n_TSK2", "pause_ratio_TSK2",
]

# CSV header ########################################################
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_columns)
    writer.writeheader()

# Loop pres subjekty ################################################
for prsnl_ID in os.listdir(base_dir):
    subject_dir = os.path.join(base_dir, prsnl_ID)
    if not os.path.isdir(subject_dir):
        continue

    print(f"[OK!] zpracovávám 🤓 {prsnl_ID}")

    path = subject_dir + os.sep

    # ---------- TSK1 ----------
    if os.path.exists(os.path.join(path, f"{prsnl_ID}_CZ-AZV-TSK1_1.wav")):
        # artikulace
        RelF1SD_TSK1, RelF2SD_TSK1 = ac.RelF1SD_RelF2SD(prsnl_ID, "1")
        VAI_TSK1, VSHA_TSK1, AAVS_TSK1 = ac.VAI_VSHA_AAVS(prsnl_ID, "1")

        # prozodie
        RelF0SD_TSK1, RelSE0SD_TSK1 = ac.RelF0SD_RelSE0SD(prsnl_ID, "1")
        DurMED_TSK1, DurMAD_TSK1, SPIR_TSK1 = ac.Pros_parameters(prsnl_ID, "1")
        pause_n_TSK1, pause_ratio_TSK1 = ac.Pause_metrics(prsnl_ID, "1")
    else:
        RelF1SD_TSK1 = RelF2SD_TSK1 = math.nan
        VAI_TSK1 = VSHA_TSK1 = AAVS_TSK1 = math.nan
        RelF0SD_TSK1 = RelSE0SD_TSK1 = math.nan
        DurMED_TSK1 = DurMAD_TSK1 = SPIR_TSK1 = math.nan
        pause_n_TSK1 = pause_ratio_TSK1 = math.nan

    # ---------- TSK2 ----------
    if os.path.exists(os.path.join(path, f"{prsnl_ID}_CZ-AZV-TSK2_1.wav")):
        # respirace
        RR_TSK2 = ac.Respiratory_Rate(prsnl_ID, "2")
        IPAmean_TSK2 = ac.IPA_mean(prsnl_ID, "2")

        # artikulace
        RelF1SD_TSK2, RelF2SD_TSK2 = ac.RelF1SD_RelF2SD(prsnl_ID, "2")
        VAI_TSK2, VSHA_TSK2, AAVS_TSK2 = ac.VAI_VSHA_AAVS(prsnl_ID, "2")

        # prozodie
        RelF0SD_TSK2, RelSE0SD_TSK2 = ac.RelF0SD_RelSE0SD(prsnl_ID, "2")
        DurMED_TSK2, DurMAD_TSK2, SPIR_TSK2 = ac.Pros_parameters(prsnl_ID, "2")
        pause_n_TSK2, pause_ratio_TSK2 = ac.Pause_metrics(prsnl_ID, "2")
    else:
        RR_TSK2 = IPAmean_TSK2 = math.nan
        RelF1SD_TSK2 = RelF2SD_TSK2 = math.nan
        VAI_TSK2 = VSHA_TSK2 = AAVS_TSK2 = math.nan
        RelF0SD_TSK2 = RelSE0SD_TSK2 = math.nan
        DurMED_TSK2 = DurMAD_TSK2 = SPIR_TSK2 = math.nan
        pause_n_TSK2 = pause_ratio_TSK2 = math.nan

    # ---------- TSK3 ----------
    if os.path.exists(os.path.join(path, f"{prsnl_ID}_CZ-AZV-TSK3_1.wav")):
        HNR_TSK3, PPQ_TSK3, APQ_TSK3 = ac.HNR_PPQ_APQ(prsnl_ID, "3")
        HRF_TSK3 = ac.HRF(prsnl_ID, "3")
        Jitter_TSK3, Shimmer_TSK3 = ac.jitter_shimmer(prsnl_ID, "3")
        DF0_TSK3, DDF0_TSK3 = ac.DF0_DDF0(prsnl_ID, "3")
        DUV_TSK3 = ac.DUV(prsnl_ID, "3")
        RelF1SD_TSK3, RelF2SD_TSK3 = ac.RelF1SD_RelF2SD(prsnl_ID, "3")
    else:
        HNR_TSK3 = HRF_TSK3 = math.nan
        Jitter_TSK3 = Shimmer_TSK3 = math.nan
        PPQ_TSK3 = APQ_TSK3 = math.nan
        DF0_TSK3 = DDF0_TSK3 = math.nan
        DUV_TSK3 = math.nan
        RelF1SD_TSK3 = RelF2SD_TSK3 = math.nan

    # ---------- TSK4 ----------
    if os.path.exists(os.path.join(path, f"{prsnl_ID}_CZ-AZV-TSK4_1.wav")):
        HNR_TSK4, PPQ_TSK4, APQ_TSK4 = ac.HNR_PPQ_APQ(prsnl_ID, "4")
        HRF_TSK4 = ac.HRF(prsnl_ID, "4")
        Jitter_TSK4, Shimmer_TSK4 = ac.jitter_shimmer(prsnl_ID, "4")
        DF0_TSK4, DDF0_TSK4 = ac.DF0_DDF0(prsnl_ID, "4")
        DUV_TSK4 = ac.DUV(prsnl_ID, "4")
        RelF1SD_TSK4, RelF2SD_TSK4 = ac.RelF1SD_RelF2SD(prsnl_ID, "4")
    else:
        HNR_TSK4 = HRF_TSK4 = math.nan
        Jitter_TSK4 = Shimmer_TSK4 = math.nan
        PPQ_TSK4 = APQ_TSK4 = math.nan
        DF0_TSK4 = DDF0_TSK4 = math.nan
        DUV_TSK4 = math.nan
        RelF1SD_TSK4 = RelF2SD_TSK4 = math.nan

    # ---------- TSK5 ----------
    if os.path.exists(os.path.join(path, f"{prsnl_ID}_CZ-AZV-TSK5_1.wav")):
        HNR_TSK5, PPQ_TSK5, APQ_TSK5 = ac.HNR_PPQ_APQ(prsnl_ID, "5")
        HRF_TSK5 = ac.HRF(prsnl_ID, "5")
        Jitter_TSK5, Shimmer_TSK5 = ac.jitter_shimmer(prsnl_ID, "5")
        DF0_TSK5, DDF0_TSK5 = ac.DF0_DDF0(prsnl_ID, "5")
        DUV_TSK5 = ac.DUV(prsnl_ID, "5")
        RelF1SD_TSK5, RelF2SD_TSK5 = ac.RelF1SD_RelF2SD(prsnl_ID, "5")
    else:
        HNR_TSK5 = HRF_TSK5 = math.nan
        Jitter_TSK5 = Shimmer_TSK5 = math.nan
        PPQ_TSK5 = APQ_TSK5 = math.nan
        DF0_TSK5 = DDF0_TSK5 = math.nan
        DUV_TSK5 = math.nan
        RelF1SD_TSK5 = RelF2SD_TSK5 = math.nan

    # ---------- TSK6 ----------
    if os.path.exists(os.path.join(path, f"{prsnl_ID}_CZ-AZV-TSK6_1.wav")):
        HNR_TSK6, PPQ_TSK6, APQ_TSK6 = ac.HNR_PPQ_APQ(prsnl_ID, "6")
        HRF_TSK6 = ac.HRF(prsnl_ID, "6")
        Jitter_TSK6, Shimmer_TSK6 = ac.jitter_shimmer(prsnl_ID, "6")
        DF0_TSK6, DDF0_TSK6 = ac.DF0_DDF0(prsnl_ID, "6")
        DUV_TSK6 = ac.DUV(prsnl_ID, "6")
        RelF1SD_TSK6, RelF2SD_TSK6 = ac.RelF1SD_RelF2SD(prsnl_ID, "6")
        MPT_TSK6 = ac.MPT(prsnl_ID, "6")
    else:
        HNR_TSK6 = HRF_TSK6 = math.nan
        Jitter_TSK6 = Shimmer_TSK6 = math.nan
        PPQ_TSK6 = APQ_TSK6 = math.nan
        DF0_TSK6 = DDF0_TSK6 = math.nan
        DUV_TSK6 = math.nan
        RelF1SD_TSK6 = RelF2SD_TSK6 = math.nan
        MPT_TSK6 = math.nan

    # ---------- TSK7 ----------
    if os.path.exists(os.path.join(path, f"{prsnl_ID}_CZ-AZV-TSK7_1.wav")):
        try:
            PR_TSK7, COV_TSK7, PA_TSK7, RI_TSK7, RA_TSK7 = ac.Art_parameters(prsnl_ID, "7")
        except FileNotFoundError:
            PR_TSK7 = COV_TSK7 = PA_TSK7 = RI_TSK7 = RA_TSK7 = math.nan
    else:
        PR_TSK7 = COV_TSK7 = PA_TSK7 = RI_TSK7 = RA_TSK7 = math.nan

    # CSV architektura ################################################
    row = {
        "Subject": prsnl_ID,

        "RR_TSK2": RR_TSK2,
        "IPAmean_TSK2": IPAmean_TSK2,

        "MPT_TSK6": MPT_TSK6,

        "HNR_TSK3": HNR_TSK3,
        "HNR_TSK4": HNR_TSK4,
        "HNR_TSK5": HNR_TSK5,
        "HNR_TSK6": HNR_TSK6,

        "HRF_TSK3": HRF_TSK3,
        "HRF_TSK4": HRF_TSK4,
        "HRF_TSK5": HRF_TSK5,
        "HRF_TSK6": HRF_TSK6,

        "Jitter_TSK3": Jitter_TSK3,
        "Jitter_TSK4": Jitter_TSK4,
        "Jitter_TSK5": Jitter_TSK5,
        "Jitter_TSK6": Jitter_TSK6,

        "Shimmer_TSK3": Shimmer_TSK3,
        "Shimmer_TSK4": Shimmer_TSK4,
        "Shimmer_TSK5": Shimmer_TSK5,
        "Shimmer_TSK6": Shimmer_TSK6,

        "PPQ_TSK3": PPQ_TSK3,
        "PPQ_TSK4": PPQ_TSK4,
        "PPQ_TSK5": PPQ_TSK5,
        "PPQ_TSK6": PPQ_TSK6,

        "APQ_TSK3": APQ_TSK3,
        "APQ_TSK4": APQ_TSK4,
        "APQ_TSK5": APQ_TSK5,
        "APQ_TSK6": APQ_TSK6,

        "DF0_TSK3": DF0_TSK3,
        "DF0_TSK4": DF0_TSK4,
        "DF0_TSK5": DF0_TSK5,
        "DF0_TSK6": DF0_TSK6,

        "DDF0_TSK3": DDF0_TSK3,
        "DDF0_TSK4": DDF0_TSK4,
        "DDF0_TSK5": DDF0_TSK5,
        "DDF0_TSK6": DDF0_TSK6,

        "DUV_TSK3": DUV_TSK3,
        "DUV_TSK4": DUV_TSK4,
        "DUV_TSK5": DUV_TSK5,
        "DUV_TSK6": DUV_TSK6,

        "RelF1SD_TSK1": RelF1SD_TSK1,
        "RelF1SD_TSK2": RelF1SD_TSK2,
        "RelF1SD_TSK3": RelF1SD_TSK3,
        "RelF1SD_TSK4": RelF1SD_TSK4,
        "RelF1SD_TSK5": RelF1SD_TSK5,
        "RelF1SD_TSK6": RelF1SD_TSK6,

        "RelF2SD_TSK1": RelF2SD_TSK1,
        "RelF2SD_TSK2": RelF2SD_TSK2,
        "RelF2SD_TSK3": RelF2SD_TSK3,
        "RelF2SD_TSK4": RelF2SD_TSK4,
        "RelF2SD_TSK5": RelF2SD_TSK5,
        "RelF2SD_TSK6": RelF2SD_TSK6,

        "VAI_TSK1": VAI_TSK1,
        "VSHA_TSK1": VSHA_TSK1,
        "AAVS_TSK1": AAVS_TSK1,

        "VAI_TSK2": VAI_TSK2,
        "VSHA_TSK2": VSHA_TSK2,
        "AAVS_TSK2": AAVS_TSK2,

        "PR_TSK7": PR_TSK7,
        "COV_TSK7": COV_TSK7,
        "PA_TSK7": PA_TSK7,
        "RI_TSK7": RI_TSK7,
        "RA_TSK7": RA_TSK7,

        "RelF0SD_TSK1": RelF0SD_TSK1,
        "RelF0SD_TSK2": RelF0SD_TSK2,

        "RelSE0SD_TSK1": RelSE0SD_TSK1,
        "RelSE0SD_TSK2": RelSE0SD_TSK2,

        "DurMED_TSK1": DurMED_TSK1,
        "DurMAD_TSK1": DurMAD_TSK1,
        "SPIR_TSK1": SPIR_TSK1,
        "pause_n_TSK1": pause_n_TSK1,
        "pause_ratio_TSK1": pause_ratio_TSK1,

        "DurMED_TSK2": DurMED_TSK2,
        "DurMAD_TSK2": DurMAD_TSK2,
        "SPIR_TSK2": SPIR_TSK2,
        "pause_n_TSK2": pause_n_TSK2,
        "pause_ratio_TSK2": pause_ratio_TSK2,
    }

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writerow(row)

print("Hotovo! 🎧 akustické parametry jsou v:", output_csv)
