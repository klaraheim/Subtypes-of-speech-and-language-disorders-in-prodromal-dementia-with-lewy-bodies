import parselmouth
from parselmouth.praat import call
import numpy as np
import pandas as pd
import math
import os
import time
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.stats import median_abs_deviation
import scipy
import disvoice
import sys
from disvoice.phonation import Phonation
from disvoice.prosody import Prosody
from disvoice.articulation import Articulation
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import re
from sklearn.linear_model import LinearRegression
from statistics import mean, stdev
from collections import Counter
import speechpy.processing as spk

# Respirace ########################################################
def Respiratory_Rate(
    prsnl_ID, task_nmbr,
    silence_margin_db: float = 8.0,
    min_breath_pause: float = 0.25
):
    base_dir = r"E:/diplomka/DLBNLP"
    file_name = prsnl_ID + "_CZ-AZV-TSK" + task_nmbr + "_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)

    if not os.path.exists(wav_path):
        print("file not found")
        return math.nan

    sound = parselmouth.Sound(wav_path)
    total_dur = sound.get_total_duration()

    intensity = call(sound, "To Intensity", 75, 0.01, "yes")
    time_step = call(intensity, "Get time step")

    intensity_values = intensity.values.T.flatten()
    intensity_values = np.nan_to_num(intensity_values, nan=0.0)

    nonzero = intensity_values[intensity_values > 0]
    if len(nonzero) == 0 or total_dur <= 0:
        return math.nan

    mean_intensity = float(np.mean(nonzero))
    silence_threshold = mean_intensity - silence_margin_db
    silence_mask = intensity_values < silence_threshold

    n_breaths = 0
    current_val = silence_mask[0]
    start_idx = 0

    for i in range(1, len(silence_mask) + 1):
        if i == len(silence_mask) or silence_mask[i] != current_val:
            end_idx = i
            if current_val:
                seg_duration = (end_idx - start_idx) * time_step
                if seg_duration >= min_breath_pause:
                    n_breaths += 1
            if i < len(silence_mask):
                current_val = silence_mask[i]
                start_idx = i

    rr = n_breaths / (total_dur / 60.0) if total_dur > 0 else math.nan
    return rr


def IPA_mean(
    prsnl_ID, task_nmbr,  
    silence_margin_db: float = 8.0,
    min_inspiratory_pause: float = 0.3,  
):
    base_dir = r"E:/diplomka/DLBNLP"
    file_name = prsnl_ID + "_CZ-AZV-TSK" + task_nmbr + "_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)

    sound = parselmouth.Sound(wav_path)
    duration = sound.get_total_duration()
    intensity = call(sound, "To Intensity", 75, 0.01, "yes")
    time_step = call(intensity, "Get time step")
    intensity_values = np.nan_to_num(intensity.values.T.flatten(), nan=0.0)
    nonzero = intensity_values[intensity_values > 0]
    if len(nonzero) == 0:
        return float("nan")
    mean_intensity = float(np.mean(nonzero))
    silence_threshold = mean_intensity - silence_margin_db
    silence_mask = intensity_values < silence_threshold

    segments = []
    current_val = silence_mask[0]
    start_idx = 0

    for i in range(1, len(silence_mask) + 1):
        if i == len(silence_mask) or silence_mask[i] != current_val:
            seg_start = start_idx * time_step
            seg_end = i * time_step
            seg_duration = seg_end - seg_start
            segments.append((current_val, seg_duration))
            if i < len(silence_mask):
                current_val = silence_mask[i]
                start_idx = i

    ipa_segments = [d for s, d in segments if s and d >= min_inspiratory_pause]
    if len(ipa_segments) == 0:
        return float("nan")

    ipa_mean = float(np.mean(ipa_segments))
    return ipa_mean


def MPT(prsnl_ID, task_nmbr):
    base_dir = r"E:/diplomka/DLBNLP"
    file_name = prsnl_ID + "_CZ-AZV-TSK" + task_nmbr + "_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)

    sound = parselmouth.Sound(wav_path)
    MPT_duration = call(sound, "Get total duration")
    return MPT_duration


# Fonace ########################################################
def HRF(prsnl_ID, task_nmbr):
    base_dir = r"E:/diplomka/DLBNLP"
    file_name = prsnl_ID + "_CZ-AZV-TSK" + task_nmbr + "_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)

    sound = parselmouth.Sound(wav_path)
    samples = sound.values[0]
    sr = sound.sampling_frequency

    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(len(samples), 1 / sr)
    power = np.abs(spectrum) ** 2

    low_mask = freqs < 500.0
    high_mask = freqs >= 500.0

    E_low = power[low_mask].sum()
    E_high = power[high_mask].sum()

    if E_low == 0:
        HRF_value = float("nan")
    else:
        HRF_value = E_high / E_low

    return HRF_value


def HNR_PPQ_APQ(prsnl_ID, task_nmbr):
    base_dir = r"E:/diplomka/DLBNLP"
    file_name = prsnl_ID + "_CZ-AZV-TSK" + task_nmbr + "_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)

    sound = parselmouth.Sound(wav_path)

    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    HNR = call(harmonicity, "Get mean", 0, 0)

    pitch = sound.to_pitch()
    pointProcess = call([sound, pitch], "To PointProcess (cc)")

    PPQ = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    APQ = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    return HNR, PPQ, APQ

def jitter_shimmer(prsnl_ID, task_nmbr):
    base_dir = r"E:/diplomka/DLBNLP"
    file_name = prsnl_ID + "_CZ-AZV-TSK" + task_nmbr + "_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)

    phonation = disvoice.Phonation()
    features = phonation.extract_features_file(
        wav_path,
        static=True,
        plots=False,
        fmt="dataframe"
    )

    jitter = features.iloc[0]["avg Jitter"]
    shimmer = features.iloc[0]["avg Shimmer"]
    return jitter, shimmer

def DF0_DDF0(prsnl_ID, task_nmbr):
    base_dir = r"E:/diplomka/DLBNLP"
    file_name = prsnl_ID + "_CZ-AZV-TSK" + task_nmbr + "_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)

    features = disvoice.Phonation().extract_features_file(
        wav_path,
        static=False,
        plots=False,
        fmt="dataframe"
    )

    DF0 = features.iloc[0]["DF0"]
    DDF0 = features.iloc[0]["DDF0"]

    return DF0, DDF0

def DUV(prsnl_ID, task_nmbr):
    base_dir = r"E:/diplomka/DLBNLP"
    file_name = f"{prsnl_ID}_CZ-AZV-TSK{task_nmbr}_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)

    sound = parselmouth.Sound(wav_path)
    pitch = sound.to_pitch()
    pointProcess = call([sound, pitch], "To PointProcess (cc)")
    voice_report = parselmouth.praat.call([sound, pitch, pointProcess], "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03, 0.45)

    ML_list = [line for line in voice_report.split('\n') if "Fraction of locally unvoiced frames:" in line]
    ML_split1 = ML_list[0].split(':')
    ML_split2 = ML_split1[1].split(' ')
    DUV = float(ML_split2[1].replace("%",""))
    
    return DUV

# Artikulace ########################################################
def RelF1SD_RelF2SD(prsnl_ID, task_nmbr):
    base_dir = r"E:/diplomka/DLBNLP"
    file_name = prsnl_ID + "_CZ-AZV-TSK" + task_nmbr + "_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)

    articulation = disvoice.Articulation()
    features = articulation.extract_features_file(
        wav_path,
        static=True,
        plots=False,
        fmt="dataframe"
    )

    relF1SD = features.iloc[0]["std F1"] / features.iloc[0]["avg F1"]
    relF2SD = features.iloc[0]["std F2"] / features.iloc[0]["avg F2"]

    return relF1SD, relF2SD


def extract_f1f2_from_task(prsnl_ID, task_nmbr):
    base_dir = r"E:/diplomka/DLBNLP"
    file_name = f"{prsnl_ID}_CZ-AZV-TSK{task_nmbr}_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)
    if not os.path.exists(wav_path):
        return None
    sound = parselmouth.Sound(wav_path)
    pitch = sound.to_pitch(time_step=0.001, pitch_floor=70, pitch_ceiling=400)
    formant = sound.to_formant_burg(time_step=0.001)
    vals = []
    n = formant.get_number_of_frames()
    for i in range(1, n):
        t = formant.get_time_from_frame_number(i)
        f0 = pitch.get_value_at_time(t)
        if not math.isnan(f0):
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            if not math.isnan(f1) and not math.isnan(f2):
                vals.append([f1, f2])
    if len(vals) == 0:
        return None
    return np.array(vals)


def VAI(means):
    F2u = np.min(means, 0)[1]
    F1u = means[np.argmin(means, axis=0)[1], 0]
    F2i = np.max(means, 0)[1]
    F1i = means[np.argmax(means, axis=0)[1], 0]
    F1a = np.max(means, 0)[0]
    F2a = means[np.argmax(means, axis=0)[0], 1]
    return (F2i + F1a) / (F2u + F2a + F1i + F1u)


def VSHA(means):
    hull = ConvexHull(means)
    return float(hull.volume)


def AAVS(means):
    COV = np.cov(means, rowvar=False)
    DET = np.linalg.det(COV)
    if DET < 0:
        DET = 0
    return math.sqrt(DET)


def VAI_VSHA_AAVS(prsnl_ID, task_nmbr):
    f1f2 = extract_f1f2_from_task(prsnl_ID, task_nmbr)
    if f1f2 is None:
        return math.nan, math.nan, math.nan
    if f1f2.shape[0] < 30:
        filt = f1f2
    else:
        gmm = GaussianMixture(3, covariance_type="full", random_state=10)
        gmm.fit(f1f2)
        like = np.exp(gmm.score_samples(f1f2))
        th = 0.3 * np.mean(like)
        filt = f1f2[like >= th]
        if filt.shape[0] < 10:
            filt = f1f2
    if filt.shape[0] < 39:
        k = max(3, min(10, filt.shape[0] // 3))
    else:
        k = 13
    km = KMeans(n_clusters=k, random_state=10)
    km.fit(filt)
    centers = km.cluster_centers_
    return VAI(centers), VSHA(centers), AAVS(filt)

def Art_parameters(prsnl_ID: str, task_nmbr: str):
    base_dir = r"E:/diplomka/DLBNLP"
    wav_file = f"{prsnl_ID}_CZ-AZV-TSK{task_nmbr}_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, wav_file)

    lab_file = f"{prsnl_ID}_CZ-AZV-TSK{task_nmbr}_1.lab"
    lab_path = os.path.join(base_dir, prsnl_ID, lab_file)

    if not os.path.exists(lab_path):
        raise FileNotFoundError(lab_path)

    data = pd.read_csv(
        lab_path,
        delimiter=" ",
        header=0,
        names=["Start", "Stop", "Syllable"]
    )

    data["Duration"] = data["Stop"] - data["Start"]

    cycles = []
    n_syll = len(data.index)
    for i in range(math.floor(n_syll / 3) - 1):
        cycles.append(
            data["Duration"][i * 3]
            + data["Duration"][i * 3 + 1]
            + data["Duration"][i * 3 + 2]
        )

    cycles = np.array(cycles, dtype=float)

    if len(cycles) > 1:
        X = np.linspace(0, len(cycles) - 1, len(cycles)).reshape(-1, 1)
        LR = LinearRegression().fit(X, cycles)
        reg = LR.predict(X)
        dev = abs(reg - cycles)
        RA = float(LR.coef_[0])
    else:
        reg = np.array([])
        dev = np.array([])
        RA = math.nan

    if len(data["Stop"]) >= 30:
        PR = 30 / (data["Stop"].iloc[29] - data["Start"].iloc[0])
    else:
        PR = math.nan

    if len(cycles) >= 9:
        COV = 100 * (stdev(cycles[3:9]) / mean(cycles[0:2]))
        PA = 100 * ((mean(cycles[3:5]) - mean(cycles[6:8])) / mean(cycles[0:2]))
    else:
        COV = math.nan
        PA = math.nan

    if len(dev) > 0:
        total_time = data["Stop"].iloc[-1]
        RI = float(sum(dev) / total_time)
    else:
        RI = math.nan

    return PR, COV, PA, RI, RA


# Prozodie ########################################################

def RelF0SD_RelSE0SD(prsnl_ID, task_nmbr):
    base_dir = r"E:/diplomka/DLBNLP"
    #task_nmbr = "2"
    file_name = prsnl_ID + "_CZ-AZV-TSK" + task_nmbr + "_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)

    prosody = disvoice.Prosody()
    features = prosody.extract_features_file(
        wav_path,
        static=True,
        plots=False,
        fmt="dataframe"
    )

    relF0SD = features.iloc[0]["F0std"] / features.iloc[0]["F0avg"]

    avgE = features.iloc[0]["avgEvoiced"]
    stdE = features.iloc[0]["stdEvoiced"]

    if np.isnan(avgE) or avgE == 0 or np.isnan(stdE):
        relSE0SD = math.nan
    else:
        relSE0SD = stdE / abs(avgE)

    return relF0SD, relSE0SD

def Pros_parameters(prsnl_ID, task_nmbr):
    base_dir = r"E:/diplomka/DLBNLP"
    file_name = f"{prsnl_ID}_CZ-AZV-TSK{task_nmbr}_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)

    sound = parselmouth.Sound(wav_path)
    total_duration = sound.duration

    tg = call(
        sound,
        "To TextGrid (silences)",
        100.0,
        0.0,
        -25.0,
        0.05,
        0.05,
        "silent",
        "speech"
    )

    pauses = []
    total_pause = 0.0

    n_intervals = int(call(tg, "Get number of intervals", 1))
    for i in range(1, n_intervals + 1):
        label = call(tg, "Get label of interval", 1, i)
        if label == "silent":
            start = call(tg, "Get start time of interval", 1, i)
            end = call(tg, "Get end time of interval", 1, i)
            dur = end - start
            if dur > 0.05:
                pauses.append(dur)
                total_pause += dur

    pauses = np.array(pauses) if len(pauses) > 0 else np.array([])

    speech_time = total_duration - total_pause if total_duration > 0 else 0.0
    long_pauses = pauses[pauses >= 0.25] if pauses.size > 0 else np.array([])

    DurMED = float(np.median(long_pauses)) if long_pauses.size > 0 else 0.0
    DurMAD = float(median_abs_deviation(long_pauses)) if long_pauses.size > 0 else 0.0
    SPIR = float(len(pauses) / speech_time) if speech_time > 0 and pauses.size > 0 else 0.0

    return DurMED, DurMAD, SPIR

def Pause_metrics(prsnl_ID, task_nmbr,
    silence_margin_db: float = 8.0,
    min_pause_duration: float = 0.25  
):
    base_dir = r"E:/diplomka/DLBNLP"
    file_name = f"{prsnl_ID}_CZ-AZV-TSK{task_nmbr}_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)

    sound = parselmouth.Sound(wav_path)
    duration = sound.get_total_duration()

    intensity = call(sound, "To Intensity", 75, 0.01, "yes")
    time_step = call(intensity, "Get time step")
    intensity_values = np.nan_to_num(intensity.values.T.flatten(), nan=0.0)

    nonzero = intensity_values[intensity_values > 0]
    if len(nonzero) == 0 or duration <= 0:
        return 0, 0.0

    mean_intensity = float(np.mean(nonzero))
    silence_threshold = mean_intensity - silence_margin_db
    silence_mask = intensity_values < silence_threshold

    pauses = []
    current = silence_mask[0]
    start_idx = 0

    for i in range(1, len(silence_mask) + 1):
        if i == len(silence_mask) or silence_mask[i] != current:
            seg_start = start_idx * time_step
            seg_end = i * time_step
            seg_duration = seg_end - seg_start

            if current and seg_duration >= min_pause_duration:
                pauses.append(seg_duration)

            if i < len(silence_mask):
                current = silence_mask[i]
                start_idx = i

    pause_count = len(pauses)
    pause_ratio = 100 * sum(pauses) / duration if duration > 0 else 0.0

    return pause_count, pause_ratio



