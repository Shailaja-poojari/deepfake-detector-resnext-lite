import numpy as np
import librosa
import cv2
from scipy.spatial.distance import cosine

def extract_mouth_motion(frames):
    activity = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # lower face region (approx)
        mouth = gray[int(h * 0.55):int(h * 0.80), int(w * 0.25):int(w * 0.75)]
        blur = cv2.GaussianBlur(mouth, (5, 5), 0)
        diff = cv2.absdiff(mouth, blur)
        activity.append(np.mean(diff))

    return np.array(activity)


def extract_audio_energy(audio_path):
    audio, sr = librosa.load(audio_path)
    energy = librosa.feature.rms(y=audio).flatten()
    return energy


def av_consistency(frames, audio_path):
    mouth = extract_mouth_motion(frames)
    audio = extract_audio_energy(audio_path)

    L = min(len(mouth), len(audio))
    mouth, audio = mouth[:L], audio[:L]

    score = 1 - cosine(mouth, audio)
    return max(0, float(score))
