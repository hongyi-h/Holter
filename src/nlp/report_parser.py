"""Parse Chinese Holter ECG report conclusions into structured labels."""
import re
import numpy as np

LABEL_NAMES = [
    "sinus_rhythm", "sinus_tachycardia", "sinus_bradycardia",
    "atrial_premature", "ventricular_premature", "atrial_fibrillation",
    "atrial_flutter", "svt", "ventricular_tachycardia",
    "av_block", "st_change", "long_qt", "pause",
]

LABEL_PATTERNS = {
    "sinus_rhythm": [r"窦性心律", r"窦性心率"],
    "sinus_tachycardia": [r"窦性心动过速", r"窦速"],
    "sinus_bradycardia": [r"窦性心动过缓", r"窦缓"],
    "atrial_premature": [r"房性早搏", r"房早", r"房性期前收缩"],
    "ventricular_premature": [r"室性早搏", r"室早", r"室性期前收缩"],
    "atrial_fibrillation": [r"房颤", r"心房颤动"],
    "atrial_flutter": [r"房扑", r"心房扑动"],
    "svt": [r"室上性心动过速", r"室上速", r"SVT"],
    "ventricular_tachycardia": [r"室性心动过速", r"室速", r"VT"],
    "av_block": [r"房室传导阻滞", r"AVB", r"传导阻滞"],
    "st_change": [r"ST段", r"ST-T", r"ST改变"],
    "long_qt": [r"QT延长", r"长QT", r"QTc"],
    "pause": [r"停搏", r"长间歇", r"RR间期"],
}


def parse_conclusion(text):
    """Parse a single conclusion text into binary labels."""
    if not text or not isinstance(text, str):
        return {name: 0 for name in LABEL_NAMES}
    labels = {}
    for name, patterns in LABEL_PATTERNS.items():
        labels[name] = 0
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                labels[name] = 1
                break
    return labels


def extract_numeric_from_text(text):
    """Extract numeric values from report text."""
    numerics = {}
    # Heart rate
    hr_match = re.search(r'(?:平均|总)?(?:心率|HR)[^\d]*(\d+)', text)
    if hr_match:
        numerics["mean_hr"] = float(hr_match.group(1))
    # Max HR
    max_hr = re.search(r'(?:最快|最大|最高)(?:心率|HR)[^\d]*(\d+)', text)
    if max_hr:
        numerics["max_hr"] = float(max_hr.group(1))
    # Min HR
    min_hr = re.search(r'(?:最慢|最小|最低)(?:心率|HR)[^\d]*(\d+)', text)
    if min_hr:
        numerics["min_hr"] = float(min_hr.group(1))
    # Total beats
    beats = re.search(r'(?:总|心搏)[^\d]*(\d+)', text)
    if beats:
        numerics["total_beats"] = float(beats.group(1))
    # PVC count
    pvc = re.search(r'室(?:性)?早[^\d]*(\d+)', text)
    if pvc:
        numerics["pvc_count"] = float(pvc.group(1))
    # PAC count
    pac = re.search(r'房(?:性)?早[^\d]*(\d+)', text)
    if pac:
        numerics["pac_count"] = float(pac.group(1))
    return numerics


def parse_patient_labels(csv_meta):
    """
    Parse patient labels from CSV metadata.
    csv_meta: dict with conclusion text (key: 'conclusion' or chr(32467)+chr(35770))
    Returns: dict with 'binary' and 'numeric' sub-dicts
    """
    conclusion_keys = ["conclusion", "\u7ed3\u8bba", "结论"]
    text = ""
    for key in conclusion_keys:
        if key in csv_meta:
            text = str(csv_meta[key])
            break

    # Also try any key containing 结论
    if not text:
        for key, val in csv_meta.items():
            if "结论" in str(key) or "conclusion" in str(key).lower():
                text = str(val)
                break

    binary = parse_conclusion(text)
    numeric = extract_numeric_from_text(text)
    return {"binary": binary, "numeric": numeric, "has_text": bool(text)}
