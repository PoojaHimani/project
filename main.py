import cv2
import numpy as np
import time
import string
import os
import warnings
import subprocess
from collections import deque
from typing import List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

warnings.filterwarnings(
    "ignore",
    message=".*'pin_memory' argument is set as true but no accelerator is found.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*SymbolDatabase.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)

from hand_tracking import HandTracker
from ui_utils import draw_ui
from hdc_encoder import HDCEncoder, AssociativeMemory
import pyttsx3
try:
    import easyocr
except Exception:
    easyocr = None

# Parameters
SMOOTHING_WINDOW = 5
STABLE_THRESHOLD = 3  # frames to confirm a gesture state
DRAW_COLOR = (0, 255, 0)  # BGR green (use single color for drawing + cursor)
ERASE_COLOR = (0, 0, 0)
THICKNESS = 8
ERASE_RADIUS = 48
MIN_POINTS_FOR_RECOGNITION = 24
HDC_ACCEPT_THRESHOLD = 0.08
SHAPE_ACCEPT_THRESHOLD = 0.40
COMBINED_ACCEPT_THRESHOLD = 0.26
WRITE_STABLE_FRAMES = 1
ERASE_STABLE_FRAMES = 4
LOST_TRACK_GRACE_FRAMES = 8
WRITE_HOLD_FRAMES = 4
MIN_MOVE_PIXELS = 1
EMA_ALPHA = 0.35
MAX_POINT_JUMP = 80
MAX_BRIDGE_JUMP = 180
LINE_INTERPOLATION_STEP = 4
MIN_ERASE_FINGER_GAP = 35
HELLO_RELAX_FACTOR = 0.65
HELLO_SHAPE_FALLBACK = 0.18
FORCE_HELLO_FALLBACK = False
UNRECOGNIZED_SPEECH_TEXT = "not recognized"


def normalize_stroke_points(stroke: List[Tuple[int, int]], samples: int = 64) -> List[Tuple[int, int]]:
    if len(stroke) <= 1:
        return stroke

    pts = np.array(stroke, dtype=np.float32)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    norm = (pts - min_xy) / span

    segment_lengths = np.linalg.norm(np.diff(norm, axis=0), axis=1)
    total_length = float(segment_lengths.sum())
    if total_length <= 1e-6:
        scaled = np.rint(norm * 999).astype(np.int32)
        return [(int(x), int(y)) for x, y in scaled]

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    targets = np.linspace(0.0, total_length, samples, dtype=np.float32)
    out = []
    for target in targets:
        idx = int(np.searchsorted(cumulative, target, side="right") - 1)
        idx = max(0, min(idx, len(norm) - 2))
        local_len = cumulative[idx + 1] - cumulative[idx]
        alpha = 0.0 if local_len <= 1e-6 else float((target - cumulative[idx]) / local_len)
        point = norm[idx] * (1.0 - alpha) + norm[idx + 1] * alpha
        out.append((int(round(point[0] * 999)), int(round(point[1] * 999))))
    return out


def encode_word_strokes(encoder: HDCEncoder, strokes: List[List[Tuple[int, int]]]) -> np.ndarray:
    hvs = []
    for stroke in strokes:
        if not stroke:
            continue
        normalized = normalize_stroke_points(stroke, samples=64)
        hvs.append(encoder.encode_sequence(normalized, (1000, 1000)))
    return encoder.bundle(hvs)


def stroke_to_feature(stroke: List[Tuple[int, int]], samples: int = 64) -> np.ndarray:
    if not stroke:
        return np.zeros(samples * 2, dtype=np.float32)
    normalized = normalize_stroke_points(stroke, samples=samples)
    arr = np.array(normalized, dtype=np.float32).reshape(-1)
    arr -= arr.mean()
    norm = np.linalg.norm(arr) + 1e-9
    return arr / norm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def clean_ocr_token(token: str) -> str:
    if not token:
        return ""
    keep_chars = [ch for ch in token if ch.isalnum()]
    return "".join(keep_chars)


def resize_with_padding(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.full((target_h, target_w), 255, dtype=np.uint8)
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((target_h, target_w), 255, dtype=np.uint8)
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    out = np.full((target_h, target_w), 255, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    out[y0:y0 + new_h, x0:x0 + new_w] = resized
    return out


def build_alnum_shape_templates():
    templates = {}
    chars = string.ascii_letters + string.digits
    for ch in chars:
        glyph = np.zeros((140, 140), dtype=np.uint8)
        cv2.putText(glyph, ch, (14, 112), cv2.FONT_HERSHEY_SIMPLEX, 3.2, 255, 9, cv2.LINE_AA)
        _, glyph_bin = cv2.threshold(glyph, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(glyph_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        templates[ch] = max(contours, key=cv2.contourArea)
    return templates


def build_alnum_knn_classifier():
    classes = list(string.ascii_letters + string.digits)
    class_to_idx = {ch: i for i, ch in enumerate(classes)}
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]
    scales = [1.8, 2.1, 2.4]
    thicknesses = [2, 3, 4]
    angles = [-9.0, -4.0, 0.0, 4.0, 9.0]

    samples = []
    labels = []
    for ch in classes:
        for font in fonts:
            for scale in scales:
                for thick in thicknesses:
                    for angle in angles:
                        glyph = np.zeros((96, 96), dtype=np.uint8)
                        cv2.putText(glyph, ch, (10, 75), font, scale, 255, thick, cv2.LINE_AA)
                        rot = cv2.getRotationMatrix2D((48, 48), angle, 1.0)
                        rotated = cv2.warpAffine(glyph, rot, (96, 96), flags=cv2.INTER_LINEAR, borderValue=0)
                        _, bw = cv2.threshold(rotated, 10, 255, cv2.THRESH_BINARY)
                        pts = cv2.findNonZero(bw)
                        if pts is None:
                            continue
                        x, y, w, h = cv2.boundingRect(pts)
                        roi = bw[y:y + h, x:x + w]
                        norm = resize_with_padding(roi, 32, 32)
                        feature = (norm.astype(np.float32).reshape(-1) / 255.0)
                        samples.append(feature)
                        labels.append(class_to_idx[ch])

    if not samples:
        return None, classes

    train_x = np.array(samples, dtype=np.float32)
    train_y = np.array(labels, dtype=np.int32).reshape(-1, 1)
    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(5)
    knn.train(train_x, cv2.ml.ROW_SAMPLE, train_y)
    return knn, classes


def compute_hog_feature(binary_img: np.ndarray) -> np.ndarray:
    if binary_img is None or binary_img.size == 0:
        return np.zeros((324,), dtype=np.float32)
    norm = resize_with_padding(binary_img, 32, 32)
    hog = cv2.HOGDescriptor(
        _winSize=(32, 32),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    desc = hog.compute(norm)
    if desc is None:
        return np.zeros((324,), dtype=np.float32)
    return desc.reshape(-1).astype(np.float32)


def merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]], max_gap: int = 10) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    cur_x, cur_y, cur_w, cur_h = sorted_boxes[0]
    cur_x2 = cur_x + cur_w
    cur_y2 = cur_y + cur_h

    for x, y, w, h in sorted_boxes[1:]:
        x2 = x + w
        y2 = y + h
        overlaps_or_close = x <= (cur_x2 + max_gap)
        if overlaps_or_close:
            cur_x = min(cur_x, x)
            cur_y = min(cur_y, y)
            cur_x2 = max(cur_x2, x2)
            cur_y2 = max(cur_y2, y2)
        else:
            merged.append((cur_x, cur_y, cur_x2 - cur_x, cur_y2 - cur_y))
            cur_x, cur_y, cur_x2, cur_y2 = x, y, x2, y2

    merged.append((cur_x, cur_y, cur_x2 - cur_x, cur_y2 - cur_y))
    return merged


def build_alnum_hog_knn_classifier():
    classes = list(string.ascii_letters + string.digits)
    class_to_idx = {ch: i for i, ch in enumerate(classes)}
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
    ]
    scales = [1.7, 2.0, 2.3]
    thicknesses = [2, 3, 4]
    angles = [-12.0, -7.0, -3.0, 0.0, 3.0, 7.0, 12.0]

    samples = []
    labels = []
    for ch in classes:
        for font in fonts:
            for scale in scales:
                for thick in thicknesses:
                    for angle in angles:
                        glyph = np.zeros((96, 96), dtype=np.uint8)
                        cv2.putText(glyph, ch, (8, 78), font, scale, 255, thick, cv2.LINE_AA)
                        rot = cv2.getRotationMatrix2D((48, 48), angle, 1.0)
                        rotated = cv2.warpAffine(glyph, rot, (96, 96), flags=cv2.INTER_LINEAR, borderValue=0)
                        _, bw = cv2.threshold(rotated, 10, 255, cv2.THRESH_BINARY)
                        pts = cv2.findNonZero(bw)
                        if pts is None:
                            continue
                        x, y, w, h = cv2.boundingRect(pts)
                        roi = bw[y:y + h, x:x + w]
                        feature = compute_hog_feature(roi)
                        samples.append(feature)
                        labels.append(class_to_idx[ch])

    if not samples:
        return None, classes

    train_x = np.array(samples, dtype=np.float32)
    train_y = np.array(labels, dtype=np.int32).reshape(-1, 1)
    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(7)
    knn.train(train_x, cv2.ml.ROW_SAMPLE, train_y)
    return knn, classes


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker(maxHands=1, detectionCon=0.7, trackCon=0.5)

    canvas = None
    pts = deque(maxlen=SMOOTHING_WINDOW)
    last_point = None
    filtered_point = None
    pen_enabled = True  # manual toggle to force writing on/off
    recording = False
    recorded_strokes = []
    current_stroke = []
    encoder = HDCEncoder(dim=5000, x_bins=64, y_bins=48)
    amem = AssociativeMemory()
    tts = None
    voice_enabled = True
    ocr_reader = None
    ocr_init_attempted = False
    alnum_shape_templates = build_alnum_shape_templates()
    alnum_knn, alnum_classes = build_alnum_knn_classifier()
    alnum_hog_knn, alnum_hog_classes = build_alnum_hog_knn_classifier()

    if easyocr is None:
        print("⚠️ EasyOCR not installed. Using HDC fallback only.")

    def ensure_ocr_reader():
        nonlocal ocr_reader, ocr_init_attempted
        if ocr_reader is not None:
            return True
        if ocr_init_attempted:
            return False
        ocr_init_attempted = True
        if easyocr is None:
            return False
        try:
            print("⏳ Initializing EasyOCR (first run may download models)...")
            ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("✅ EasyOCR initialized")
            return True
        except Exception as e:
            print(f"⚠️ EasyOCR init failed: {e}")
            return False

    def speak_windows_fallback(text: str):
        safe = text.replace("'", "''")
        cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Volume=100; $s.Rate=0; "
            f"$s.Speak('{safe}')"
        )
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Sta", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=12,
            )
            return res.returncode == 0
        except Exception:
            return False

    def speak_with_pyttsx3(text: str):
        nonlocal tts
        try:
            tts = pyttsx3.init()
            tts.setProperty("volume", 1.0)
            tts.setProperty("rate", 165)
            tts.stop()
            tts.say(text)
            tts.runAndWait()
            return True
        except Exception:
            return False

    def speak_output(text: str):
        nonlocal voice_enabled
        if not voice_enabled:
            voice_enabled = True
        phrase = str(text).strip() if text is not None else ""
        if (not phrase) or phrase.lower() == "none":
            phrase = UNRECOGNIZED_SPEECH_TEXT
        if speak_windows_fallback(phrase):
            return
        if speak_with_pyttsx3(phrase):
            return
        print(f"⚠️ Speech output failed: {phrase}")

    def recognize_with_easyocr(canvas_img):
        if canvas_img is None:
            return ""
        if not ensure_ocr_reader():
            return ""
        try:
            gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
            _, stroke_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(stroke_mask) < 80:
                return ""

            points = cv2.findNonZero(stroke_mask)
            if points is None:
                return ""

            x, y, bw, bh = cv2.boundingRect(points)
            pad = 20
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(stroke_mask.shape[1], x + bw + pad)
            y2 = min(stroke_mask.shape[0], y + bh + pad)
            roi = stroke_mask[y1:y2, x1:x2]
            if roi.size == 0:
                return ""

            base = cv2.bitwise_not(roi)
            base = cv2.resize(base, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            variant_a = cv2.GaussianBlur(base, (3, 3), 0)
            variant_b = cv2.dilate(base, kernel, iterations=1)
            variant_c = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel, iterations=1)
            variants = [variant_a, variant_b, variant_c]

            best_text = ""
            best_score = -1.0

            for ocr_input in variants:
                results = ocr_reader.readtext(
                    ocr_input,
                    detail=1,
                    paragraph=False,
                    decoder='beamsearch',
                    allowlist='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                )
                if not results:
                    continue

                ordered = sorted(results, key=lambda item: min(pt[0] for pt in item[0]))
                tokens = []
                weighted_conf = 0.0
                total_chars = 0
                for _, txt, conf in ordered:
                    token = clean_ocr_token(str(txt).strip())
                    if not token:
                        continue
                    tokens.append(token)
                    token_len = max(1, len(token))
                    conf_val = float(conf) if conf is not None else 0.0
                    weighted_conf += max(0.0, conf_val) * token_len
                    total_chars += token_len

                if not tokens:
                    continue

                candidate_text = "".join(tokens).strip()
                avg_conf = weighted_conf / max(1, total_chars)
                length_bonus = min(total_chars, 12) / 12.0
                candidate_score = 0.7 * avg_conf + 0.3 * length_bonus
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_text = candidate_text

            return best_text
        except Exception as e:
            print(f"⚠️ EasyOCR read failed: {e}")
            return ""

    def recognize_alnum_chars(canvas_img):
        if canvas_img is None:
            return "", 0.0
        try:
            gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
            _, stroke_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(stroke_mask) < 60:
                return "", 0.0

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(stroke_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            cleaned = cv2.dilate(cleaned, kernel, iterations=1)

            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
            char_boxes = []
            for idx in range(1, n_labels):
                x, y, bw, bh, area = stats[idx]
                if area < 45 or bw < 4 or bh < 8:
                    continue
                char_boxes.append((x, y, bw, bh))

            if not char_boxes:
                return "", 0.0

            char_boxes = merge_overlapping_boxes(char_boxes, max_gap=10)
            char_boxes.sort(key=lambda b: b[0])
            out_chars = []
            conf_sum = 0.0
            conf_count = 0
            for x, y, bw, bh in char_boxes:
                pad = 8
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(cleaned.shape[1], x + bw + pad)
                y2 = min(cleaned.shape[0], y + bh + pad)
                char_roi = cleaned[y1:y2, x1:x2]
                if char_roi.size == 0:
                    continue

                char_canvas_bgr = cv2.cvtColor(char_roi, cv2.COLOR_GRAY2BGR)
                best_char, best_conf = recognize_single_char_ensemble(char_canvas_bgr)
                if best_char and best_conf >= 0.22:
                    out_chars.append(best_char)
                    conf_sum += best_conf
                    conf_count += 1

            if not out_chars:
                return "", 0.0
            return "".join(out_chars), (conf_sum / max(1, conf_count))
        except Exception as e:
            print(f"⚠️ Alnum OCR fallback failed: {e}")
            return "", 0.0

    def recognize_single_char_shape(canvas_img):
        if canvas_img is None:
            return "", 0.0
        try:
            gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
            _, stroke_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(stroke_mask) < 50:
                return "", 0.0

            points = cv2.findNonZero(stroke_mask)
            if points is None:
                return "", 0.0
            x, y, bw, bh = cv2.boundingRect(points)
            pad = 12
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(stroke_mask.shape[1], x + bw + pad)
            y2 = min(stroke_mask.shape[0], y + bh + pad)
            roi = stroke_mask[y1:y2, x1:x2]
            if roi.size == 0:
                return "", 0.0

            normalized = resize_with_padding(roi, 140, 140)
            contours, _ = cv2.findContours(normalized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return "", 0.0
            query_contour = max(contours, key=cv2.contourArea)

            best_char = ""
            best_score = float("inf")
            for ch, template_contour in alnum_shape_templates.items():
                score = cv2.matchShapes(query_contour, template_contour, cv2.CONTOURS_MATCH_I1, 0.0)
                if score < best_score:
                    best_score = score
                    best_char = ch

            confidence = 1.0 / (1.0 + max(0.0, best_score) * 6.0)
            return best_char, confidence
        except Exception as e:
            print(f"⚠️ Shape char recognition failed: {e}")
            return "", 0.0

    def recognize_single_char_knn(canvas_img):
        if canvas_img is None or alnum_knn is None:
            return "", 0.0
        try:
            gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
            _, stroke_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(stroke_mask) < 50:
                return "", 0.0
            pts = cv2.findNonZero(stroke_mask)
            if pts is None:
                return "", 0.0
            x, y, bw, bh = cv2.boundingRect(pts)
            pad = 12
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(stroke_mask.shape[1], x + bw + pad)
            y2 = min(stroke_mask.shape[0], y + bh + pad)
            roi = stroke_mask[y1:y2, x1:x2]
            if roi.size == 0:
                return "", 0.0

            norm = resize_with_padding(roi, 32, 32)
            sample = (norm.astype(np.float32).reshape(1, -1) / 255.0)
            _, result, neighbours, dist = alnum_knn.findNearest(sample, k=5)
            idx = int(result[0][0])
            pred = alnum_classes[idx] if 0 <= idx < len(alnum_classes) else ""

            if dist is not None and dist.size > 0:
                d0 = float(dist[0][0])
            else:
                d0 = 10.0
            conf = 1.0 / (1.0 + d0 / 8.0)
            if neighbours is not None and neighbours.size > 0:
                neigh_idx = [int(v) for v in neighbours[0].tolist()]
                vote = sum(1 for n in neigh_idx if n == idx)
                conf = min(1.0, conf * (0.8 + 0.08 * vote))
            return pred, conf
        except Exception as e:
            print(f"⚠️ KNN char recognition failed: {e}")
            return "", 0.0

    def recognize_single_char_hog_knn(canvas_img):
        if canvas_img is None or alnum_hog_knn is None:
            return "", 0.0
        try:
            gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
            _, stroke_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(stroke_mask) < 50:
                return "", 0.0
            pts = cv2.findNonZero(stroke_mask)
            if pts is None:
                return "", 0.0
            x, y, bw, bh = cv2.boundingRect(pts)
            pad = 12
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(stroke_mask.shape[1], x + bw + pad)
            y2 = min(stroke_mask.shape[0], y + bh + pad)
            roi = stroke_mask[y1:y2, x1:x2]
            if roi.size == 0:
                return "", 0.0

            feat = compute_hog_feature(roi).reshape(1, -1)
            _, result, neighbours, dist = alnum_hog_knn.findNearest(feat, k=7)
            idx = int(result[0][0])
            pred = alnum_hog_classes[idx] if 0 <= idx < len(alnum_hog_classes) else ""

            d0 = float(dist[0][0]) if (dist is not None and dist.size > 0) else 20.0
            conf = 1.0 / (1.0 + d0 / 6.0)
            if neighbours is not None and neighbours.size > 0:
                neigh_idx = [int(v) for v in neighbours[0].tolist()]
                vote = sum(1 for n in neigh_idx if n == idx)
                conf = min(1.0, conf * (0.75 + 0.05 * vote))
            return pred, conf
        except Exception as e:
            print(f"⚠️ HOG-KNN char recognition failed: {e}")
            return "", 0.0

    def recognize_single_char_easyocr(canvas_img):
        if canvas_img is None:
            return "", 0.0
        if not ensure_ocr_reader():
            return "", 0.0
        try:
            gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
            _, stroke_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(stroke_mask) < 50:
                return "", 0.0

            pts = cv2.findNonZero(stroke_mask)
            if pts is None:
                return "", 0.0
            x, y, bw, bh = cv2.boundingRect(pts)
            pad = 14
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(stroke_mask.shape[1], x + bw + pad)
            y2 = min(stroke_mask.shape[0], y + bh + pad)
            roi = stroke_mask[y1:y2, x1:x2]
            if roi.size == 0:
                return "", 0.0

            inv = cv2.bitwise_not(roi)
            base = resize_with_padding(inv, 96, 96)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            variants = [
                cv2.GaussianBlur(base, (3, 3), 0),
                cv2.dilate(base, kernel, iterations=1),
                cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel, iterations=1),
            ]

            best_char = ""
            best_conf = -1.0
            for variant in variants:
                results = ocr_reader.readtext(
                    variant,
                    detail=1,
                    paragraph=False,
                    decoder='beamsearch',
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz'
                )
                for _, txt, conf in results:
                    token = clean_ocr_token(str(txt).strip())
                    if not token:
                        continue
                    ch = token[0]
                    conf_val = float(conf) if conf is not None else 0.0
                    if conf_val > best_conf:
                        best_conf = conf_val
                        best_char = ch

            if best_char:
                return best_char, max(0.0, min(1.0, best_conf))
            return "", 0.0
        except Exception as e:
            print(f"⚠️ Single-char OCR failed: {e}")
            return "", 0.0

    def recognize_single_char_ensemble(canvas_img, ocr_hint=""):
        ocr_char, ocr_conf = recognize_single_char_easyocr(canvas_img)
        shape_char, shape_conf = recognize_single_char_shape(canvas_img)
        knn_char, knn_conf = recognize_single_char_knn(canvas_img)
        hog_char, hog_conf = recognize_single_char_hog_knn(canvas_img)

        if ocr_char and ocr_conf >= 0.62:
            return ocr_char, ocr_conf

        scores = {}

        def add_score(ch, value):
            if not ch:
                return
            scores[ch] = scores.get(ch, 0.0) + float(value)

        add_score(ocr_char, 1.45 * max(ocr_conf, 0.25 if ocr_char else 0.0))
        add_score(shape_char, 0.55 * shape_conf)
        add_score(knn_char, 0.95 * knn_conf)
        add_score(hog_char, 1.15 * hog_conf)
        if ocr_hint and len(ocr_hint) == 1:
            add_score(ocr_hint[0], 0.30)

        if not scores:
            return "", 0.0

        best_char = max(scores.items(), key=lambda kv: kv[1])[0]
        total = sum(scores.values()) + 1e-9
        confidence = scores[best_char] / total
        return best_char, confidence
    
    # Automatically train "hello" template
    print("📝 Auto-training 'hello' template...")
    shape_templates_by_label = {}
    hello_patterns = [
        [(100, 240), (120, 170), (140, 240), (170, 170), (190, 240), (220, 200), (260, 200), (300, 170), (300, 240), (340, 170), (340, 240)],
        [(110, 250), (130, 180), (150, 250), (180, 180), (200, 250), (235, 210), (275, 210), (315, 180), (315, 250), (355, 180), (355, 250)],
        [(100, 220), (130, 165), (150, 220), (185, 165), (205, 220), (240, 195), (280, 195), (320, 165), (320, 225), (360, 165), (360, 225)]
    ]
    hello_hv = encode_word_strokes(encoder, [hello_patterns[0]])
    hello_hv = encoder.bundle([hello_hv] + [encode_word_strokes(encoder, [p]) for p in hello_patterns[1:]])
    amem.add("hello", hello_hv)
    shape_templates_by_label["hello"] = [stroke_to_feature(p, samples=64) for p in hello_patterns]
    hello_stroke_hvs = [encode_word_strokes(encoder, [p]) for p in hello_patterns]

    # Adaptive thresholds from hello exemplars (leave-one-out for shape).
    hello_shape_loocv = []
    for i, feat in enumerate(shape_templates_by_label["hello"]):
        others = [t for j, t in enumerate(shape_templates_by_label["hello"]) if j != i]
        if others:
            hello_shape_loocv.append(max(cosine_sim(feat, t) for t in others))
    if hello_shape_loocv:
        adaptive_shape_threshold = max(SHAPE_ACCEPT_THRESHOLD, float(min(hello_shape_loocv)) - 0.45)
        adaptive_shape_threshold = min(adaptive_shape_threshold, 0.55)
    else:
        adaptive_shape_threshold = SHAPE_ACCEPT_THRESHOLD

    hello_hdc_scores = [encoder.similarity(hv, hello_hv) for hv in hello_stroke_hvs]
    if hello_hdc_scores:
        adaptive_hdc_threshold = max(HDC_ACCEPT_THRESHOLD, float(min(hello_hdc_scores)) - 0.45)
        adaptive_hdc_threshold = min(adaptive_hdc_threshold, 0.18)
    else:
        adaptive_hdc_threshold = HDC_ACCEPT_THRESHOLD

    print(f"🎚️ hello thresholds -> shape>={adaptive_shape_threshold:.3f}, hdc>={adaptive_hdc_threshold:.3f}")
    print(f"✅ 'hello' added to memory!")
    
    print(f"📚 Total words in memory: {len(amem.prototypes)}")
    
    # accumulate recognized letters into a running word when not recording
    word_buffer = ""
    last_appended_label = None
    # extracted text / speech output
    extracted_text = ""
    speech_text = ""
    # GUI button rects (absolute coords; will be updated each frame)
    rec_btn_rect = [10, 10, 230, 60]
    play_btn_rect = [0, 0, 0, 0]

    # Mouse handler for clickable buttons (will be attached to window)
    def on_mouse(event, x, y, flags, param):
        nonlocal recording, recorded_strokes, current_stroke, extracted_text, speech_text, word_buffer, last_appended_label, recognized_label, recognized_conf, canvas, pts, last_point, filtered_point
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        rx1, ry1, rx2, ry2 = rec_btn_rect
        if rx1 <= x <= rx2 and ry1 <= y <= ry2:
            # toggle recording via button
            recording = not recording
            if recording:
                recorded_strokes = []
                current_stroke = []
                extracted_text = ""
                speech_text = ""
                last_appended_label = None
                recognized_label = None
                recognized_conf = 0.0
                canvas = np.zeros_like(canvas)
                pts.clear()
                last_point = None
                filtered_point = None
            else:
                finalize_recording_and_generate_output()
                recorded_strokes = []
                current_stroke = []
                pts.clear()
                last_point = None
                filtered_point = None
            return
        px1, py1, px2, py2 = play_btn_rect
        if px1 <= x <= px2 and py1 <= y <= py2:
            speak_output(speech_text)
            return

    try:
        cv2.namedWindow("Air Writing System", cv2.WINDOW_NORMAL)
    except cv2.error as e:
        print("❌ OpenCV GUI is not available in this environment.")
        print("Run: python -m pip uninstall -y opencv-python-headless")
        print("Then: python -m pip install --upgrade --force-reinstall opencv-python")
        print(f"Details: {e}")
        cap.release()
        return
    cv2.setMouseCallback("Air Writing System", on_mouse)
    # automatic stroke capture
    recognized_label = None
    recognized_conf = 0.0

    def recognize_recorded_word(strokes):
        if not strokes:
            return None, 0.0, []
        merged_points = [pt for stroke in strokes for pt in stroke]
        if not merged_points:
            return None, 0.0, []
        if len(merged_points) < MIN_POINTS_FOR_RECOGNITION:
            return None, 0.0, []

        # Primary decision: geometric shape similarity against trained templates
        query_shape = stroke_to_feature(merged_points, samples=64)
        shape_rank = []
        for label_name, templates in shape_templates_by_label.items():
            scores = [cosine_sim(query_shape, t) for t in templates]
            if scores:
                shape_rank.append((label_name, max(scores)))
        shape_rank.sort(key=lambda x: x[1], reverse=True)
        if not shape_rank:
            if FORCE_HELLO_FALLBACK:
                return "hello", 0.51, [("shape", 0.0), ("hdc", 0.0), ("combined", 0.0), ("forced", 1.0)]
            return None, 0.0, []

        shape_label, shape_conf = shape_rank[0]

        # Secondary score: HDC associative memory
        query_hv = encode_word_strokes(encoder, strokes)
        top = amem.query(query_hv, topk=2)
        if not top:
            if FORCE_HELLO_FALLBACK and shape_label == "hello":
                fallback_conf = max(shape_conf, 0.51)
                return "hello", fallback_conf, [("shape", shape_conf), ("hdc", 0.0), ("combined", fallback_conf), ("forced", 1.0)]
            return None, 0.0, shape_rank
        label, conf = top[0]

        # Closed-set acceptance for hello-only mode.
        combined_conf = 0.65 * shape_conf + 0.35 * max(conf, 0.0)
        if (
            label == "hello"
            and shape_label == "hello"
            and conf >= adaptive_hdc_threshold
            and shape_conf >= adaptive_shape_threshold
            and combined_conf >= COMBINED_ACCEPT_THRESHOLD
        ):
            return label, combined_conf, [("shape", shape_conf), ("hdc", conf), ("combined", combined_conf)]

        # Relaxed acceptance for single-word hello mode to avoid frequent false negatives.
        if label == "hello" and shape_label == "hello":
            relaxed_ok = (
                conf >= adaptive_hdc_threshold * HELLO_RELAX_FACTOR
                or shape_conf >= adaptive_shape_threshold * HELLO_RELAX_FACTOR
                or combined_conf >= COMBINED_ACCEPT_THRESHOLD * HELLO_RELAX_FACTOR
                or shape_conf >= HELLO_SHAPE_FALLBACK
            )
            if relaxed_ok:
                return "hello", combined_conf, [("shape", shape_conf), ("hdc", conf), ("combined", combined_conf)]

        if FORCE_HELLO_FALLBACK and (label == "hello" or shape_label == "hello"):
            fallback_conf = max(combined_conf, 0.51)
            return "hello", fallback_conf, [("shape", shape_conf), ("hdc", conf), ("combined", combined_conf), ("forced", 1.0)]

        return None, combined_conf, [("shape", shape_conf), ("hdc", conf), ("combined", combined_conf)]

    def finalize_recording_and_generate_output():
        nonlocal current_stroke, recorded_strokes, recognized_label, recognized_conf, extracted_text, speech_text, word_buffer, last_appended_label

        if current_stroke:
            recorded_strokes.append(current_stroke.copy())
            current_stroke = []

        if not recorded_strokes:
            recognized_label = "none"
            recognized_conf = 0.0
            extracted_text = "none"
            speech_text = UNRECOGNIZED_SPEECH_TEXT
            speak_output(speech_text)
            return

        ocr_text = recognize_with_easyocr(canvas)
        ocr_confidence = 1.0
        segmented_text, segmented_conf = recognize_alnum_chars(canvas)
        if len(segmented_text) == 1 and segmented_conf >= 0.32:
            ocr_text = segmented_text
            ocr_confidence = max(ocr_confidence, segmented_conf)
        if segmented_text:
            if (not ocr_text) or (len(segmented_text) > len(ocr_text)):
                ocr_text = segmented_text
                ocr_confidence = max(0.45, segmented_conf)
            elif len(segmented_text) == len(ocr_text) and segmented_conf > 0.58:
                ocr_text = segmented_text
                ocr_confidence = segmented_conf
        if len(ocr_text) == 1:
            fused_char, fused_conf = recognize_single_char_ensemble(canvas, ocr_hint=ocr_text)
            if fused_char and fused_conf >= 0.42:
                ocr_text = fused_char
                ocr_confidence = fused_conf
        if ocr_text:
            recognized_label = ocr_text
            recognized_conf = ocr_confidence
            extracted_text = ocr_text
            speech_text = ocr_text
            if ocr_text != last_appended_label:
                word_buffer += ocr_text
                last_appended_label = ocr_text
            speak_output(ocr_text)
            return

        single_char, single_conf = recognize_single_char_ensemble(canvas)
        if single_char and single_conf >= 0.35:
            recognized_label = single_char
            recognized_conf = single_conf
            extracted_text = single_char
            speech_text = single_char
            if single_char != last_appended_label:
                word_buffer += single_char
                last_appended_label = single_char
            speak_output(single_char)
            return

        label, conf, raw = recognize_recorded_word(recorded_strokes)
        print(f"🔍 Debug: HDC fallback result: {raw}")
        if label:
            recognized_label = label
            recognized_conf = conf
            extracted_text = label
            speech_text = label
            if label != last_appended_label:
                word_buffer += label
                last_appended_label = label
            speak_output(label)
        else:
            recognized_label = "none"
            recognized_conf = 0.0
            extracted_text = "none"
            speech_text = UNRECOGNIZED_SPEECH_TEXT
            speak_output(speech_text)

    # gesture stability counters
    idx_up_count = 0
    two_up_count = 0

    mode = "IDLE"
    prev_time = 0
    lost_track_frames = 0
    write_hold_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        if canvas is None:
            canvas = np.zeros_like(frame)

        landmarks = tracker.find_hands(frame, draw=True)
        if landmarks:
            lm = landmarks[0]
            index_up, middle_up = tracker.fingers_up(lm)
            point = tracker.get_point(lm)
            lm_map = {pid: (px, py) for pid, px, py in lm}
            finger_gap_ok = False
            if 8 in lm_map and 12 in lm_map:
                gap_x = lm_map[8][0] - lm_map[12][0]
                gap_y = lm_map[8][1] - lm_map[12][1]
                finger_gap_ok = ((gap_x * gap_x + gap_y * gap_y) ** 0.5) >= MIN_ERASE_FINGER_GAP
            # gesture stability logic
            erase_detected = recording and point is not None and index_up and middle_up and finger_gap_ok
            write_detected = recording and pen_enabled and point is not None and index_up and (not erase_detected)

            if erase_detected:
                two_up_count += 1
                idx_up_count = 0
            elif write_detected:
                idx_up_count += 1
                two_up_count = 0
            else:
                idx_up_count = 0
                two_up_count = 0

            if erase_detected and two_up_count >= ERASE_STABLE_FRAMES:
                mode = "ERASE"
                write_hold_count = 0
            elif idx_up_count >= WRITE_STABLE_FRAMES:
                mode = "WRITE"
                write_hold_count = WRITE_HOLD_FRAMES
            elif write_hold_count > 0 and point is not None and not erase_detected:
                mode = "WRITE"
                write_hold_count -= 1
            else:
                mode = "IDLE"
                write_hold_count = 0

            # Drawing logic - smooth continuous lines
            if recording and point is not None and mode == "WRITE":
                lost_track_frames = 0
                pts.append(point)
                avg_x = int(sum(p[0] for p in pts) / len(pts))
                avg_y = int(sum(p[1] for p in pts) / len(pts))
                if filtered_point is None:
                    filtered_point = (avg_x, avg_y)
                else:
                    fx = int((1.0 - EMA_ALPHA) * filtered_point[0] + EMA_ALPHA * avg_x)
                    fy = int((1.0 - EMA_ALPHA) * filtered_point[1] + EMA_ALPHA * avg_y)
                    filtered_point = (fx, fy)
                current_point = filtered_point

                if last_point is not None:
                    dx = current_point[0] - last_point[0]
                    dy = current_point[1] - last_point[1]
                    jump = (dx * dx + dy * dy) ** 0.5
                    if jump > MAX_BRIDGE_JUMP:
                        last_point = current_point
                    else:
                        if jump >= MIN_MOVE_PIXELS:
                            steps = max(1, int(jump // LINE_INTERPOLATION_STEP))
                            prev_pt = last_point
                            for step_idx in range(1, steps + 1):
                                alpha = step_idx / float(steps)
                                ix = int(round(last_point[0] * (1.0 - alpha) + current_point[0] * alpha))
                                iy = int(round(last_point[1] * (1.0 - alpha) + current_point[1] * alpha))
                                interp_pt = (ix, iy)
                                cv2.line(canvas, prev_pt, interp_pt, DRAW_COLOR, THICKNESS, cv2.LINE_AA)
                                prev_pt = interp_pt
                        last_point = current_point
                        if recording:
                            current_stroke.append(current_point)
                else:
                    last_point = current_point
                    cv2.circle(canvas, current_point, THICKNESS // 2, DRAW_COLOR, -1, cv2.LINE_AA)
                    if recording:
                        current_stroke.append(current_point)
            elif recording and point is not None and mode == "ERASE":
                lost_track_frames = 0
                if current_stroke:
                    recorded_strokes.append(current_stroke.copy())
                    current_stroke = []
                pts.clear()
                last_point = None
                filtered_point = None
                cv2.circle(canvas, point, ERASE_RADIUS, ERASE_COLOR, -1, cv2.LINE_AA)
            else:
                lost_track_frames += 1
                if lost_track_frames >= LOST_TRACK_GRACE_FRAMES:
                    pts.clear()
                    last_point = None
                    filtered_point = None
                    if recording and current_stroke:
                        recorded_strokes.append(current_stroke.copy())
                        current_stroke = []
        else:
            # no hand detected; keep a short grace period before ending stroke
            lost_track_frames += 1
            if lost_track_frames >= LOST_TRACK_GRACE_FRAMES:
                pts.clear()
                last_point = None
                filtered_point = None
                if recording and current_stroke:
                    recorded_strokes.append(current_stroke.copy())
                    current_stroke = []
            idx_up_count = 0
            two_up_count = 0
            if lost_track_frames >= LOST_TRACK_GRACE_FRAMES:
                mode = "IDLE"

        # overlay canvas on frame - simplified approach
        output = frame.copy()
        # Create mask for non-black areas of canvas
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        mask = gray_canvas > 10
        # Apply canvas drawings to output where mask is True
        output[mask] = canvas[mask]
        
        # Add real-time cursor indicator on main frame
        if landmarks and point is not None:
            # use the same draw color for cursor to keep color scheme consistent
            cv2.circle(output, point, 8, DRAW_COLOR, -1)
            cv2.circle(output, point, 12, DRAW_COLOR, 2)

        # draw UI
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time) if prev_time else 0
        prev_time = cur_time
        if not recording:
            mode = "IDLE"
        mode_text = f"MODE: {mode} AIR WRITING SYSTEM"
        draw_ui(output, mode_text=mode_text, fps=fps)
        # overlay recognized label
        if recognized_label:
            cv2.putText(output, f"Recognized: {recognized_label} ({recognized_conf:.2f})", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, DRAW_COLOR, 2)
        # overlay accumulated word when not recording
        if not recording and word_buffer:
            cv2.putText(output, f"Word: {word_buffer}", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, DRAW_COLOR, 2)
        # overlay recording / recorded strokes count
        rec_text = f"Strokes: {len(recorded_strokes)}  Recording:{'ON' if recording else 'OFF'}"
        cv2.putText(output, rec_text, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 1)
        if not recording:
            cv2.putText(output, "Recording OFF - input locked", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Bottom output panel (Extracted Text + Speech Output + buttons)
        output_height = 140
        bottom = np.full((output_height, w, 3), 240, dtype=np.uint8)
        # update absolute button rects (full-window coordinates)
        rec_btn_rect[0] = 10
        rec_btn_rect[1] = h + 10
        rec_btn_rect[2] = 230
        rec_btn_rect[3] = h + 60
        play_btn_rect[0] = w - 120
        play_btn_rect[1] = h + 10
        play_btn_rect[2] = w - 20
        play_btn_rect[3] = h + 60
        # draw recording toggle button (on bottom panel coordinates)
        cv2.rectangle(bottom, (10, 10), (230, 50), (50, 50, 50), -1)
        rec_label = f"Recording: {'ON' if recording else 'OFF'}"
        cv2.putText(bottom, rec_label, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # draw play button
        cv2.rectangle(bottom, (w - 120, 10), (w - 20, 50), (50, 50, 50), -1)
        cv2.putText(bottom, "Play", (w - 95, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Draw output labels and text
        cv2.putText(bottom, "Extracted Text Output:", (260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # wrap extracted_text if long
        ex_display = extracted_text if extracted_text else "(none)"
        cv2.putText(bottom, ex_display, (260, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(bottom, "Speech Output:", (260, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        sp_display = speech_text if speech_text else "(none)"
        cv2.putText(bottom, sp_display, (260, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        full_view = np.vstack([output, bottom])
        cv2.imshow("Air Writing System", full_view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            pen_enabled = not pen_enabled
        if key == ord('r'):
            recording = not recording
            if recording:
                # starting a new recording session - clear any old buffer
                recorded_strokes = []
                current_stroke = []
                extracted_text = ""
                speech_text = ""
                last_appended_label = None
                canvas = np.zeros_like(canvas)
                pts.clear()
                last_point = None
                filtered_point = None
                print("🔴 Recording started...")
                recognized_label = None
                recognized_conf = 0.0
            else:
                # recording turned off - finalize and recognize
                print("⏹️ Recording stopped.")
                finalize_recording_and_generate_output()
                recorded_strokes = []
                current_stroke = []
                pts.clear()
                last_point = None
                filtered_point = None
        if key == ord('t'):
            # teach last_hv with user label via input (blocking)
            try:
                label = input('Enter label for last stroke: ')
            except Exception:
                label = 'word'
            if 'last_hv' in locals():
                amem.add(label, last_hv)
                print('Added prototype for', label)
        if key == ord('v'):
            voice_enabled = True
            print("🔊 Voice output is always ON")
        if key == ord('q'):
            break
        if key == ord('c'):
            canvas = np.zeros_like(frame)
            # also clear accumulated recognized word
            word_buffer = ""
            last_appended_label = None
            pts.clear()
            last_point = None
            filtered_point = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()