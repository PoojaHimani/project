import cv2
import numpy as np
import time
import os
import subprocess
import re
import threading
import queue
from collections import deque
from hand_tracking import HandTracker, HAND_CONNECTIONS
from ui_utils import draw_ui
import pyttsx3
try:
    import winsound
except Exception:
    winsound = None

try:
    import easyocr
except Exception:
    easyocr = None

# Parameters
SMOOTHING_WINDOW = 5
DRAW_COLOR = (0, 0, 255)  # BGR red to match landmark visualization style
ERASE_COLOR = (0, 0, 0)
THICKNESS = 8
WRITE_STABLE_FRAMES = 2
ERASE_STABLE_FRAMES = 5
STROKE_BREAK_FRAMES = 8
FAST_OCR_STROKES_ONLY = False
FAST_OCR_SKIP_FALLBACK = False
FAST_OCR_MAX_POINTS = 900
OCR_CONFIDENCE_STRONG = 0.55
SHOW_FULL_HAND_LANDMARKS = True
OCR_ALLOW_DIGITS = True


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker(maxHands=1, detectionCon=0.7, trackCon=0.5)
    mp_backend_ready = tracker.backend != "none"

    canvas = None
    pts = deque(maxlen=SMOOTHING_WINDOW)
    last_point = None
    recording = False
    recorded_strokes = []
    current_stroke = []
    # accumulate recognized letters into a running word when not recording
    word_buffer = ""
    last_appended_label = None
    # extracted text / speech output
    extracted_text = ""
    speech_text = ""
    pending_speech_text = None
    # GUI button rects (absolute coords; will be updated each frame)
    rec_btn_rect = [10, 10, 230, 60]
    play_btn_rect = [0, 0, 0, 0]

    recognized_label = None
    recognized_conf = 0.0
    ocr_reader = None
    ocr_init_attempted = False
    ocr_lock = threading.Lock()
    ocr_result_queue = queue.Queue()
    # Keep all recognition speech events so each recording-off action gets spoken.
    speech_queue = queue.Queue()
    write_state_count = 0
    erase_state_count = 0
    stroke_break_count = 0
    processing_recognition = False
    mode = "IDLE"
    prev_time = 0

    def _speak_once_blocking(message, engine):
        # Low-latency first path: persistent pyttsx3 engine.
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
            try:
                engine.say(message)
                engine.runAndWait()
                return True
            except Exception:
                pass

        # Fallback: Windows System.Speech via PowerShell.
        if os.name == "nt":
            try:
                safe_msg = message.replace("'", "''")
                ps_cmd = (
                    "$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); "
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "$s.Volume = 100; $s.Rate = 0; "
                    f"$s.Speak('{safe_msg}')"
                )
                res = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_cmd],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                )
                if res.returncode == 0:
                    return True
            except Exception:
                pass

        if os.name == "nt" and winsound is not None:
            try:
                winsound.MessageBeep(winsound.MB_OK)
            except Exception:
                pass
        return False

    def _speech_worker():
        while True:
            msg = speech_queue.get()
            if msg is None:
                break
            engine = None
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 185)
                engine.setProperty("volume", 1.0)
            except Exception:
                engine = None

            _speak_once_blocking(msg, engine)

            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    speech_thread = threading.Thread(target=_speech_worker, daemon=True)
    speech_thread.start()

    def speak_text(msg, priority=False):
        if not msg:
            return
        message = str(msg).strip()
        if not message:
            message = "No text recognized"

        # Priority speech should be heard right away (used for recording-off prompt).
        if priority:
            try:
                while True:
                    _ = speech_queue.get_nowait()
            except queue.Empty:
                pass
        speech_queue.put(message)

    def _normalize_ocr_text(text):
        cleaned = re.sub(r"[^A-Za-z0-9 ]", " ", text or "")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    def _postprocess_ocr_text(text):
        # Correct common OCR confusions for handwritten alphabetic words.
        if not text:
            return ""

        fixed = text
        if not OCR_ALLOW_DIGITS:
            fixed = fixed.translate(str.maketrans({
                "0": "o",
                "1": "l",
                "3": "e",
                "5": "s",
                "6": "g",
                "8": "b",
            }))
        fixed = re.sub(r"\s+", " ", fixed).strip()
        return fixed

    def _build_ocr_image(strokes):
        points = [p for stroke in strokes for p in stroke]
        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        margin = 40
        min_x = max(0, min(xs) - margin)
        min_y = max(0, min(ys) - margin)
        max_x = max(xs) + margin
        max_y = max(ys) + margin

        crop_w = max(64, max_x - min_x + 1)
        crop_h = max(64, max_y - min_y + 1)

        # Smaller target for faster OCR turnaround when recording turns off.
        target = 192
        scale = min((target - 32) / float(crop_w), (target - 32) / float(crop_h))
        ox = int((target - crop_w * scale) / 2.0)
        oy = int((target - crop_h * scale) / 2.0)

        img = np.full((target, target), 255, dtype=np.uint8)
        for stroke in strokes:
            if not stroke:
                continue
            # Downsample long strokes to speed up OCR image rendering.
            step = 1
            if len(stroke) > FAST_OCR_MAX_POINTS:
                step = max(1, len(stroke) // FAST_OCR_MAX_POINTS)
            sampled = stroke[::step]
            if len(sampled) < 2:
                continue
            shifted = []
            for x, y in sampled:
                sx = int((x - min_x) * scale + ox)
                sy = int((y - min_y) * scale + oy)
                shifted.append((sx, sy))
            for i in range(1, len(shifted)):
                cv2.line(img, shifted[i - 1], shifted[i], 0, 14, cv2.LINE_AA)

        img = cv2.GaussianBlur(img, (5, 5), 0)
        kernel = np.ones((2, 2), dtype=np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        return img

    def _build_ocr_image_from_canvas(canvas_img):
        if canvas_img is None:
            return None

        gray = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2GRAY)
        mask = gray > 10
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None

        margin = 25
        min_x = max(0, int(np.min(xs)) - margin)
        min_y = max(0, int(np.min(ys)) - margin)
        max_x = min(gray.shape[1] - 1, int(np.max(xs)) + margin)
        max_y = min(gray.shape[0] - 1, int(np.max(ys)) + margin)

        crop = gray[min_y:max_y + 1, min_x:max_x + 1]
        if crop.size == 0:
            return None

        # Convert colored strokes to dark text on white background for OCR.
        inv = cv2.bitwise_not(crop)
        target = 224
        h, w = inv.shape
        scale = min((target - 40) / float(max(1, w)), (target - 40) / float(max(1, h)))
        rw = max(1, int(w * scale))
        rh = max(1, int(h * scale))
        resized = cv2.resize(inv, (rw, rh), interpolation=cv2.INTER_CUBIC)
        canvas_norm = np.full((target, target), 255, dtype=np.uint8)
        ox = (target - rw) // 2
        oy = (target - rh) // 2
        canvas_norm[oy:oy + rh, ox:ox + rw] = resized

        canvas_norm = cv2.GaussianBlur(canvas_norm, (5, 5), 0)
        kernel = np.ones((2, 2), dtype=np.uint8)
        canvas_norm = cv2.morphologyEx(canvas_norm, cv2.MORPH_CLOSE, kernel, iterations=1)
        return canvas_norm

    def _build_ocr_variants(ocr_img):
        if ocr_img is None:
            return []

        variants = [ocr_img]
        sharpen = cv2.GaussianBlur(ocr_img, (0, 0), 1.0)
        sharpen = cv2.addWeighted(ocr_img, 1.4, sharpen, -0.4, 0)
        variants.append(sharpen)
        adaptive = cv2.adaptiveThreshold(
            ocr_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            5,
        )
        variants.append(adaptive)
        return variants

    def _score_candidate(text, conf):
        if not text:
            return -1.0
        alpha = sum(1 for ch in text if ch.isalpha())
        digits = sum(1 for ch in text if ch.isdigit())
        spaces = sum(1 for ch in text if ch.isspace())
        word_bonus = 0.12 if alpha >= 3 else 0.0
        return float(conf) + 0.02 * len(text) + 0.08 * alpha + word_bonus - 0.02 * digits - 0.01 * spaces

    def _extract_ocr_candidates(ocr_img, ocr_engine, allow):
        candidates = []
        if ocr_img is None:
            return candidates

        try:
            res_para = ocr_engine.readtext(
                ocr_img,
                detail=0,
                paragraph=False,
                allowlist=allow,
                decoder="greedy",
            )
        except Exception:
            res_para = []

        for txt in res_para:
            cleaned = _normalize_ocr_text(txt)
            if cleaned:
                candidates.append((cleaned, 0.66 + min(0.28, 0.020 * len(cleaned))))

        if not candidates and not FAST_OCR_SKIP_FALLBACK:
            try:
                res_detail = ocr_engine.readtext(
                    ocr_img,
                    detail=1,
                    paragraph=False,
                    allowlist=allow,
                    decoder="greedy",
                    beamWidth=2,
                )
            except Exception:
                res_detail = []

            if not res_detail:
                try:
                    res_detail = ocr_engine.readtext(
                        ocr_img,
                        detail=1,
                        paragraph=False,
                        allowlist=allow,
                        decoder="beamsearch",
                        beamWidth=5,
                    )
                except Exception:
                    res_detail = []

            if res_detail:
                ordered = sorted(res_detail, key=lambda r: min(p[0] for p in r[0]))
                joined = _normalize_ocr_text(" ".join(r[1] for r in ordered))
                if joined:
                    avg_conf = float(sum(float(r[2]) for r in ordered) / len(ordered))
                    candidates.append((joined, avg_conf))
                for _, txt, conf in ordered:
                    cleaned = _normalize_ocr_text(txt)
                    if cleaned:
                        candidates.append((cleaned, float(conf)))

        return candidates

    def recognize_with_easyocr(strokes, canvas_snapshot):
        nonlocal ocr_reader, ocr_init_attempted
        if not strokes and canvas_snapshot is None:
            return "", 0.0

        if easyocr is None:
            return "", 0.0

        def ensure_reader():
            nonlocal ocr_reader, ocr_init_attempted
            with ocr_lock:
                if ocr_reader is not None:
                    return True
                if ocr_init_attempted:
                    return False
                ocr_init_attempted = True
                try:
                    ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                    return True
                except Exception:
                    ocr_reader = None
                    return False

        if ocr_reader is None and not ensure_reader():
            return "", 0.0

        allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "
        if OCR_ALLOW_DIGITS:
            allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
        candidates = []

        stroke_img = _build_ocr_image(strokes) if strokes else None
        stroke_variants = _build_ocr_variants(stroke_img)
        for variant in stroke_variants:
            candidates.extend(_extract_ocr_candidates(variant, ocr_reader, allow))

        # Use canvas OCR only if stroke-based confidence is weak.
        stroke_best_score = max((_score_candidate(t, c) for t, c in candidates), default=-1.0)
        if (not FAST_OCR_STROKES_ONLY) or stroke_best_score < (OCR_CONFIDENCE_STRONG + 0.20):
            canvas_img = _build_ocr_image_from_canvas(canvas_snapshot)
            for variant in _build_ocr_variants(canvas_img):
                candidates.extend(_extract_ocr_candidates(variant, ocr_reader, allow))

        if not candidates:
            return "", 0.0

        best_text, best_conf = max(candidates, key=lambda item: _score_candidate(item[0], item[1]))
        best_text = _postprocess_ocr_text(best_text)
        return best_text, float(min(0.99, best_conf))

    # Preload OCR models in the background to avoid first recording-off delay.
    if easyocr is not None:
        threading.Thread(target=lambda: recognize_with_easyocr([[(0, 0), (1, 1)]], None), daemon=True).start()

    def start_recognition_async(strokes_copy, canvas_snapshot):
        def _worker():
            if not strokes_copy and canvas_snapshot is None:
                ocr_result_queue.put({
                    "text": "(no input captured)",
                    "conf": 0.0,
                    "ok": False,
                })
                return

            text, conf = recognize_with_easyocr(strokes_copy, canvas_snapshot)
            if text:
                ocr_result_queue.put({
                    "text": text,
                    "conf": conf,
                    "ok": True,
                })
            else:
                ocr_result_queue.put({
                    "text": "(no text recognized)" if easyocr is not None else "(EasyOCR not installed)",
                    "conf": 0.0,
                    "ok": False,
                })

        threading.Thread(target=_worker, daemon=True).start()

    def finalize_recording_output(frame_w, frame_h):
        nonlocal extracted_text, speech_text, recorded_strokes, current_stroke, processing_recognition
        nonlocal word_buffer, last_appended_label, recognized_label, recognized_conf
        nonlocal pending_speech_text

        if current_stroke:
            recorded_strokes.append(current_stroke.copy())
            current_stroke = []

        strokes_copy = [s.copy() for s in recorded_strokes if s]
        canvas_snapshot = None
        if (not FAST_OCR_STROKES_ONLY) and canvas is not None:
            canvas_snapshot = canvas.copy()
        recorded_strokes = []
        extracted_text = "(processing...)"
        speech_text = extracted_text
        pending_speech_text = None
        recognized_label = None
        recognized_conf = 0.0
        processing_recognition = True
        start_recognition_async(strokes_copy, canvas_snapshot)

    # Mouse handler for clickable buttons (will be attached to window)
    def on_mouse(event, x, y, flags, param):
        nonlocal recording, recorded_strokes, current_stroke, extracted_text, speech_text, word_buffer, last_appended_label
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
            else:
                finalize_recording_output(frame.shape[1], frame.shape[0])
            return
        px1, py1, px2, py2 = play_btn_rect
        if px1 <= x <= px2 and py1 <= y <= py2:
            if speech_text:
                speak_text(speech_text)
            return

    cv2.namedWindow("Air Writing System", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Air Writing System", on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        if canvas is None:
            canvas = np.zeros_like(frame)

        landmarks = tracker.find_hands(frame, draw=False)
        point = None
        if landmarks:
            lm = landmarks[0]
            index_up, middle_up = tracker.fingers_up(lm)
            point = tracker.get_point(lm)

            if index_up and not middle_up:
                write_state_count += 1
                erase_state_count = 0
            elif index_up and middle_up:
                erase_state_count += 1
                write_state_count = 0
            else:
                write_state_count = 0
                erase_state_count = 0

            if erase_state_count >= ERASE_STABLE_FRAMES:
                mode = "ERASE"
            elif write_state_count >= WRITE_STABLE_FRAMES:
                mode = "WRITE"
            else:
                mode = "IDLE"

            # Drawing logic
            if point is not None and (mode == "WRITE" or mode == "ERASE"):
                pts.append(point)
                # smoothing
                avg_x = int(sum(p[0] for p in pts) / len(pts))
                avg_y = int(sum(p[1] for p in pts) / len(pts))
                smoothed_point = (avg_x, avg_y)

                if last_point is None:
                    last_point = smoothed_point

                # draw on canvas
                if mode == "WRITE":
                    # Draw only while recording is enabled.
                    if recording and last_point is not None:
                        cv2.line(canvas, last_point, smoothed_point, DRAW_COLOR, THICKNESS, cv2.LINE_AA)
                    stroke_break_count = 0
                elif mode == "ERASE":
                    # Erase only while recording is enabled.
                    if recording:
                        if current_stroke:
                            recorded_strokes.append(current_stroke)
                            current_stroke = []
                        cv2.circle(canvas, smoothed_point, 40, ERASE_COLOR, -1)
                    stroke_break_count = STROKE_BREAK_FRAMES

                last_point = smoothed_point
                if mode == "WRITE" and recording:
                    current_stroke.append(smoothed_point)
            else:
                # Keep a short grace window so brief tracking drops do not break writing.
                pts.clear()
                stroke_break_count += 1
                if stroke_break_count >= STROKE_BREAK_FRAMES:
                    last_point = None
                    if recording and current_stroke:
                        recorded_strokes.append(current_stroke)
                        current_stroke = []
        else:
            # no hand detected
            pts.clear()
            stroke_break_count += 1
            if stroke_break_count >= STROKE_BREAK_FRAMES:
                last_point = None
            write_state_count = 0
            erase_state_count = 0
            mode = "IDLE"
            if recording and current_stroke and stroke_break_count >= STROKE_BREAK_FRAMES:
                recorded_strokes.append(current_stroke)
                current_stroke = []

        # Apply background OCR result as soon as it is ready.
        try:
            res = ocr_result_queue.get_nowait()
            processing_recognition = False
            extracted_text = res["text"]
            speech_text = extracted_text
            recognized_conf = float(res.get("conf", 0.0))
            if res.get("ok", False):
                recognized_label = extracted_text
                word_buffer += extracted_text
                last_appended_label = extracted_text[-1] if extracted_text else None
            else:
                recognized_label = None
            pending_speech_text = speech_text
        except queue.Empty:
            pass

        # overlay canvas on frame - simplified approach
        output = frame.copy()
        # Create mask for non-black areas of canvas
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        mask = gray_canvas > 10
        # Apply canvas drawings to output where mask is True
        output[mask] = canvas[mask]

        # Draw full hand landmarks on top of the final composited frame.
        if SHOW_FULL_HAND_LANDMARKS and landmarks:
            for hand in landmarks:
                lm_map = {pid: (px, py) for pid, px, py in hand}
                for a, b in HAND_CONNECTIONS:
                    if a in lm_map and b in lm_map:
                        cv2.line(output, lm_map[a], lm_map[b], (255, 255, 255), 2)
                for _, lx, ly in hand:
                    cv2.circle(output, (lx, ly), 3, (0, 0, 255), -1)
                    cv2.circle(output, (lx, ly), 5, (255, 255, 255), 1)
        
        # Add real-time cursor indicator on main frame
        if landmarks and point is not None:
            # use the same draw color for cursor to keep color scheme consistent
            cv2.circle(output, point, 8, DRAW_COLOR, -1)
            cv2.circle(output, point, 12, DRAW_COLOR, 2)

        if not mp_backend_ready:
            cv2.putText(
                output,
                "MediaPipe landmarks backend not available. Install mediapipe compatible with your Python.",
                (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        # draw UI
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time) if prev_time else 0
        prev_time = cur_time
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
        if pending_speech_text:
            speak_text(pending_speech_text, priority=True)
            pending_speech_text = None
        cv2.imshow("Air Writing System", full_view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = not recording
            if recording:
                recorded_strokes = []
                current_stroke = []
                extracted_text = ""
                speech_text = ""
                pending_speech_text = None
                last_appended_label = None
            else:
                finalize_recording_output(w, h)
        if key == ord('q'):
            break
        if key == ord('c'):
            canvas = np.zeros_like(frame)
            # also clear accumulated recognized word
            word_buffer = ""
            last_appended_label = None

    cap.release()
    try:
        speech_queue.put(None)
    except Exception:
        pass
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
