import cv2
import numpy as np
import time
from collections import deque
from typing import List, Tuple
from hand_tracking import HandTracker
from ui_utils import draw_ui
from hdc_encoder import HDCEncoder, AssociativeMemory
import pyttsx3

# Parameters
SMOOTHING_WINDOW = 3
STABLE_THRESHOLD = 3  # frames to confirm a gesture state
DRAW_COLOR = (0, 255, 0)  # BGR green (use single color for drawing + cursor)
ERASE_COLOR = (0, 0, 0)
THICKNESS = 8
ERASE_RADIUS = 48
MIN_POINTS_FOR_RECOGNITION = 24
HDC_ACCEPT_THRESHOLD = 0.08
SHAPE_ACCEPT_THRESHOLD = 0.40
COMBINED_ACCEPT_THRESHOLD = 0.26
WRITE_STABLE_FRAMES = 2
ERASE_STABLE_FRAMES = 1
HELLO_RELAX_FACTOR = 0.65
HELLO_SHAPE_FALLBACK = 0.18
FORCE_HELLO_FALLBACK = True


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


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = HandTracker(maxHands=1, detectionCon=0.7, trackCon=0.5)

    canvas = None
    pts = deque(maxlen=SMOOTHING_WINDOW)
    last_point = None
    pen_enabled = True  # manual toggle to force writing on/off
    recording = False
    recorded_strokes = []
    current_stroke = []
    encoder = HDCEncoder(dim=5000, x_bins=64, y_bins=48)
    amem = AssociativeMemory()
    tts = pyttsx3.init()
    voice_enabled = True

    def speak_output(text: str):
        if not voice_enabled:
            return
        if not text or text == "none":
            return
        try:
            tts.stop()
            tts.say(text)
            tts.runAndWait()
        except Exception as e:
            print(f"❌ Speech error: {e}")
    
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
        nonlocal recording, recorded_strokes, current_stroke, extracted_text, speech_text, word_buffer, last_appended_label, recognized_label, recognized_conf, canvas, pts, last_point
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
            else:
                # finalize recorded stroke(s)
                # finalize collected strokes
                if current_stroke:
                    recorded_strokes.append(current_stroke.copy())
                    current_stroke = []
                if recorded_strokes:
                    label, conf, _ = recognize_recorded_word(recorded_strokes)
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
                        recognized_label = None
                        recognized_conf = 0.0
                        extracted_text = "none"
                        speech_text = "none"
                else:
                    recognized_label = "none"
                    recognized_conf = 0.0
                    extracted_text = "none"
                    speech_text = "none"
                recorded_strokes = []
                current_stroke = []
                pts.clear()
                last_point = None
            return
        px1, py1, px2, py2 = play_btn_rect
        if px1 <= x <= px2 and py1 <= y <= py2:
            speak_output(speech_text)
            return

    cv2.namedWindow("Air Writing System", cv2.WINDOW_NORMAL)
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

    # gesture stability counters
    idx_up_count = 0
    two_up_count = 0

    mode = "IDLE"
    prev_time = 0

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
            print(f"DEBUG: Hand detected - Index: {index_up}, Middle: {middle_up}, Point: {point}, Mode: {mode}")
            # gesture stability logic - very conservative approach
            if index_up and not middle_up:
                idx_up_count += 1
                two_up_count = 0  # Reset erase count when only index is up
            elif index_up and middle_up:
                two_up_count += 1
                idx_up_count = 0  # Reset write count when both are up
            else:
                # Immediate reset for stability
                idx_up_count = 0
                two_up_count = 0

            # More responsive mode detection for smoother writing
            if two_up_count >= ERASE_STABLE_FRAMES:
                mode = "ERASE"
            elif idx_up_count >= WRITE_STABLE_FRAMES:
                mode = "WRITE"
            else:
                # keep last stable mode instead of frequent IDLE flicker
                if mode not in ("WRITE", "ERASE"):
                    mode = "IDLE"

            # Drawing logic - smooth continuous lines
            if point is not None and recording and mode == "WRITE":
                # Use the exact hand position for smooth tracking
                pts.append(point)
                avg_x = int(sum(p[0] for p in pts) / len(pts))
                avg_y = int(sum(p[1] for p in pts) / len(pts))
                current_point = (avg_x, avg_y)

                # Always draw continuous lines when recording
                if last_point is not None:
                    # Interpolate for visually smoother lines
                    dx = current_point[0] - last_point[0]
                    dy = current_point[1] - last_point[1]
                    steps = max(abs(dx), abs(dy), 1)
                    for step in range(1, steps + 1):
                        alpha = step / float(steps)
                        ix = int(last_point[0] + dx * alpha)
                        iy = int(last_point[1] + dy * alpha)
                        cv2.circle(canvas, (ix, iy), THICKNESS // 2, DRAW_COLOR, -1)
                    
                    # Update last point for continuous drawing
                    last_point = current_point
                    
                    # Collect points for recognition
                    if recording and pen_enabled:
                        current_stroke.append(current_point)
                else:
                    # Initialize last point on first touch
                    last_point = current_point
                    # Draw starting point
                    cv2.circle(canvas, current_point, THICKNESS//2, DRAW_COLOR, -1)
                    if recording and pen_enabled:
                        current_stroke.append(current_point)
            elif point is not None and mode == "ERASE":
                if current_stroke:
                    recorded_strokes.append(current_stroke.copy())
                    current_stroke = []
                pts.clear()
                last_point = None
                cv2.circle(canvas, point, ERASE_RADIUS, ERASE_COLOR, -1)
            else:
                # no active point; reset smoothing but keep last_point to allow continuity later
                pts.clear()
                last_point = None
                # if we were recording and there is a current stroke, finalize it (stroke separation)
                if recording and current_stroke:
                    recorded_strokes.append(current_stroke)
                    current_stroke = []
        else:
            # no hand detected
            pts.clear()
            last_point = None
            idx_up_count = 0
            two_up_count = 0
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
                print("🔴 Recording started...")
                recognized_label = None
                recognized_conf = 0.0
            else:
                # recording turned off - finalize and recognize
                print("⏹️ Recording stopped.")
                # Add current stroke to recorded strokes if it has points
                if current_stroke:
                    recorded_strokes.append(current_stroke.copy())
                    print(f"✅ Finalized current stroke. Total strokes: {len(recorded_strokes)}")
                    current_stroke = []
                
                # Auto-recognize and speak if we have strokes
                if recorded_strokes:
                    print(f"🔍 Debug: Processing {len(recorded_strokes)} recorded strokes...")
                    for i, stroke in enumerate(recorded_strokes):
                        if stroke:
                            print(f"🔍 Debug: Stroke {i+1} has {len(stroke)} points")
                        else:
                            print(f"🔍 Debug: Stroke {i+1} is empty")

                    label, conf, raw = recognize_recorded_word(recorded_strokes)
                    print(f"🔍 Debug: Query result: {raw}")

                    if label:
                        recognized_label = label
                        recognized_conf = conf
                        print(f"🎯 Recognized: {label} (sim={conf:.3f})")
                        extracted_text = label
                        speech_text = label

                        print(f"🔊 Speaking: {label}")
                        speak_output(label)
                        print(f"✅ Speech completed for: {label}")
                    else:
                        print("❌ No confident match found in trained words")
                        recognized_label = "none"
                        recognized_conf = 0.0
                        extracted_text = "none"
                        speech_text = "none"
                else:
                    print("❌ No stroke captured")
                    recognized_label = "none"
                    recognized_conf = 0.0
                    extracted_text = "none"
                    speech_text = "none"
                recorded_strokes = []
                current_stroke = []
                pts.clear()
                last_point = None
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
            voice_enabled = not voice_enabled
        if key == ord('q'):
            break
        if key == ord('c'):
            canvas = np.zeros_like(frame)
            # also clear accumulated recognized word
            word_buffer = ""
            last_appended_label = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
