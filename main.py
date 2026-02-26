import cv2
import numpy as np
import time
from collections import deque
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
    voice_enabled = False
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
                # finalize recorded stroke(s)
                # finalize collected strokes
                if recorded_strokes:
                    extracted_text = ""
                    for stroke in recorded_strokes:
                        if not stroke:
                            continue
                        hv = encoder.encode_sequence(stroke, (frame.shape[1], frame.shape[0]))
                        res = amem.query(hv, topk=1)
                        if res:
                            label, conf = res[0]
                            if conf >= RECOG_THRESHOLD:
                                extracted_text += label
                                if label != last_appended_label:
                                    word_buffer += label
                                    last_appended_label = label
                    speech_text = extracted_text
                    recorded_strokes = []
                    current_stroke = []
            return
        px1, py1, px2, py2 = play_btn_rect
        if px1 <= x <= px2 and py1 <= y <= py2:
            if speech_text:
                tts.say(speech_text)
                tts.runAndWait()
            return

    cv2.namedWindow("Air Writing System", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Air Writing System", on_mouse)
    # automatic stroke capture
    live_points = []
    stroke_end_count = 0
    STROKE_END_FRAMES = 6
    recognized_label = None
    recognized_conf = 0.0
    RECOG_THRESHOLD = 0.45

    # gesture stability counters
    idx_up_count = 0
    two_up_count = 0

    mode = "WRITE"
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
            # gesture stability logic
            if index_up and not middle_up:
                idx_up_count += 1
            else:
                idx_up_count = 0
            if index_up and middle_up:
                two_up_count += 1
            else:
                two_up_count = 0

            # Decide mode based on stable counts (or manual toggle)
            if two_up_count >= STABLE_THRESHOLD:
                mode = "ERASE"
            elif idx_up_count >= STABLE_THRESHOLD or pen_enabled:
                # if pen is manually enabled, treat as write
                if idx_up_count >= 1 or pen_enabled:
                    mode = "WRITE"
                else:
                    mode = "IDLE"
            else:
                mode = "IDLE"

            # Drawing logic - use exact hand position, no smoothing
            if point is not None and (mode == "WRITE" or mode == "ERASE"):
                # Use the exact point from hand tracking, no smoothing
                current_point = point

                if last_point is None:
                    last_point = current_point

                # draw on canvas
                if mode == "WRITE":
                    # Only draw when recording is ON
                    if recording:
                        if last_point is not None:
                            cv2.line(canvas, last_point, current_point, DRAW_COLOR, THICKNESS)
                        # cursor at fingertip when recording
                        cv2.circle(canvas, current_point, 6, DRAW_COLOR, -1)
                elif mode == "ERASE":
                    # allow erasing only when recording ON (canvas editable)
                    if recording:
                        cv2.circle(canvas, current_point, 40, ERASE_COLOR, -1)

                last_point = current_point
                # collect points only when recording (app is editable only in recording)
                if mode == "WRITE" and recording and pen_enabled:
                    current_stroke.append(current_point)
                    stroke_end_count = 0
            else:
                # no active point; reset smoothing but keep last_point to allow continuity later
                pts.clear()
                last_point = None
                # if we were recording and there is a current stroke, finalize it (stroke separation)
                if recording and current_stroke:
                    recorded_strokes.append(current_stroke)
                    current_stroke = []
                # if we have live points and the finger lifted, count toward stroke end
                if live_points:
                    stroke_end_count += 1
                    if stroke_end_count >= STROKE_END_FRAMES:
                        # finalize stroke
                        hv = encoder.encode_sequence(live_points, (w, h))
                        # automatic recognition
                        res = amem.query(hv, topk=1)
                        if res:
                            label, conf = res[0]
                            recognized_label = label
                            recognized_conf = conf
                            print(f"Recognized: {label} (sim={conf:.3f})")
                            # when not recording, append recognized labels to the running word
                            if not recording and conf >= RECOG_THRESHOLD:
                                # append label to word buffer (avoid immediate duplicates)
                                if label != last_appended_label:
                                    word_buffer += label
                                    last_appended_label = label
                                    print(f"Word now: {word_buffer}")
                                    if voice_enabled:
                                        tts.say(word_buffer)
                                        tts.runAndWait()
                            # also optionally speak the single label if recording (original behavior)
                            elif voice_enabled:
                                tts.say(label)
                                tts.runAndWait()
                        else:
                            recognized_label = None
                            recognized_conf = 0.0
                        live_points = []
                        stroke_end_count = 0
        else:
            # no hand detected
            pts.clear()
            last_point = None
            idx_up_count = 0
            two_up_count = 0
            mode = "IDLE"
            # if hand lost while we were collecting points, count toward stroke end
            if live_points:
                stroke_end_count += 1
                if stroke_end_count >= STROKE_END_FRAMES:
                    hv = encoder.encode_sequence(live_points, (w, h))
                    res = amem.query(hv, topk=1)
                    if res:
                        label, conf = res[0]
                        recognized_label = label
                        recognized_conf = conf
                        print(f"Recognized: {label} (sim={conf:.3f})")
                        if not recording and conf >= RECOG_THRESHOLD:
                            if label != last_appended_label:
                                word_buffer += label
                                last_appended_label = label
                                print(f"Word now: {word_buffer}")
                                if voice_enabled:
                                    tts.say(word_buffer)
                                    tts.runAndWait()
                        elif voice_enabled:
                            tts.say(label)
                            tts.runAndWait()
                    live_points = []
                    stroke_end_count = 0

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
        cv2.imshow("Air Writing System", full_view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            pen_enabled = not pen_enabled
        if key == ord('r'):
            recording = not recording
            if recording:
                # starting a new recording session - clear any old buffer
                recorded_points = []
                last_hv = None
            else:
                # recording turned off - finalize and recognize the recorded stroke
                if recorded_points:
                    hv = encoder.encode_sequence(recorded_points, (w,h))
                    res = amem.query(hv, topk=1)
                    if res:
                        label, conf = res[0]
                        recognized_label = label
                        recognized_conf = conf
                        print(f"Recorded Recognized: {label} (sim={conf:.3f})")
                        if conf >= RECOG_THRESHOLD:
                            if label != last_appended_label:
                                word_buffer += label
                                last_appended_label = label
                                print(f"Word now: {word_buffer}")
                                extracted_text = label
                                speech_text = label
                                if voice_enabled:
                                    tts.say(word_buffer)
                                    tts.runAndWait()
                    recorded_points = []
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
