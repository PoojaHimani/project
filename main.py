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
DRAW_COLOR = (0, 0, 255)  # BGR red
ERASE_COLOR = (0, 0, 0)
THICKNESS = 6


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
    recorded_points = []
    encoder = HDCEncoder(dim=5000, x_bins=64, y_bins=48)
    amem = AssociativeMemory()
    tts = pyttsx3.init()
    voice_enabled = False
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
                    if pen_enabled:
                        cv2.line(canvas, last_point, smoothed_point, DRAW_COLOR, THICKNESS)
                    # always show a small cursor circle at fingertip
                    cv2.circle(canvas, smoothed_point, 4, (0,255,0), -1)
                elif mode == "ERASE":
                    # draw circle filled with black to erase
                    cv2.circle(canvas, smoothed_point, 40, ERASE_COLOR, -1)

                last_point = smoothed_point
                # collect live points for automatic recognition when writing
                if mode == "WRITE" and pen_enabled:
                    live_points.append(smoothed_point)
                    stroke_end_count = 0
            else:
                # no active point; reset smoothing but keep last_point to allow continuity later
                pts.clear()
                last_point = None
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
                            if voice_enabled:
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
                        if voice_enabled:
                            tts.say(label)
                            tts.runAndWait()
                    live_points = []
                    stroke_end_count = 0

        # overlay canvas on frame
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        output = cv2.add(frame_bg, canvas_fg)

        # draw UI
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time) if prev_time else 0
        prev_time = cur_time
        mode_text = f"MODE: {mode} AIR WRITING SYSTEM"
        draw_ui(output, mode_text=mode_text, fps=fps)
        # overlay recognized label
        if recognized_label:
            cv2.putText(output, f"Recognized: {recognized_label} ({recognized_conf:.2f})", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        # overlay recording / live points count
        rec_text = f"LivePts: {len(live_points)}  Recording:{'ON' if recording else 'OFF'}"
        cv2.putText(output, rec_text, (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 1)

        cv2.imshow("Air Writing System", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            pen_enabled = not pen_enabled
        if key == ord('r'):
            recording = not recording
            if not recording and recorded_points:
                # finalize recorded stroke (no label)
                hv = encoder.encode_sequence(recorded_points, (w,h))
                # keep last hv for teaching
                last_hv = hv
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
