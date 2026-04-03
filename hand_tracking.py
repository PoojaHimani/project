import cv2
try:
    import mediapipe as mp
except Exception:
    mp = None
import os
import time
import urllib.request

"""Hand tracking via MediaPipe landmarks only (solutions or tasks API)."""

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def _has_mp_solutions():
    try:
        return hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands')
    except Exception:
        return False


def _has_mp_tasks_hand_landmarker():
    try:
        return (
            hasattr(mp, 'tasks')
            and hasattr(mp.tasks, 'vision')
            and hasattr(mp.tasks.vision, 'HandLandmarker')
            and hasattr(mp.tasks, 'BaseOptions')
        )
    except Exception:
        return False


def _ensure_task_model(model_path):
    if os.path.exists(model_path):
        return True
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(TASK_MODEL_URL, model_path)
        return True
    except Exception:
        return False

class HandTracker:
    def __init__(self, maxHands=1, detectionCon=0.7, trackCon=0.5):
        self.maxHands = maxHands
        self.backend = "none"
        self.use_mp = False

        if _has_mp_solutions():
            self.backend = "solutions"
            self.use_mp = True
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(static_image_mode=False,
                                            max_num_hands=self.maxHands,
                                            min_detection_confidence=detectionCon,
                                            min_tracking_confidence=trackCon)
            self.mpDraw = mp.solutions.drawing_utils
            self.lm_style = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)
            self.conn_style = self.mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1)
            self.tasks_landmarker = None
        elif _has_mp_tasks_hand_landmarker():
            self.backend = "tasks"
            self.use_mp = True
            self.mpHands = None
            self.hands = None
            self.mpDraw = None
            self.tasks_landmarker = None

            model_path = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")
            if _ensure_task_model(model_path):
                try:
                    options = mp.tasks.vision.HandLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                        running_mode=mp.tasks.vision.RunningMode.VIDEO,
                        num_hands=self.maxHands,
                        min_hand_detection_confidence=detectionCon,
                        min_hand_presence_confidence=detectionCon,
                        min_tracking_confidence=trackCon,
                    )
                    self.tasks_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
                except Exception:
                    self.backend = "none"
                    self.use_mp = False
        else:
            self.mpHands = None
            self.hands = None
            self.mpDraw = None
            self.tasks_landmarker = None

    def find_hands(self, frame, draw=True):
        """Return list of hands; each hand is a list of (id,x,y) tuples.
        For compatibility with the rest of the app we synthesize landmarks
        for ids we need (8,6,12,10)."""
        if self.backend == "solutions":
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.results = self.hands.process(imgRGB)
            all_landmarks = []
            h, w, _ = frame.shape
            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    single_hand = []
                    for id, lm in enumerate(handLms.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        single_hand.append((id, cx, cy))
                    all_landmarks.append(single_hand)
                    if draw:
                        self.mpDraw.draw_landmarks(
                            frame,
                            handLms,
                            self.mpHands.HAND_CONNECTIONS,
                            self.lm_style,
                            self.conn_style,
                        )
            return all_landmarks
        elif self.backend == "tasks" and self.tasks_landmarker is not None:
            h, w, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            timestamp_ms = int(time.time() * 1000)

            try:
                result = self.tasks_landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception:
                return []

            all_landmarks = []
            if result.hand_landmarks:
                for hand_lms in result.hand_landmarks:
                    single_hand = []
                    for idx, lm in enumerate(hand_lms):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        single_hand.append((idx, cx, cy))
                    all_landmarks.append(single_hand)

                    if draw:
                        lm_map = {pid: (px, py) for pid, px, py in single_hand}
                        for a, b in HAND_CONNECTIONS:
                            if a in lm_map and b in lm_map:
                                cv2.line(frame, lm_map[a], lm_map[b], (255, 255, 255), 2)
                        for _, lx, ly in single_hand:
                            cv2.circle(frame, (lx, ly), 3, (0, 0, 255), -1)
                            cv2.circle(frame, (lx, ly), 5, (255, 255, 255), 1)
            return all_landmarks
        else:
            return []

    def fingers_up(self, landmarks):
        # Accepts landmarks list for one hand (list of (id,x,y)). Returns (index_up, middle_up)
        # Index tip id 8, pip id 6. Middle tip 12, pip 10. y smaller => finger is up (image coord)
        if not landmarks:
            return False, False
        # Map id->(x,y)
        lm = {p[0]: (p[1], p[2]) for p in landmarks}
        index_up = False
        middle_up = False
        index_up_margin = 8   # lower margin makes index-up easier to hold while writing
        middle_up_margin = 16  # stricter middle-up helps avoid accidental erase mode
        try:
            # Smaller y is up in image coordinates; require a margin for stability.
            index_up = lm[8][1] < (lm[6][1] - index_up_margin)
            middle_up = False
            if 12 in lm and 10 in lm:
                raw_middle_up = lm[12][1] < (lm[10][1] - middle_up_margin)
                # Require visible separation between index and middle fingertips.
                tip_dx = lm[12][0] - lm[8][0]
                tip_dy = lm[12][1] - lm[8][1]
                tip_sep = (tip_dx * tip_dx + tip_dy * tip_dy) ** 0.5
                middle_up = raw_middle_up and tip_sep > 42
        except KeyError:
            pass
        return index_up, middle_up

    def get_point(self, landmarks):
        # returns (x,y) of index fingertip if available
        if not landmarks:
            return None
        for id, x, y in landmarks:
            if id == 8:
                return (x, y)
        return None
