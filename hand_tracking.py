import cv2
try:
    import mediapipe as mp
except Exception:
    mp = None
import numpy as np
"""
This module supports two modes:
- If `mp.solutions` is available (older MediaPipe API), use MediaPipe Hands.
- Otherwise, fall back to a lightweight OpenCV contour-based heuristic so
  the app still runs when only newer MediaPipe (tasks API) is installed
  or when MediaPipe is not available.

The fallback is less accurate but good enough for demo/writing use.
"""

def _has_mp_solutions():
    try:
        return hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands')
    except Exception:
        return False

class HandTracker:
    def __init__(self, maxHands=1, detectionCon=0.7, trackCon=0.5):
        self.maxHands = maxHands
        self.use_mp = _has_mp_solutions()
        if self.use_mp:
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(static_image_mode=False,
                                            max_num_hands=self.maxHands,
                                            min_detection_confidence=detectionCon,
                                            min_tracking_confidence=trackCon)
            self.mpDraw = mp.solutions.drawing_utils
        else:
            # fallback parameters
            self.min_area = 3000
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    def find_hands(self, frame, draw=True):
        """Return list of hands; each hand is a list of (id,x,y) tuples.
        For compatibility with the rest of the app we synthesize landmarks
        for ids we need (8,6,12,10)."""
        if self.use_mp:
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
                        self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
            return all_landmarks
        else:
            # Fallback: simple skin segmentation + largest contour fingertip heuristic
            h, w, _ = frame.shape
            img_blur = cv2.GaussianBlur(frame, (5,5), 0)
            img_ycrcb = cv2.cvtColor(img_blur, cv2.COLOR_BGR2YCrCb)
            # skin color range in YCrCb
            lower = np.array([0, 133, 77], dtype=np.uint8)
            upper = np.array([255, 173, 127], dtype=np.uint8)
            mask = cv2.inRange(img_ycrcb, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_landmarks = []
            if contours:
                c = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(c)
                if area > self.min_area:
                    # centroid
                    M = cv2.moments(c)
                    if M['m00'] == 0:
                        return []
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    # find farthest contour point from centroid -> fingertip
                    pts = c.reshape(-1,2)
                    dists = np.linalg.norm(pts - np.array([cx,cy]), axis=1)
                    idx_far = np.argmax(dists)
                    tip = tuple(pts[idx_far].tolist())

                    # try to detect a second fingertip (middle) by removing neighbourhood
                    pts2 = np.delete(pts, np.s_[max(0, idx_far-10):idx_far+10], axis=0)
                    if len(pts2) > 0:
                        dists2 = np.linalg.norm(pts2 - np.array([cx,cy]), axis=1)
                        idx_far2 = np.argmax(dists2)
                        tip2 = tuple(pts2[idx_far2].tolist())
                        # if second fingertip is sufficiently far from first, consider two fingers
                        two_fingers = np.linalg.norm(np.array(tip)-np.array(tip2)) > 40
                    else:
                        tip2 = None
                        two_fingers = False

                    # create synthetic landmarks list
                    single_hand = []
                    # id 8: index tip
                    single_hand.append((8, int(tip[0]), int(tip[1])))
                    # id 6: index pip -> point between centroid and tip
                    pip = (int((tip[0]+cx)/2), int((tip[1]+cy)/2))
                    single_hand.append((6, pip[0], pip[1]))
                    if two_fingers and tip2 is not None:
                        single_hand.append((12, int(tip2[0]), int(tip2[1])))
                        pip2 = (int((tip2[0]+cx)/2), int((tip2[1]+cy)/2))
                        single_hand.append((10, pip2[0], pip2[1]))
                    all_landmarks.append(single_hand)

                    if draw:
                        cv2.drawContours(frame, [c], -1, (0,255,0), 2)
                        cv2.circle(frame, tip, 8, (0,0,255), -1)
                        if tip2 is not None and two_fingers:
                            cv2.circle(frame, tip2, 8, (0,255,255), -1)
            return all_landmarks

    def fingers_up(self, landmarks):
        # Accepts landmarks list for one hand (list of (id,x,y)). Returns (index_up, middle_up)
        # Index tip id 8, pip id 6. Middle tip 12, pip 10. y smaller => finger is up (image coord)
        if not landmarks:
            return False, False
        # Map id->(x,y)
        lm = {p[0]: (p[1], p[2]) for p in landmarks}
        index_up = False
        middle_up = False
        try:
            # For both MediaPipe and fallback, smaller y is up on image coordinates
            index_up = lm[8][1] < lm[6][1]
            middle_up = False
            if 12 in lm and 10 in lm:
                middle_up = lm[12][1] < lm[10][1]
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
