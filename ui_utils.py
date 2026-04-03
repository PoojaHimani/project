import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_ui(frame, mode_text="MODE: WRITE", fps=None):
    h, w, _ = frame.shape
    # top-left mode
    cv2.rectangle(frame, (0,0), (w,40), (0,0,0), -1)
    cv2.putText(frame, mode_text, (10,28), FONT, 0.8, (0,0,255), 2, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(frame, f"FPS: {int(fps)}", (w-120,28), FONT, 0.6, (255,255,255), 1, cv2.LINE_AA)
    # small instruction at bottom
    cv2.putText(frame, "Index Only: Write | Index+Middle: Erase | 'r': Record | 'c': Clear | 'q': Quit",
                (10, h-10), FONT, 0.5, (200,200,200), 1, cv2.LINE_AA)
