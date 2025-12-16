import sys
import time
import json
import base64
import threading

from openai import OpenAI
import pyautogui
import keyboard  # global hotkeys

from PyQt5.QtCore import Qt, QBuffer, QIODevice, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication, QWidget

# ==========================
# CONFIG
# ==========================

# ⚠️ Put your real API key here locally (better: load from env/file)
from dotenv import load_dotenv
load_dotenv()  # <-- THIS loads the .env file automatically


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


PROMPT_ID = "pmpt_6937d1bf42cc81969109e24af757a56e048d4233b9be9cff"
PROMPT_VERSION = "7"  # use the version you are actually testing


class TransparentBorderWindow(QWidget):
    # Signals so the global hotkey thread can safely trigger Qt methods
    capture_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Overlay window setup
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        # Click-through overlay so pyautogui clicks hit what's behind
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self.showFullScreen()
        self.setFocusPolicy(Qt.StrongFocus)
        self.show()

        # For debug draw
        self.last_click_pos = None

        # Connect signals to slots
        self.capture_requested.connect(self.capture_and_send)
        self.quit_requested.connect(QApplication.instance().quit)

        # Start global hotkey listener in a background thread
        t = threading.Thread(target=self._setup_global_hotkeys, daemon=True)
        t.start()

    def _setup_global_hotkeys(self):
        """
        Set up global hotkeys that work even when this window
        does not have focus.
        """
        # Press 'p' anywhere to capture + send
        keyboard.add_hotkey("p", lambda: self.capture_requested.emit())
        # Press 'esc' anywhere to quit
        keyboard.add_hotkey("esc", lambda: self.quit_requested.emit())
        # Keep this thread alive
        keyboard.wait()

    def paintEvent(self, event):
        """
        Draw only a green border around the full screen.
        Also draw a red dot at the last click (debug).
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor(0, 255, 0, 255))  # green border
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRect(rect)

        # Debug: show last click position
        if self.last_click_pos is not None:
            x, y = self.last_click_pos
            pen_dot = QPen(QColor(255, 0, 0, 255))
            painter.setPen(pen_dot)
            painter.setBrush(QColor(255, 0, 0, 180))
            r = 8
            painter.drawEllipse(x - r, y - r, r * 2, r * 2)

    # ---------------------------------------------
    # Capture screen and send to OpenAI's Responses
    # ---------------------------------------------
    def capture_and_send(self):
        app = QApplication.instance()
        screen = app.primaryScreen()

        if not screen:
            print("No screen detected.")
            return

        # Capture full desktop
        pixmap = screen.grabWindow(0)
        img_w, img_h = pixmap.width(), pixmap.height()
        print(f"[{time.strftime('%H:%M:%S')}] Captured: {img_w}x{img_h}")

        # Convert QPixmap -> PNG bytes in memory
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        pixmap.save(buffer, "PNG")
        image_bytes = buffer.data().data()

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Instruction matching the JSON you are getting now (dict with x/y)
        instruction = (
            f"The image resolution is {img_w}x{img_h} pixels. "
            f"Use a 2D coordinate system where (0,0) is TOP-LEFT, "
            f"x to the RIGHT and y DOWN. "
            f"Respond ONLY with a JSON object with exactly these keys: "
            f"'Question read', 'Answer selected', 'Answer location', 'Answer bounding box'. "
            f"'Answer location' must be an object with integer fields 'x' and 'y'. "
            f"'Answer bounding box' must be an object with keys 'top-left' and 'bottom-right', "
            f"each of which is an object with integer 'x' and 'y'. "
            f"All coordinates must be within the image bounds."
        )

        try:
            response = client.responses.create(
                model="gpt-4o",
                prompt={"id": PROMPT_ID, "version": PROMPT_VERSION},
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": instruction},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{b64_image}",
                            },
                        ],
                    }
                ],
                text={"format": {"type": "json_object"}},
            )
        except Exception as e:
            print("OpenAI error:", e)
            return

        # --- 1. Get raw text safely ---
        raw_text = (response.output_text or "").strip()
        print("Raw model output:", raw_text)

        if not raw_text:
            print("Empty model output, nothing to parse.")
            return

        # --- 2. Parse JSON ---
        try:
            data = json.loads(raw_text)
        except Exception as e:
            print("JSON Parse Error:", e)
            return

        print("Parsed JSON:", data)

        # --- 3. Extract coordinates robustly ---

        # ---- Answer location ----
        x_img = y_img = None
        answer_loc = data.get("Answer location") or data.get("answer_location")

        if isinstance(answer_loc, dict):
            # Format: {"x": 1160, "y": 540}
            x_img = answer_loc.get("x")
            y_img = answer_loc.get("y")
            print(f"Model Answer location dict: ({x_img}, {y_img})")
        elif (
            isinstance(answer_loc, (list, tuple))
            and len(answer_loc) == 2
        ):
            # Format: [1160, 540]
            x_img, y_img = answer_loc
            print(f"Model Answer location list: ({x_img}, {y_img})")
        else:
            print("Answer location missing or invalid format:", answer_loc)

        # ---- Bounding box (optionally refine center) ----
        bbox = data.get("Answer bounding box") or data.get(
            "answer_bounding_box")

        if isinstance(bbox, dict):
            # Try keys 'top-left' or 'top_left'
            tl = bbox.get("top-left") or bbox.get("top_left")
            br = bbox.get("bottom-right") or bbox.get("bottom_right")

            def get_xy(value):
                """Support dict {'x':..,'y':..} or list [x, y]."""
                if isinstance(value, dict):
                    return value.get("x"), value.get("y")
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    return value[0], value[1]
                return None, None

            x1, y1 = get_xy(tl)
            x2, y2 = get_xy(br)

            if None not in (x1, y1, x2, y2):
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                print(f"Bounding box top-left={tl}, bottom-right={br}")
                print(f"Using center of bounding box: ({cx}, {cy})")
                x_img, y_img = cx, cy

        if x_img is None or y_img is None:
            print("No valid coordinates to click.")
            return

        # Clamp within image bounds just in case
        x_img = max(0, min(img_w - 1, int(x_img)))
        y_img = max(0, min(img_h - 1, int(y_img)))

        # --- 4. Map to real screen coordinates (scaling) ---
        screen_w, screen_h = pyautogui.size()
        print(f"pyautogui screen size: {screen_w}x{screen_h}")

        scale_x = screen_w / img_w
        scale_y = screen_h / img_h

        x_click = int(x_img * scale_x)
        y_click = int(y_img * scale_y)

        print(f"Clicking at scaled coords: ({x_click}, {y_click})")

        self.last_click_pos = (x_click, y_click)
        self.update()

        time.sleep(0.2)
        pyautogui.click(x_click, y_click)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TransparentBorderWindow()
    sys.exit(app.exec_())
