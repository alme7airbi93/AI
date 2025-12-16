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
PROMPT_VERSION = "7"  # update if you create a new version


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
        Interior is fully transparent.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor(0, 255, 0, 255))  # green border
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRect(rect)

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

        # Instruction to force JSON + precise coordinate system + bounding box
        instruction = instruction = (
            f"The image resolution is {img_w}x{img_h} pixels. "
            f"Remember: (0,0) is TOP-LEFT, x to the RIGHT, y DOWN. "
            f"Respond ONLY with a JSON object following the schema you were instructed: "
            f"keys 'Question read', 'Answer selected', 'Answer location', 'Answer bounding box'. "
            f"'Answer location' must be the CENTER of 'Answer bounding box' in this pixel system."
        )

        try:
            response = client.responses.create(
                model="gpt-4o",
                prompt={
                    "id": PROMPT_ID,
                    "version": PROMPT_VERSION,
                },
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": instruction
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{b64_image}"
                            }
                        ]
                    }
                ],
                text={"format": {"type": "json_object"}},
            )
        except Exception as e:
            print("OpenAI error:", e)
            return
        print(response)
        raw_text = response.output_text
        print("Raw model output:", raw_text)

        try:
            data = json.loads(raw_text)
        except Exception as e:
            print("JSON Parse Error:", e)
            return

        print("Parsed JSON:", data)

        # --- Read answer location + bounding box ---
        answer_loc = data.get("Answer location") or data.get("answer_location")
        bbox = data.get("Answer bounding box") or data.get(
            "answer_bounding_box")

        # Default center = from Answer location
        x_img = y_img = None
        if answer_loc and len(answer_loc) == 2:
            x_img, y_img = answer_loc
            print(f"Model Answer location: ({x_img}, {y_img})")
        else:
            print("No 'Answer location' provided or invalid.")

        # If bounding box exists, compute center from it (more stable)
        if isinstance(bbox, dict):
            top_left = bbox.get("top_left")
            bottom_right = bbox.get("bottom_right")
            if (
                isinstance(top_left, list) and len(top_left) == 2 and
                isinstance(bottom_right, list) and len(bottom_right) == 2
            ):
                x1, y1 = top_left
                x2, y2 = bottom_right
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                print(
                    f"Bounding box top_left={top_left}, bottom_right={bottom_right}")
                print(f"Using center of bounding box: ({cx}, {cy})")
                x_img, y_img = cx, cy

        if x_img is None or y_img is None:
            print("No valid coordinates to click.")
            return

        # Get actual screen size from pyautogui (may differ due to scaling)
        screen_w, screen_h = pyautogui.size()
        print(f"pyautogui screen size: {screen_w}x{screen_h}")

        # Scale from image coordinate system to real screen coordinate system
        scale_x = screen_w / img_w
        scale_y = screen_h / img_h

        x_click = int(x_img * scale_x)
        y_click = int(y_img * scale_y)

        print(f"Clicking at scaled coords: ({x_click}, {y_click})")

        time.sleep(0.2)
        pyautogui.click(x_click, y_click)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TransparentBorderWindow()
    sys.exit(app.exec_())
