import sys
import time
import json
import base64
import threading

from openai import OpenAI
import keyboard  # global hotkeys

from PyQt5.QtCore import Qt, QBuffer, QIODevice, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox

import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ==========================
# CONFIG
# ==========================

# ⚠️ Put your real API key here locally (better: load from env/file)
API_KEY = client
client = OpenAI(api_key=API_KEY)

PROMPT_ID = "pmpt_6937d1bf42cc81969109e24af757a56e048d4233b9be9cff"
PROMPT_VERSION = "7"  # use the version you’re actually using on the platform

# Initial window position and size (you can edit these)
INITIAL_X = 500
INITIAL_Y = 500
INITIAL_WIDTH = 900
INITIAL_HEIGHT = 1200


class CaptureWindow(QWidget):
    # Signals so the global hotkey thread can safely trigger Qt methods
    capture_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Frameless, always-on-top, transparent background
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Set initial geometry (you can change constants above)
        self.setGeometry(INITIAL_X, INITIAL_Y, INITIAL_WIDTH, INITIAL_HEIGHT)

        self.setFocusPolicy(Qt.StrongFocus)
        self.show()

        # For dragging the window with the mouse
        self._dragging = False
        self._drag_offset = QPoint(0, 0)

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
        # Press 'p' anywhere to capture + ask GPT
        keyboard.add_hotkey("p", lambda: self.capture_requested.emit())
        # Press 'esc' anywhere to quit
        keyboard.add_hotkey("esc", lambda: self.quit_requested.emit())
        keyboard.wait()

    # -------- Window dragging with mouse --------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.move(event.globalPos() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            event.accept()

    # -------- Visual: transparent area with green border --------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Green border
        pen = QPen(QColor(0, 255, 0, 255))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRect(rect)

    # ---------------------------------------------
    # Capture only the window region and send to GPT
    # ---------------------------------------------
    def capture_and_send(self):
        app = QApplication.instance()
        screen = app.primaryScreen()

        if not screen:
            print("No screen detected.")
            return

        # Get window geometry in screen coordinates
        rect = self.geometry()
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        print(f"[{time.strftime('%H:%M:%S')}] Capture region: {x},{y} {w}x{h}")

        # Temporarily hide the window so the border doesn't appear in capture
        self.hide()
        app.processEvents()

        pixmap = screen.grabWindow(0, x, y, w, h)

        # Show window again
        self.show()
        app.processEvents()

        # Convert QPixmap -> PNG bytes in memory
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        pixmap.save(buffer, "PNG")
        image_bytes = buffer.data().data()

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # We no longer care about coordinates – only question + answer
        instruction = (
            f"You are looking only at the contents of this cropped region. "
            f"The image resolution is {w}x{h} pixels. "
            f"Read the question and determine the correct answer. "
            f"Respond ONLY with a JSON object with exactly these keys: "
            f"'Question read' and 'Answer selected'. "
            f'Example: {{\"Question read\": \"...\", \"Answer selected\": \"...\"}}'
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

        raw_text = (response.output_text or "").strip()
        print("Raw model output:", raw_text)

        if not raw_text:
            print("Empty model output, nothing to parse.")
            return

        try:
            data = json.loads(raw_text)
        except Exception as e:
            print("JSON Parse Error:", e)
            return

        print("Parsed JSON:", data)

        question = data.get("Question read") or data.get("question") or ""
        answer = data.get("Answer selected") or data.get("answer") or ""

        # Show popup with question + answer (no clicking)
        msg = f"Question:\n{question}\n\nAnswer:\n{answer}"
        QMessageBox.information(self, "GPT Answer", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CaptureWindow()
    sys.exit(app.exec_())
