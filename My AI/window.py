import sys
import time
import json
import base64

from openai import OpenAI
import pyautogui

from PyQt5.QtCore import Qt, QBuffer, QIODevice
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication, QWidget

# ⚠️ Put your real API key here (better to load from a file or env var)
from dotenv import load_dotenv
load_dotenv()  # <-- THIS loads the .env file automatically


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Your prompt (safe to embed)
PROMPT_ID = "pmpt_6937d1bf42cc81969109e24af757a56e048d4233b9be9cff"
PROMPT_VERSION = "3"


class TransparentBorderWindow(QWidget):
    def __init__(self):
        super().__init__()

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
    # Capture screen and send to OpenAI's Responses
    # ---------------------------------------------
    def capture_and_send(self):
        app = QApplication.instance()
        screen = app.primaryScreen()  # ✅ fixed typo

        if not screen:
            print("No screen detected.")
            return

        pixmap = screen.grabWindow(0)
        print(
            f"[{time.strftime('%H:%M:%S')}] Captured: {pixmap.width()}x{pixmap.height()}"
        )

        # QPixmap -> PNG bytes in memory
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        pixmap.save(buffer, "PNG")
        image_bytes = buffer.data().data()

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        try:
            response = client.responses.create(
                model="gpt-4o",  # ✅ model is required; you can change to gpt-4.1
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
                                "text": "Analyze the image and respond ONLY as a JSON object. Include fields 'Question read', 'Answer selected', and 'Answer location'."
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
            print(response)
        except Exception as e:
            print("OpenAI error:", e)
            return

        raw_text = response.output_text
        print("Raw model output:", raw_text)

        try:
            data = json.loads(raw_text)
        except Exception as e:
            print("JSON Parse Error:", e)
            return

        print("Parsed JSON:", data)

        # Expecting "Answer location": [x, y]
        answer_loc = data.get("Answer location") or data.get("answer_location")

        if not answer_loc or len(answer_loc) != 2:
            print("No valid clicking coordinates found.")
            return

        x, y = answer_loc
        print(f"Clicking at: {x}, {y}")

        time.sleep(0.2)
        pyautogui.click(x, y)

    # ---------------------
    # Keyboard handling
    # ---------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            QApplication.quit()
        elif event.key() == Qt.Key_P:
            self.capture_and_send()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TransparentBorderWindow()
    sys.exit(app.exec_())
