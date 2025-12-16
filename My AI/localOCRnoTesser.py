import sys
import time
import json
import threading
import base64
import io

import keyboard
import pyautogui
import requests
from PIL import Image

from PyQt5.QtCore import Qt, QBuffer, QIODevice, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox

# ==========================
# APP CONFIG
# ==========================
INITIAL_X = 200
INITIAL_Y = 200
INITIAL_WIDTH = 1500
INITIAL_HEIGHT = 800

# If model bbox is a bit off, add padding (in pixels) before clicking center
CLICK_PADDING_PX = 2

# ==========================
# LOCAL VISION LLM CONFIG
# ==========================
# OpenAI-compatible endpoint (llama.cpp server / LM Studio OpenAI server / etc.)
LOCAL_LLM_URL = "http://localhost:1234/v1/chat/completions"

# Put your actual model name if required by your server
# Examples: "qwen2.5-vl-3b-instruct", "qwen2.5-vl", or "local-model"
LOCAL_LLM_MODEL = "qwen/qwen2.5-vl-7b"

# Timeout for model call
MODEL_TIMEOUT_SEC = 90


def _img_bytes_to_data_url_png(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def call_local_vl_model(image_bytes: bytes) -> dict:
    """
    Sends the screenshot region to Qwen2.5-VL (via OpenAI-compatible chat/completions)
    and expects JSON output:
      {
        "Question read": "...",
        "Answer selected": "...",
        "Answer bbox": [x1,y1,x2,y2],
        "Bbox type": "normalized"   # coords are 0..1 relative to the provided image
      }

    If the model cannot find a click target, it should return:
      "Answer bbox": null
    """

    data_url = _img_bytes_to_data_url_png(image_bytes)

    system_prompt = """
You are an exam helper AI.

You will receive ONE screenshot image that contains:
- A question
- Answer options (usually multiple choice, or True/False)

Your job:
1) Read the question from the image.
2) Identify the correct answer option (factually correct).
3) Return JSON with:
   - "Question read": the question text you understood
   - "Answer selected": EXACT text of the chosen option as seen in the image (copy as close as possible)
   - "Answer bbox": [x1, y1, x2, y2] bounding box that tightly covers the chosen answer option (or at least its text)
   - "Bbox type": must be "normalized"

Rules for bbox:
- Use NORMALIZED coordinates in range 0..1 relative to the image.
- (0,0) is top-left of the image.
- (1,1) is bottom-right of the image.
- If you are not confident about the bbox, set "Answer bbox" to null.

Output rules:
- Return ONLY a JSON object (no markdown, no explanation).
- JSON keys MUST be exactly:
  "Question read"
  "Answer selected"
  "Answer bbox"
  "Bbox type"
""".strip()

    user_content = [
        {"type": "text", "text": "Analyze this screenshot and return the required JSON."},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]

    payload = {
        "model": LOCAL_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "stream": False,
    }

    try:
        resp = requests.post(LOCAL_LLM_URL, json=payload,
                             timeout=MODEL_TIMEOUT_SEC)
    except Exception as e:
        print("Error calling local VL model:", e)
        return {"Question read": "", "Answer selected": "", "Answer bbox": None, "Bbox type": "normalized"}

    if resp.status_code != 200:
        print("VL model HTTP error:", resp.status_code, resp.text[:500])
        return {"Question read": "", "Answer selected": "", "Answer bbox": None, "Bbox type": "normalized"}

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        print("Unexpected VL response format:", data)
        return {"Question read": "", "Answer selected": "", "Answer bbox": None, "Bbox type": "normalized"}

    content = (content or "").strip()

    # Extract JSON from any extra text (just in case)
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        print("Could not find JSON in model output, raw text:")
        print(content)
        return {"Question read": "", "Answer selected": "", "Answer bbox": None, "Bbox type": "normalized"}

    json_str = content[start:end + 1]

    try:
        parsed = json.loads(json_str)
    except Exception as e:
        print("JSON parse error:", e)
        print("JSON candidate:", json_str)
        return {"Question read": "", "Answer selected": "", "Answer bbox": None, "Bbox type": "normalized"}

    # Normalize keys (but keep your expected outputs)
    q = parsed.get("Question read", "") or ""
    a = parsed.get("Answer selected", "") or ""
    bbox = parsed.get("Answer bbox", None)
    bbox_type = parsed.get("Bbox type", "normalized") or "normalized"

    # Basic validation
    if bbox is not None:
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(isinstance(x, (int, float)) for x in bbox)
        ):
            bbox = None

    return {
        "Question read": q,
        "Answer selected": a,
        "Answer bbox": bbox,
        "Bbox type": bbox_type,
    }


class CaptureWindow(QWidget):
    capture_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.setGeometry(INITIAL_X, INITIAL_Y, INITIAL_WIDTH, INITIAL_HEIGHT)

        self.setFocusPolicy(Qt.StrongFocus)
        self.show()

        self._dragging = False
        self._drag_offset = QPoint(0, 0)

        # For debug draw of last bbox in global coords
        self.last_bbox_global = None  # (x1, y1, x2, y2)

        self.capture_requested.connect(self.capture_and_send)
        self.quit_requested.connect(QApplication.instance().quit)

        t = threading.Thread(target=self._setup_global_hotkeys, daemon=True)
        t.start()

    def _setup_global_hotkeys(self):
        keyboard.add_hotkey("p", lambda: self.capture_requested.emit())
        keyboard.add_hotkey("esc", lambda: self.quit_requested.emit())
        keyboard.wait()

    # Dragging
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

    # Painting
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

        # Red bbox
        if self.last_bbox_global is not None:
            x1_g, y1_g, x2_g, y2_g = self.last_bbox_global
            win_x, win_y = self.x(), self.y()
            x1 = x1_g - win_x
            y1 = y1_g - win_y
            x2 = x2_g - win_x
            y2 = y2_g - win_y

            pen_box = QPen(QColor(255, 0, 0, 255))
            pen_box.setWidth(2)
            painter.setPen(pen_box)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

    def show_qa_popup(self, question, answer, clicked_pos=None, no_click=False, reason=""):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)

        if no_click:
            title = "Model Answer (no click)"
            extra = f"\n\nNo click performed. {reason}".strip()
        else:
            title = "Model Answer + Click"
            extra = f"\n\nClicked at ({clicked_pos[0]}, {clicked_pos[1]})." if clicked_pos else ""

        box.setWindowTitle(title)
        box.setText(f"Question:\n{question}\n\nAnswer:\n{answer}{extra}")
        box.setStandardButtons(QMessageBox.Ok)

        app = QApplication.instance()
        screen = app.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            box.move(geo.x() + 20, geo.y() + 20)
        else:
            box.move(20, 20)

        box.exec_()

    def capture_and_send(self):
        app = QApplication.instance()
        screen = app.primaryScreen()
        if not screen:
            print("No screen detected.")
            return

        rect = self.geometry()
        region_x, region_y, region_w, region_h = rect.x(
        ), rect.y(), rect.width(), rect.height()

        print(
            f"[{time.strftime('%H:%M:%S')}] Capture region: {region_x},{region_y} {region_w}x{region_h}")

        # Hide border
        self.hide()
        app.processEvents()

        pixmap = screen.grabWindow(0, region_x, region_y, region_w, region_h)

        # Show again
        self.show()
        app.processEvents()

        # QPixmap -> PNG bytes
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        pixmap.save(buffer, "PNG")
        image_bytes = buffer.data().data()

        # Get image size for bbox conversion
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = pil_img.size

        # 1) Call VL model (NO OCR)
        result = call_local_vl_model(image_bytes)
        print("VL model JSON:", result)

        question = (result.get("Question read") or "").strip()
        answer = (result.get("Answer selected") or "").strip()
        bbox = result.get("Answer bbox", None)
        bbox_type = (result.get("Bbox type") or "normalized").strip().lower()

        if not answer:
            self.last_bbox_global = None
            self.update()
            self.show_qa_popup(
                question or "(no question)",
                "(no answer)",
                no_click=True,
                reason="Model returned empty answer."
            )
            return

        if bbox is None:
            self.last_bbox_global = None
            self.update()
            self.show_qa_popup(
                question or "(no question)",
                answer,
                no_click=True,
                reason="Model did not return a bbox (Answer bbox is null)."
            )
            return

        if bbox_type != "normalized":
            # You requested normalized only; if model violates it, refuse to click safely
            self.last_bbox_global = None
            self.update()
            self.show_qa_popup(
                question or "(no question)",
                answer,
                no_click=True,
                reason=f"Model returned unsupported bbox type: {bbox_type!r} (expected 'normalized')."
            )
            return

        # Convert normalized bbox -> local pixels
        x1n, yn1, x2n, yn2 = bbox
        x1n = _clamp(float(x1n), 0.0, 1.0)
        yn1 = _clamp(float(yn1), 0.0, 1.0)
        x2n = _clamp(float(x2n), 0.0, 1.0)
        yn2 = _clamp(float(yn2), 0.0, 1.0)

        # Ensure ordering
        x1n, x2n = min(x1n, x2n), max(x1n, x2n)
        yn1, yn2 = min(yn1, yn2), max(yn1, yn2)

        x1_local = int(x1n * img_w)
        y1_local = int(yn1 * img_h)
        x2_local = int(x2n * img_w)
        y2_local = int(yn2 * img_h)

        # Add tiny padding if desired
        x1_local = max(0, x1_local - CLICK_PADDING_PX)
        y1_local = max(0, y1_local - CLICK_PADDING_PX)
        x2_local = min(img_w - 1, x2_local + CLICK_PADDING_PX)
        y2_local = min(img_h - 1, y2_local + CLICK_PADDING_PX)

        # Convert to global coords
        x1_g = region_x + x1_local
        y1_g = region_y + y1_local
        x2_g = region_x + x2_local
        y2_g = region_y + y2_local

        self.last_bbox_global = (x1_g, y1_g, x2_g, y2_g)
        self.update()

        # Click center
        cx = int((x1_g + x2_g) / 2)
        cy = int((y1_g + y2_g) / 2)

        print(f"Clicking at: ({cx}, {cy})")
        time.sleep(0.2)
        pyautogui.click(cx, cy)

        self.show_qa_popup(question or "(no question)", answer,
                           clicked_pos=(cx, cy), no_click=False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CaptureWindow()
    sys.exit(app.exec_())
