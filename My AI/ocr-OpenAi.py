from dotenv import load_dotenv
import sys
import time
import json
import base64
import threading
import io
import difflib
import os

from openai import OpenAI
import keyboard  # global hotkeys
import pyautogui

import numpy as np
import cv2
import pytesseract
from PIL import Image

from PyQt5.QtCore import Qt, QBuffer, QIODevice, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox

# ==========================
# TESSERACT CONFIG
# ==========================
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# ==========================
# CONFIG
# ==========================

# 🔑 Use env var if possible (replace with your real key if needed)
load_dotenv()  # <-- THIS loads the .env file automatically
print(os.getenv("OPENAI_API_KEY"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# client = OpenAI(api_key=API_KEY)

PROMPT_ID = "pmpt_6937d1bf42cc81969109e24af757a56e048d4233b9be9cff"
PROMPT_VERSION = "7"  # use the version you’re actually using on the platform

# Initial window position and size
INITIAL_X = 200
INITIAL_Y = 200
INITIAL_WIDTH = 1500
INITIAL_HEIGHT = 800

# OCR matching similarity threshold (0..1)
MIN_MATCH_SCORE = 0.55


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

        # Set initial geometry
        self.setGeometry(INITIAL_X, INITIAL_Y, INITIAL_WIDTH, INITIAL_HEIGHT)

        self.setFocusPolicy(Qt.StrongFocus)
        self.show()

        # For dragging the window with the mouse
        self._dragging = False
        self._drag_offset = QPoint(0, 0)

        # For debug draw of OCR-selected area in global coords
        self.last_bbox_global = None  # (x1, y1, x2, y2)

        # Connect signals to slots
        self.capture_requested.connect(self.capture_and_send)
        self.quit_requested.connect(QApplication.instance().quit)

        # Start global hotkey listener in a background thread
        t = threading.Thread(target=self._setup_global_hotkeys, daemon=True)
        t.start()

    def _setup_global_hotkeys(self):
        """Set up global hotkeys that work even when this window does not have focus."""
        # Press 'p' anywhere to capture + GPT + OCR + click
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

    # -------- Visual: transparent area with green border + debug bbox --------
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

        # Draw red rectangle of the last matched OCR box (if any)
        if self.last_bbox_global is not None:
            x1_g, y1_g, x2_g, y2_g = self.last_bbox_global

            # Convert global coords to local (for drawing inside this window)
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

    # ---------------------------------------------
    # Popup helper: top-left Q&A
    # ---------------------------------------------
    def show_qa_popup(self, question, answer, clicked_pos=None, ocr_failed=False):
        """
        Small popup in top-left corner showing the question and answer.
        If ocr_failed=True, it explains that the answer text was not found on the screen.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)

        if ocr_failed:
            title = "GPT Answer (no click)"
            extra = (
                "\n\nNo matching answer text was found on the screen to click. "
                "The answer may be written-only or the OCR could not detect it."
            )
        else:
            title = "GPT Answer + Click"
            if clicked_pos:
                extra = f"\n\nClicked at ({clicked_pos[0]}, {clicked_pos[1]})."
            else:
                extra = ""

        box.setWindowTitle(title)
        box.setText(f"Question:\n{question}\n\nAnswer:\n{answer}{extra}")
        box.setStandardButtons(QMessageBox.Ok)

        # Move popup to top-left of primary screen
        app = QApplication.instance()
        screen = app.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            box.move(geo.x() + 20, geo.y() + 20)
        else:
            # Fallback: top-left of parent window
            box.move(20, 20)

        box.exec_()

    # ---------------------------------------------
    # Helper: build line+word structure from OCR
    # ---------------------------------------------
    def build_ocr_lines_with_words(self, ocr_data):
        """
        Groups Tesseract word-level data into line-level entries.
        Returns a list of dicts:
          {
            'text': "full line text",
            'words': [
               {'text': t, 'x1':..., 'y1':..., 'x2':..., 'y2':...},
               ...
            ]
          }
        in reading order.
        """
        n_boxes = len(ocr_data["text"])

        line_map = {}
        line_order = []

        for i in range(n_boxes):
            t_raw = ocr_data["text"][i]
            if not t_raw:
                continue
            t = t_raw.strip()
            if not t:
                continue

            block = ocr_data.get("block_num", [0])[i]
            par = ocr_data.get("par_num", [0])[i]
            line = ocr_data.get("line_num", [0])[i]

            key = (block, par, line)
            if key not in line_map:
                line_map[key] = {
                    "words": [],  # list of {'text', 'x1','y1','x2','y2'}
                }
                line_order.append(key)

            x = ocr_data["left"][i]
            y = ocr_data["top"][i]
            w = ocr_data["width"][i]
            h = ocr_data["height"][i]

            line_map[key]["words"].append(
                {
                    "text": t,
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                }
            )

        lines = []
        for key in line_order:
            entry = line_map[key]
            if not entry["words"]:
                continue
            full_text = " ".join(w["text"] for w in entry["words"]).strip()
            if not full_text:
                continue

            lines.append(
                {
                    "text": full_text,
                    "words": entry["words"],
                }
            )

        return lines

    # ---------------------------------------------
    # Helper: OCR + match answer text → bounding box
    # ---------------------------------------------
    def find_answer_bbox_with_ocr(
        self,
        image_bytes,
        answer_text,
        region_x,
        region_y,
        question_text=None,
    ):
        """
        image_bytes: PNG bytes of the cropped region (window rectangle)
        answer_text: string from GPT ("False", "Temp is 50", etc.)
        region_x, region_y: top-left of the region in global screen coords
        question_text: GPT's 'Question read', used for logging (optional)

        Returns (x1_global, y1_global, x2_global, y2_global) or None.
        """

        if not answer_text:
            print("No answer text provided, skipping OCR match.")
            return None

        # Normalize text
        answer_norm = answer_text.strip().lower()
        if not answer_norm:
            print("Empty normalized answer, skipping OCR.")
            return None

        question_norm = (question_text or "").strip().lower()

        # Convert bytes to a PIL image, then to OpenCV BGR
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Optional: basic preprocessing for better OCR
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # OCR with position data
        ocr_data = pytesseract.image_to_data(
            thresh, output_type=pytesseract.Output.DICT
        )

        lines = self.build_ocr_lines_with_words(ocr_data)
        print(f"OCR produced {len(lines)} lines with text.")
        for idx, l in enumerate(lines):
            print(f"  Line {idx}: {l['text']!r}")

        if not lines:
            print("No OCR lines found.")
            return None

        # ---------------------------------------------------------
        # SPECIAL CASE: True / False
        #   - Ignore *any* line that looks like the question:
        #       contains "true or false" OR contains both "true" and "false".
        #   - On the remaining lines, search for "true"/"false" at word level.
        #   - If 2+ occurrences, click the second; otherwise click the only one.
        # ---------------------------------------------------------
        if answer_norm in ("true", "false"):
            occurrences = []  # list of (line_idx, word_idx, word_obj)

            for line_idx, line in enumerate(lines):
                line_lower = line["text"].lower()

                # Skip if this line obviously looks like the True/False question
                contains_true = "true" in line_lower
                contains_false = "false" in line_lower
                is_tf_question_line = (
                    "true or false" in line_lower
                    or (contains_true and contains_false)
                )

                if is_tf_question_line:
                    print(
                        f"Skipping line {line_idx} (looks like True/False question): {line['text']!r}")
                    continue

                for word_idx, w in enumerate(line["words"]):
                    t_norm = w["text"].strip().lower()
                    if not t_norm:
                        continue

                    # Fuzzy match for "true" / "false"
                    sim = difflib.SequenceMatcher(
                        None, answer_norm, t_norm).ratio()
                    if (
                        t_norm == answer_norm
                        or answer_norm in t_norm
                        or t_norm in answer_norm
                        or sim >= 0.8
                    ):
                        occurrences.append((line_idx, word_idx, w))

            print(
                f"True/False search (skipping any 'true or false' lines): "
                f"found {len(occurrences)} word occurrences of '{answer_norm}'."
            )

            if occurrences:
                # Prefer second occurrence if exists, otherwise first
                idx = 1 if len(occurrences) >= 2 else 0
                line_i, word_i, w = occurrences[idx]

                x1_local = w["x1"]
                y1_local = w["y1"]
                x2_local = w["x2"]
                y2_local = w["y2"]

                x1_global = region_x + x1_local
                y1_global = region_y + y1_local
                x2_global = region_x + x2_local
                y2_global = region_y + y2_local

                print(
                    f"[True/False special] chosen word '{w['text']}' "
                    f"on line {line_i}, global bbox ({x1_global},{y1_global})-({x2_global},{y2_global})"
                )

                return (x1_global, y1_global, x2_global, y2_global)
            else:
                print(
                    "No True/False occurrences on non-question lines; falling back to general phrase search.")

        # ---------------------------------------------------------
        # GENERAL CASE: choose best matching *sub-phrase* (contiguous
        # words in a line) so we don't span multiple answers.
        # (Used for normal answers, and as fallback for True/False)
        # ---------------------------------------------------------
        best_score = 0.0
        best_line_idx = None
        best_start = None
        best_end = None

        for line_idx, line in enumerate(lines):
            words = line["words"]
            n = len(words)
            if n == 0:
                continue

            # Try all contiguous windows of words in this line
            for i in range(n):
                for j in range(i, n):
                    phrase_words = words[i: j + 1]
                    phrase_text = " ".join(w["text"]
                                           for w in phrase_words).strip()
                    if not phrase_text:
                        continue

                    phrase_norm = phrase_text.lower()
                    score = difflib.SequenceMatcher(
                        None, answer_norm, phrase_norm).ratio()

                    # Boost if one contains the other
                    if answer_norm in phrase_norm or phrase_norm in answer_norm:
                        score += 0.1

                    if score > best_score:
                        best_score = score
                        best_line_idx = line_idx
                        best_start = i
                        best_end = j

        print(f"Best phrase similarity score: {best_score:.3f}")
        if best_line_idx is None or best_score < MIN_MATCH_SCORE:
            print("No good phrase match found above threshold.")
            return None

        chosen_words = lines[best_line_idx]["words"][best_start: best_end + 1]

        x1_local = min(w["x1"] for w in chosen_words)
        y1_local = min(w["y1"] for w in chosen_words)
        x2_local = max(w["x2"] for w in chosen_words)
        y2_local = max(w["y2"] for w in chosen_words)

        x1_global = region_x + x1_local
        y1_global = region_y + y1_local
        x2_global = region_x + x2_local
        y2_global = region_y + y2_local

        phrase_text = " ".join(w["text"] for w in chosen_words)
        print(f"Chosen phrase: {phrase_text!r}")
        print(
            f"OCR phrase bbox global: ({x1_global},{y1_global})-({x2_global},{y2_global})"
        )

        return (x1_global, y1_global, x2_global, y2_global)

    # ---------------------------------------------
    # Capture region → GPT → OCR match → click
    # ---------------------------------------------
    def capture_and_send(self):
        app = QApplication.instance()
        screen = app.primaryScreen()

        if not screen:
            print("No screen detected.")
            return

        # Get window geometry in screen coordinates
        rect = self.geometry()
        region_x, region_y, region_w, region_h = (
            rect.x(),
            rect.y(),
            rect.width(),
            rect.height(),
        )
        print(
            f"[{time.strftime('%H:%M:%S')}] Capture region: "
            f"{region_x},{region_y} {region_w}x{region_h}"
        )

        # Temporarily hide the window so the border doesn't appear in capture
        self.hide()
        app.processEvents()

        pixmap = screen.grabWindow(0, region_x, region_y, region_w, region_h)

        # Show window again
        self.show()
        app.processEvents()

        # Convert QPixmap -> PNG bytes in memory
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        pixmap.save(buffer, "PNG")
        image_bytes = buffer.data().data()

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Ask GPT for question + answer text
        # NOTE: Updated instruction:
        # - If there are visible answer choices on the screen: choose ONE full choice
        #   and return exactly that text as it appears (the whole line/sentence).
        # - If the question is a True/False question, respond with exactly "True" or "False".
        # - If there are NO visible answers: write the correct short answer text yourself.
        instruction = (
            f"You are looking only at the contents of this cropped region. "
            f"The image resolution is {region_w}x{region_h} pixels. "
            f"Read the question carefully.\n\n"
            f"1. If there are visible answer choices (for example multiple-choice), "
            f"determine the correct answer and respond using exactly the full text of that one answer choice "
            f"as it appears on the screen (the whole line/sentence, not just a single word).\n\n"
            f"2. If the question is a True/False question, respond with exactly one word: \"True\" or \"False\".\n\n"
            f"3. If there are no visible answer choices (an open question), "
            f"write the correct short answer text yourself.\n\n"
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

        print("GPT Question:", question)
        print("GPT Answer:", answer)

        # ---- OCR + match answer → bbox ----
        bbox_global = self.find_answer_bbox_with_ocr(
            image_bytes, answer, region_x, region_y, question_text=question
        )

        if bbox_global is None:
            # No OCR match for the answer text on the screen
            self.last_bbox_global = None
            self.update()
            self.show_qa_popup(
                question,
                answer,
                clicked_pos=None,
                ocr_failed=True
            )
            return

        x1_g, y1_g, x2_g, y2_g = bbox_global
        self.last_bbox_global = bbox_global
        self.update()

        # Click center of the matched box
        cx = int((x1_g + x2_g) / 2)
        cy = int((y1_g + y2_g) / 2)

        print(f"Clicking at: ({cx}, {cy})")
        time.sleep(0.2)
        pyautogui.click(cx, cy)

        # Show popup in top-left corner with Q & A
        self.show_qa_popup(question, answer, clicked_pos=(
            cx, cy), ocr_failed=False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CaptureWindow()
    sys.exit(app.exec_())
