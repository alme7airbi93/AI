import sys
import time
import json
import threading
import io
import difflib
import os

import keyboard  # global hotkeys
import pyautogui
import numpy as np
import cv2
import pytesseract
import requests
from PIL import Image

from PyQt5.QtCore import Qt, QBuffer, QIODevice, pyqtSignal, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox

# ==========================
# TESSERACT CONFIG
# ==========================
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# ==========================
# APP CONFIG
# ==========================

# Initial window position and size
INITIAL_X = 200
INITIAL_Y = 200
INITIAL_WIDTH = 1500
INITIAL_HEIGHT = 800

# OCR matching similarity threshold (0..1)
MIN_MATCH_SCORE = 0.55

# ==========================
# LOCAL LLM API CONFIG
# ==========================
# For LM Studio:
# LOCAL_LLM_URL = "http://localhost:8080/v1/chat/completions"
# LM Studio usually ignores and uses the selected model
LOCAL_LLM_MODEL = "local-model"

# For Ollama instead, you would use:
LOCAL_LLM_URL = "http://localhost:1234/v1/chat/completions"
# LOCAL_LLM_MODEL = "qwen2.5:3b"  # or whatever model name you pulled


def call_local_model(ocr_lines):
    """
    ocr_lines: list of strings (each line from OCR).
    Returns a dict like:
      {"Question read": "...", "Answer selected": "..."}
    using a local OpenAI-compatible server (LM Studio / Ollama).

    Logic:
    - We detect the QUESTION and OPTIONS in Python.
    - If there ARE options, the model MUST choose exactly one of them.
    - If there are NO options, the model answers from its own knowledge.
    """
    # Clean lines
    clean_lines = [ln.strip() for ln in ocr_lines if ln.strip()]
    if not clean_lines:
        return {"Question read": "", "Answer selected": ""}

    # -------------------------------
    # 1) Detect question line
    # -------------------------------
    question_idx = None
    for i, ln in enumerate(clean_lines):
        lower = ln.lower()
        if "?" in ln or lower.startswith("true or false"):
            question_idx = i
            break

    if question_idx is None:
        # Fallback: first line as question
        question_idx = 0

    question_text = clean_lines[question_idx]
    # Everything else is potential options
    options = [ln for i, ln in enumerate(clean_lines) if i != question_idx]
    has_options = len(options) > 0

    # -------------------------------
    # 2) Build prompts
    # -------------------------------
    if has_options:
        options_str = "\n".join(f"- {opt}" for opt in options)
        system_prompt = """
You are an exam helper AI.

You are given:
- QUESTION: a single question string.
- OPTIONS: a list of possible answers.

Your job:
1. Read the QUESTION.
2. Use your world knowledge to decide which OPTION is factually correct.
3. Return JSON with:
   - "Question read": the QUESTION as you understood it.
   - "Answer selected": EXACTLY one of the given OPTIONS (copied verbatim).

Rules:
- "Answer selected" MUST be exactly equal to one of the OPTIONS.
- Do NOT invent new text when options are present.
- Never choose randomly; always pick the historically / factually correct option.
- Return ONLY a JSON object and nothing else.
- JSON keys MUST be exactly:
  "Question read"
  "Answer selected"
""".strip()

        user_prompt = f"""
QUESTION:
{question_text}

OPTIONS (copy exactly from here):
{options_str}

Return ONLY the JSON object as specified.
""".strip()
    else:
        # Question-only: no options on screen
        system_prompt = """
You are an exam helper AI.

You are given a QUESTION but no answer options.
Your job:
1. Understand the question.
2. Answer from your own knowledge.

Return JSON with:
- "Question read": the question.
- "Answer selected": a short, direct answer.

Rules:
- "Answer selected" MUST NOT be empty.
- It's OK if the answer does not appear in the OCR lines.
- Return ONLY a JSON object and nothing else.
- JSON keys MUST be exactly:
  "Question read"
  "Answer selected"
""".strip()

        # For question-only, we pass all lines for context
        lines_str = "\n".join(f"{i+1}. {line}" for i,
                              line in enumerate(clean_lines))
        user_prompt = f"""
OCR LINES:
{lines_str}

From these, extract the QUESTION and answer it from your knowledge.
Return ONLY the JSON object as specified.
""".strip()

    payload = {
        "model": LOCAL_LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],

        # 🔒 Deterministic / easy mode
        "temperature": 0.0,   # no randomness
        "top_p": 0.1,         # sample only from top 10% probability mass
        "top_k": 20,          # consider only top 20 tokens

        # ⚡ Make it faster by limiting output length
        "max_tokens": 64,     # you only need a short JSON, 64 is enough

        # 🧠 Simpler generation
        "n": 1,               # generate only 1 completion
        "stream": False,      # no streaming needed for your script
    }

    try:
        resp = requests.post(LOCAL_LLM_URL, json=payload, timeout=60)
    except Exception as e:
        print("Error calling local LLM server:", e)
        return {"Question read": "", "Answer selected": ""}

    if resp.status_code != 200:
        print("Local LLM HTTP error:", resp.status_code, resp.text[:300])
        return {"Question read": "", "Answer selected": ""}

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        print("Unexpected LLM response format:", data)
        return {"Question read": "", "Answer selected": ""}

    content = content.strip()
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        print("Could not find JSON in model output, raw text:")
        print(content)
        return {"Question read": "", "Answer selected": ""}

    json_str = content[start:end + 1]

    try:
        parsed = json.loads(json_str)
    except Exception as e:
        print("JSON parse error from local model:", e)
        print("JSON candidate:", json_str)
        return {"Question read": "", "Answer selected": ""}

    q = parsed.get("Question read") or parsed.get("question") or ""
    a = parsed.get("Answer selected") or parsed.get("answer") or ""

    return {
        "Question read": q,
        "Answer selected": a,
    }


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
        # Press 'p' anywhere to capture + model + OCR + click
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
            title = "Model Answer (no click)"
            extra = (
                "\n\nNo matching answer text was found on the screen to click. "
                "The answer may be written-only or the OCR could not detect it."
            )
        else:
            title = "Model Answer + Click"
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
    # NEW: Split one OCR line into multiple columns by big gaps
    # ---------------------------------------------
    def split_ocr_line_into_columns(self, words, min_gap_px=50):
        """
        Given a list of word dicts with x1/x2 coords, split them into
        multiple "columns" if there is a large horizontal gap between words.

        Returns a list of strings, e.g.
        ["Zayed Bin Mohamad", "Zayed Bin Sultan"]
        instead of one merged "Zayed Bin Mohamad Zayed Bin Sultan".
        """
        if not words:
            return []

        # Sort by left position
        words_sorted = sorted(words, key=lambda w: w["x1"])

        # Estimate a dynamic gap threshold based on average word width
        widths = [w["x2"] - w["x1"] for w in words_sorted]
        if widths:
            avg_w = sum(widths) / len(widths)
            gap_threshold = max(min_gap_px, avg_w * 1.6)
        else:
            gap_threshold = min_gap_px

        segments = [[words_sorted[0]]]

        for w in words_sorted[1:]:
            prev = segments[-1][-1]
            gap = w["x1"] - prev["x2"]

            if gap > gap_threshold:
                # Start a new segment (likely a new answer in another column)
                segments.append([w])
            else:
                segments[-1].append(w)

        # Turn each segment into a text line
        result_lines = []
        for seg in segments:
            txt = " ".join(w["text"] for w in seg).strip()
            if txt:
                result_lines.append(txt)

        return result_lines

    # ---------------------------------------------
    # Helper: OCR lines for the model (text only)
    # ---------------------------------------------
    def ocr_lines_for_model(self, image_bytes):
        """
        Run Tesseract on the region and return a list of text lines (strings)
        in reading order, for feeding into the local HF model.

        We also split a single physical line into 1..N logical lines
        if there are large horizontal gaps (e.g. two answers side-by-side).
        """
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        ocr_data = pytesseract.image_to_data(
            thresh, output_type=pytesseract.Output.DICT
        )

        # Build structured lines with words + coords
        lines_struct = self.build_ocr_lines_with_words(ocr_data)

        # Now split each physical line into 1..N logical lines
        logical_lines = []
        for line in lines_struct:
            words = line["words"]
            split_lines = self.split_ocr_line_into_columns(words)
            logical_lines.extend(split_lines)

        # Remove empties
        logical_lines = [ln for ln in logical_lines if ln.strip()]

        print(f"OCR lines for model ({len(logical_lines)}):")
        for i, line in enumerate(logical_lines):
            print(f"  [{i}] {line!r}")

        return logical_lines

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
        answer_text: string from model ("False", "Temp is 50", etc.)
        region_x, region_y: top-left of the region in global screen coords
        question_text: model's 'Question read', used for logging (optional)

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
        # GENERAL CASE: best matching sub-phrase
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
    # Capture region → OCR lines → Model → OCR match → click
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

        # -----------------------------
        # 1) OCR → get lines for model
        # -----------------------------
        ocr_lines = self.ocr_lines_for_model(image_bytes)
        if not ocr_lines:
            print("No OCR lines found for model, aborting.")
            return

        # -----------------------------
        # 2) Call local model via HTTP
        # -----------------------------
        data = call_local_model(ocr_lines)
        print("Local model JSON:", data)

        question = data.get("Question read", "") or ""
        answer = data.get("Answer selected", "") or ""

        print("Model Question:", question)
        print("Model Answer:", answer)

        if not answer.strip():
            print("Empty answer from model, aborting.")
            self.show_qa_popup(
                question or "(no question)",
                "(no answer)",
                clicked_pos=None,
                ocr_failed=True
            )
            return

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
