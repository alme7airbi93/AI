import sys
import time
import json
import threading
import io
import difflib
import os
import re

import keyboard  # global hotkeys
import pyautogui
import numpy as np
import cv2
import pytesseract
import requests
from PIL import Image

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
INITIAL_X = 200
INITIAL_Y = 200
INITIAL_WIDTH = 1500
INITIAL_HEIGHT = 800

MIN_MATCH_SCORE = 0.55

# ==========================
# RAG (CSV) CONFIG
# ==========================
QA_CSV_PATH = "My AI\\part1_kahoot_questions_answers.csv"  # <-- CHANGE THIS
RAG_MIN_SCORE = 0.38               # raise if you want stricter matching
RAG_TOP_K = 3

# ==========================
# LOCAL LLM API CONFIG
# ==========================
LOCAL_LLM_MODEL = "local-model"
LOCAL_LLM_URL = "http://localhost:1234/v1/chat/completions"

# --------------------------
# Small helpers
# --------------------------


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # remove junk OCR punctuation noise (keep ? for questions)
    s = re.sub(r"[^\w\s\?]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _best_option_from_answer(options, answer_text):
    """
    Given options shown on screen and an answer string (from CSV or LLM),
    pick the closest option text.
    """
    if not options:
        return answer_text

    ans_n = _normalize_text(answer_text)
    if not ans_n:
        return options[0]

    best = None
    best_score = -1.0
    for opt in options:
        opt_n = _normalize_text(opt)
        if not opt_n:
            continue

        # exact-ish containment gets priority
        score = difflib.SequenceMatcher(None, ans_n, opt_n).ratio()
        if ans_n == opt_n:
            score += 0.40
        elif ans_n in opt_n or opt_n in ans_n:
            score += 0.15

        if score > best_score:
            best_score = score
            best = opt

    return best if best is not None else options[0]

# --------------------------
# CSV RAG Index
# --------------------------


class CsvRagIndex:
    """
    Offline RAG for Q->A using TF-IDF char n-grams (robust to OCR errors).
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.vectorizer = None
        self.matrix = None
        self.questions_norm = []

        self.reload()

    def reload(self):
        if not os.path.exists(self.csv_path):
            print(f"[RAG] CSV not found: {self.csv_path}")
            self.df = None
            self.vectorizer = None
            self.matrix = None
            self.questions_norm = []
            return

        df = pd.read_csv(self.csv_path)

        # expected columns
        if "question" not in df.columns or "answer" not in df.columns:
            raise ValueError("[RAG] CSV must have columns: question, answer")

        df = df.dropna(subset=["question", "answer"]).copy()
        df["question"] = df["question"].astype(str)
        df["answer"] = df["answer"].astype(str)

        self.df = df
        self.questions_norm = [_normalize_text(
            q) for q in df["question"].tolist()]

        # Char ngrams are great for OCR noise
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1
        )
        self.matrix = self.vectorizer.fit_transform(self.questions_norm)

        print(f"[RAG] Loaded {len(self.df)} Q/A pairs from CSV.")

    def search(self, query: str, top_k: int = RAG_TOP_K):
        """
        Returns list of matches:
        [{"score": float, "question": str, "answer": str, "idx": int}, ...]
        """
        if self.df is None or self.vectorizer is None or self.matrix is None:
            return []

        qn = _normalize_text(query)
        if not qn:
            return []

        q_vec = self.vectorizer.transform([qn])
        sims = cosine_similarity(q_vec, self.matrix)[0]  # shape: (N,)

        # top_k indices
        top_idx = np.argsort(-sims)[:max(1, top_k)]
        results = []
        for i in top_idx:
            results.append({
                "score": float(sims[i]),
                "question": self.df.iloc[i]["question"],
                "answer": self.df.iloc[i]["answer"],
                "idx": int(i),
            })
        return results

# --------------------------
# LLM call (fallback only)
# --------------------------


def call_local_llm_fallback(question_text, options, all_lines):
    """
    Use your local OpenAI-compatible server only when RAG didn't confidently match.
    """
    has_options = len(options) > 0

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
- Return ONLY a JSON object and nothing else.
- JSON keys MUST be exactly:
  "Question read"
  "Answer selected"
""".strip()

        lines_str = "\n".join(f"{i+1}. {line}" for i,
                              line in enumerate(all_lines))
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
        "temperature": 0.0,
        "top_p": 0.1,
        "top_k": 20,
        "max_tokens": 64,
        "n": 1,
        "stream": False,
    }

    try:
        resp = requests.post(LOCAL_LLM_URL, json=payload, timeout=60)
    except Exception as e:
        print("Error calling local LLM server:", e)
        return {"Question read": question_text or "", "Answer selected": ""}

    if resp.status_code != 200:
        print("Local LLM HTTP error:", resp.status_code, resp.text[:300])
        return {"Question read": question_text or "", "Answer selected": ""}

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"].strip()
    except Exception:
        print("Unexpected LLM response format:", data)
        return {"Question read": question_text or "", "Answer selected": ""}

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        print("Could not find JSON in model output, raw text:")
        print(content)
        return {"Question read": question_text or "", "Answer selected": ""}

    json_str = content[start:end + 1]
    try:
        parsed = json.loads(json_str)
    except Exception as e:
        print("JSON parse error from local model:", e)
        print("JSON candidate:", json_str)
        return {"Question read": question_text or "", "Answer selected": ""}

    q = parsed.get("Question read") or parsed.get(
        "question") or question_text or ""
    a = parsed.get("Answer selected") or parsed.get("answer") or ""

    return {"Question read": q, "Answer selected": a}

# --------------------------
# Main decision function (RAG first)
# --------------------------


def answer_with_rag_then_llm(rag: CsvRagIndex, ocr_lines):
    """
    Returns dict:
      {"Question read": "...", "Answer selected": "...", "Source": "RAG/LLM", "RagScore": float}
    """
    clean_lines = [ln.strip() for ln in ocr_lines if ln.strip()]
    if not clean_lines:
        return {"Question read": "", "Answer selected": "", "Source": "NONE", "RagScore": 0.0}

    # Detect question line
    question_idx = None
    for i, ln in enumerate(clean_lines):
        lower = ln.lower()
        if "?" in ln or lower.startswith("true or false"):
            question_idx = i
            break
    if question_idx is None:
        question_idx = 0

    question_text = clean_lines[question_idx]
    options = [ln for i, ln in enumerate(clean_lines) if i != question_idx]

    # 1) RAG lookup
    rag_matches = rag.search(question_text, top_k=RAG_TOP_K) if rag else []
    if rag_matches:
        best = rag_matches[0]
        print(
            f"[RAG] Top match score={best['score']:.3f} Q={best['question']!r} A={best['answer']!r}")
        if best["score"] >= RAG_MIN_SCORE:
            # If options exist, map CSV answer to one of the visible options
            final_answer = _best_option_from_answer(options, best["answer"])
            return {
                "Question read": question_text,
                "Answer selected": final_answer,
                "Source": "RAG",
                "RagScore": float(best["score"]),
            }
        else:
            print(
                f"[RAG] Score below threshold ({RAG_MIN_SCORE}), fallback to LLM.")

    # 2) fallback to LLM
    llm_out = call_local_llm_fallback(question_text, options, clean_lines)
    # Safety: if options exist, force answer to be one of options
    if options:
        llm_out["Answer selected"] = _best_option_from_answer(
            options, llm_out.get("Answer selected", ""))
    return {
        "Question read": llm_out.get("Question read", question_text),
        "Answer selected": llm_out.get("Answer selected", ""),
        "Source": "LLM",
        "RagScore": 0.0,
    }

# =========================================================
# Your UI / OCR / Click code (mostly unchanged)
# =========================================================


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

        self.last_bbox_global = None

        # ---- Load RAG index once ----
        try:
            self.rag = CsvRagIndex(QA_CSV_PATH)
        except Exception as e:
            print("[RAG] Failed to load CSV index:", e)
            self.rag = None

        self.capture_requested.connect(self.capture_and_send)
        self.quit_requested.connect(QApplication.instance().quit)

        t = threading.Thread(target=self._setup_global_hotkeys, daemon=True)
        t.start()

    def _setup_global_hotkeys(self):
        keyboard.add_hotkey("p", lambda: self.capture_requested.emit())
        keyboard.add_hotkey("esc", lambda: self.quit_requested.emit())
        keyboard.wait()

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

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        pen = QPen(QColor(0, 255, 0, 255))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRect(rect)

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

    def show_qa_popup(self, question, answer, clicked_pos=None, ocr_failed=False, source=None, rag_score=None):
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)

        if ocr_failed:
            title = "Answer (no click)"
            extra = "\n\nNo matching answer text was found on the screen to click."
        else:
            title = "Answer + Click"
            extra = f"\n\nClicked at ({clicked_pos[0]}, {clicked_pos[1]})." if clicked_pos else ""

        meta = ""
        if source:
            meta += f"\n\nSource: {source}"
        if rag_score is not None:
            meta += f"\nRAG score: {rag_score:.3f}"

        box.setWindowTitle(title)
        box.setText(f"Question:\n{question}\n\nAnswer:\n{answer}{extra}{meta}")
        box.setStandardButtons(QMessageBox.Ok)

        app = QApplication.instance()
        screen = app.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            box.move(geo.x() + 20, geo.y() + 20)
        else:
            box.move(20, 20)

        box.exec_()

    def build_ocr_lines_with_words(self, ocr_data):
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
                line_map[key] = {"words": []}
                line_order.append(key)

            x = ocr_data["left"][i]
            y = ocr_data["top"][i]
            w = ocr_data["width"][i]
            h = ocr_data["height"][i]

            line_map[key]["words"].append(
                {"text": t, "x1": x, "y1": y, "x2": x + w, "y2": y + h}
            )

        lines = []
        for key in line_order:
            entry = line_map[key]
            if not entry["words"]:
                continue
            full_text = " ".join(w["text"] for w in entry["words"]).strip()
            if full_text:
                lines.append({"text": full_text, "words": entry["words"]})

        return lines

    def split_ocr_line_into_columns(self, words, min_gap_px=50):
        if not words:
            return []

        words_sorted = sorted(words, key=lambda w: w["x1"])
        widths = [w["x2"] - w["x1"] for w in words_sorted]
        gap_threshold = max(
            min_gap_px, (sum(widths) / len(widths)) * 1.6) if widths else min_gap_px

        segments = [[words_sorted[0]]]
        for w in words_sorted[1:]:
            prev = segments[-1][-1]
            gap = w["x1"] - prev["x2"]
            if gap > gap_threshold:
                segments.append([w])
            else:
                segments[-1].append(w)

        result_lines = []
        for seg in segments:
            txt = " ".join(w["text"] for w in seg).strip()
            if txt:
                result_lines.append(txt)
        return result_lines

    def ocr_lines_for_model(self, image_bytes):
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        ocr_data = pytesseract.image_to_data(
            thresh, output_type=pytesseract.Output.DICT)
        lines_struct = self.build_ocr_lines_with_words(ocr_data)

        logical_lines = []
        for line in lines_struct:
            logical_lines.extend(
                self.split_ocr_line_into_columns(line["words"]))

        logical_lines = [ln for ln in logical_lines if ln.strip()]

        print(f"OCR lines for model ({len(logical_lines)}):")
        for i, line in enumerate(logical_lines):
            print(f"  [{i}] {line!r}")
        return logical_lines

    def find_answer_bbox_with_ocr(self, image_bytes, answer_text, region_x, region_y, question_text=None):
        if not answer_text:
            return None

        answer_norm = answer_text.strip().lower()
        if not answer_norm:
            return None

        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        ocr_data = pytesseract.image_to_data(
            thresh, output_type=pytesseract.Output.DICT)
        lines = self.build_ocr_lines_with_words(ocr_data)

        # True/False special case (same logic you had)
        if answer_norm in ("true", "false"):
            occurrences = []
            for line_idx, line in enumerate(lines):
                line_lower = line["text"].lower()
                contains_true = "true" in line_lower
                contains_false = "false" in line_lower
                is_tf_question_line = ("true or false" in line_lower) or (
                    contains_true and contains_false)
                if is_tf_question_line:
                    continue
                for w in line["words"]:
                    t_norm = w["text"].strip().lower()
                    if not t_norm:
                        continue
                    sim = difflib.SequenceMatcher(
                        None, answer_norm, t_norm).ratio()
                    if t_norm == answer_norm or answer_norm in t_norm or t_norm in answer_norm or sim >= 0.8:
                        occurrences.append(w)

            if occurrences:
                w = occurrences[1] if len(occurrences) >= 2 else occurrences[0]
                x1_global = region_x + w["x1"]
                y1_global = region_y + w["y1"]
                x2_global = region_x + w["x2"]
                y2_global = region_y + w["y2"]
                return (x1_global, y1_global, x2_global, y2_global)

        # general case
        best_score = 0.0
        best_line_idx = None
        best_start = None
        best_end = None

        for line_idx, line in enumerate(lines):
            words = line["words"]
            n = len(words)
            for i in range(n):
                for j in range(i, n):
                    phrase_words = words[i:j+1]
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

        if best_line_idx is None or best_score < MIN_MATCH_SCORE:
            return None

        chosen_words = lines[best_line_idx]["words"][best_start:best_end+1]
        x1_local = min(w["x1"] for w in chosen_words)
        y1_local = min(w["y1"] for w in chosen_words)
        x2_local = max(w["x2"] for w in chosen_words)
        y2_local = max(w["y2"] for w in chosen_words)

        return (region_x + x1_local, region_y + y1_local, region_x + x2_local, region_y + y2_local)

    def capture_and_send(self):
        app = QApplication.instance()
        screen = app.primaryScreen()
        if not screen:
            return

        rect = self.geometry()
        region_x, region_y, region_w, region_h = rect.x(
        ), rect.y(), rect.width(), rect.height()
        print(
            f"[{time.strftime('%H:%M:%S')}] Capture region: {region_x},{region_y} {region_w}x{region_h}")

        self.hide()
        app.processEvents()
        pixmap = screen.grabWindow(0, region_x, region_y, region_w, region_h)
        self.show()
        app.processEvents()

        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        pixmap.save(buffer, "PNG")
        image_bytes = buffer.data().data()

        ocr_lines = self.ocr_lines_for_model(image_bytes)
        if not ocr_lines:
            return

        # ---- RAG-first answering ----
        result = answer_with_rag_then_llm(self.rag, ocr_lines)
        print("Answer result:", result)

        question = result.get("Question read", "") or ""
        answer = result.get("Answer selected", "") or ""
        source = result.get("Source", None)
        rag_score = result.get("RagScore", None)

        if not answer.strip():
            self.last_bbox_global = None
            self.update()
            self.show_qa_popup(question or "(no question)", "(no answer)",
                               ocr_failed=True, source=source, rag_score=rag_score)
            return

        bbox_global = self.find_answer_bbox_with_ocr(
            image_bytes, answer, region_x, region_y, question_text=question)
        if bbox_global is None:
            self.last_bbox_global = None
            self.update()
            self.show_qa_popup(question, answer, ocr_failed=True,
                               source=source, rag_score=rag_score)
            return

        self.last_bbox_global = bbox_global
        self.update()

        x1_g, y1_g, x2_g, y2_g = bbox_global
        cx = int((x1_g + x2_g) / 2)
        cy = int((y1_g + y2_g) / 2)

        time.sleep(0.2)
        pyautogui.click(cx, cy)

        self.show_qa_popup(question, answer, clicked_pos=(
            cx, cy), ocr_failed=False, source=source, rag_score=rag_score)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CaptureWindow()
    sys.exit(app.exec_())
