import sys, io, os, traceback, base64, json
from dataclasses import dataclass
from typing import Optional, List, Tuple

import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np
import pandas as pd

# NEW: pix2tex (LaTeX-OCR)
try:
    from pix2tex.cli import LatexOCR
except Exception:
    LatexOCR = None  # we'll error nicely at runtime

from PySide6.QtCore import Qt, Signal, QObject, QThread
from PySide6.QtWidgets import (
    QApplication, QWidget, QFileDialog, QLineEdit, QPushButton, QLabel,
    QSpinBox, QHBoxLayout, QVBoxLayout, QGridLayout, QCheckBox, QProgressBar,
    QPlainTextEdit, QMessageBox, QGroupBox, QFormLayout
)

# -------------------------
# Worker for OCR (QThread)
# -------------------------

@dataclass
class OCRConfig:
    pdf_path: str
    out_path: str
    start_page: int   # 1-indexed inclusive
    end_page: int     # 1-indexed inclusive
    dpi: int
    lang: str
    config: str
    tesseract_cmd: Optional[str] = None
    try_embedded_text_first: bool = True
    # Math options
    use_math_ocr: bool = False
    math_inline_threshold: float = 0.28  # symbol ratio for inline detection
    math_block_threshold: float = 0.40   # higher threshold -> block
    min_region_words: int = 2            # minimum clustered words to treat as a region


class OCRWorker(QObject):
    progress = Signal(int)  # 0..100
    log = Signal(str)
    done = Signal(str, list)  # (output_text_path, headings)
    error = Signal(str)

    def __init__(self, cfg: OCRConfig):
        super().__init__()
        self.cfg = cfg
        self._latex_model = None  # NEW: lazy-loaded pix2tex model

    # ---------- pix2tex helper (NEW) ----------
    def _ensure_pix2tex(self):
        if not self.cfg.use_math_ocr:
            return
        if LatexOCR is None:
            raise RuntimeError(
                "pix2tex is not installed. Run: pip install 'pix2tex[gui]' torch torchvision torchaudio"
            )
        if self._latex_model is None:
            self.log.emit("Loading pix2tex model (first time can take a bit)...")
            self._latex_model = LatexOCR()
            self.log.emit("pix2tex model loaded.")

    def _pix2tex(self, image_pil: Image.Image) -> Optional[str]:
        """
        Run LaTeX-OCR on the cropped image and return LaTeX string or None.
        """
        try:
            self._ensure_pix2tex()
            if self._latex_model is None:
                return None
            # pix2tex expects a PIL image; it returns a LaTeX string
            latex = self._latex_model(image_pil)
            if isinstance(latex, str) and latex.strip():
                return latex.strip()
        except Exception as e:
            self.log.emit(f"pix2tex error: {e}")
        return None

    @staticmethod
    def _tsv_to_df(tsv: str) -> pd.DataFrame:
        df = pd.read_csv(io.StringIO(tsv), sep="\t")
        df = df[df.conf != -1]  # drop header rows
        # Ensure numeric
        for c in ["left", "top", "width", "height", "conf", "level", "page_num", "block_num", "par_num", "line_num", "word_num"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        df["right"] = df["left"] + df["width"]
        df["bottom"] = df["top"] + df["height"]
        if "text" not in df.columns:
            df["text"] = ""
        return df

    @staticmethod
    def _symbol_ratio(s: str) -> float:
        if not s:
            return 0.0
        # Count characters that look like math operators/symbols
        math_chars = set("=+-*/^_()[]{}<>|∑∏∫∞≈≃≅≡≤≥≠′″‴·⋅∙×÷√%‰°αβγδΔεθλμνπρστυφχψωΩηξζΓΛΠΣΦΨΞ∂∇ℝℤℕℚℂ∘→←↔↦⇒⇔±∓∩∪∈∉∋∧∨⊂⊃⊆⊇⊕⊗⊥⋀⋁⊢⊨")
        sym = sum(1 for ch in s if (not ch.isalnum()) or ch in math_chars)
        return sym / max(1, len(s))

    @staticmethod
    def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        return inter / max(1.0, (a_area + b_area - inter))

    def _merge_boxes(self, boxes: List[Tuple[int,int,int,int]], iou_thresh=0.2) -> List[Tuple[int,int,int,int]]:
        """Greedy merge overlapping/close boxes."""
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        merged = []
        for b in boxes:
            merged_any = False
            for i, m in enumerate(merged):
                if self._iou(b, m) > iou_thresh:
                    x1 = min(b[0], m[0]); y1 = min(b[1], m[1])
                    x2 = max(b[2], m[2]); y2 = max(b[3], m[3])
                    merged[i] = (x1, y1, x2, y2)
                    merged_any = True
                    break
            if not merged_any:
                merged.append(b)
        return merged

    def _find_math_regions(self, tsv_df: pd.DataFrame) -> Tuple[List[Tuple[int,int,int,int]], List[Tuple[int,int,int,int]]]:
        """
        Returns (inline_regions, block_regions) as list of (x1,y1,x2,y2) in image coords.
        Heuristics:
          - compute symbol ratio per word/line; cluster high-ratio items.
        """
        if tsv_df.empty:
            return [], []

        # Build lines
        line_cols = ["page_num","block_num","par_num","line_num"]
        lines = []
        for keys, grp in tsv_df.groupby(line_cols):
            text = " ".join([str(t) for t in grp["text"].fillna("").tolist()]).strip()
            if not text:
                continue
            x1 = int(grp["left"].min()); y1 = int(grp["top"].min())
            x2 = int(grp["right"].max()); y2 = int(grp["bottom"].max())
            ratio = self._symbol_ratio(text)
            nwords = int((grp["text"].fillna("") != "").sum())
            lines.append({"bbox": (x1,y1,x2,y2), "text": text, "ratio": ratio, "n": nwords})

        inline_boxes, block_boxes = [], []
        for ln in lines:
            if ln["n"] < max(1, self.cfg.min_region_words):
                continue
            if ln["ratio"] >= self.cfg.math_block_threshold:
                block_boxes.append(ln["bbox"])
            elif ln["ratio"] >= self.cfg.math_inline_threshold:
                inline_boxes.append(ln["bbox"])

        inline_boxes = self._merge_boxes(inline_boxes, iou_thresh=0.15)
        block_boxes  = self._merge_boxes(block_boxes,  iou_thresh=0.15)
        return inline_boxes, block_boxes

    # ---------- Core OCR ----------

    def extract_text_or_ocr(self, page) -> Tuple[str, Optional[pd.DataFrame], Optional[Image.Image]]:
        if self.cfg.try_embedded_text_first:
            try:
                txt = page.get_text("text").strip()
                if txt:
                    # still return image for potential math crops
                    pix = page.get_pixmap(dpi=self.cfg.dpi)
                    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                    return txt, None, img
            except Exception:
                pass

        # OCR fallback with TSV
        pix = page.get_pixmap(dpi=self.cfg.dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        # Tesseract text (plain)
        text = pytesseract.image_to_string(
            img, lang=self.cfg.lang, config=self.cfg.config
        )

        # TSV for region detection
        try:
            tsv = pytesseract.image_to_data(
                img, lang=self.cfg.lang, config=self.cfg.config, output_type=pytesseract.Output.STRING
            )
            df = self._tsv_to_df(tsv)
        except Exception:
            df = None

        return text, df, img

    def _inject_math_latex(self, page_text: str, img: Image.Image, tsv_df: Optional[pd.DataFrame]) -> str:
        """Find math regions, OCR them with pix2tex, and replace approximate spans in text with LaTeX."""
        if not self.cfg.use_math_ocr or tsv_df is None:
            return page_text

        inline_boxes, block_boxes = self._find_math_regions(tsv_df)
        if not inline_boxes and not block_boxes:
            return page_text

        # Build a simple map: for each line, try to replace that whole line with LaTeX if tagged as mathy.
        def crop(b):
            x1,y1,x2,y2 = b
            pad = 4
            x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
            x2 = min(img.width, x2 + pad); y2 = min(img.height, y2 + pad)
            return img.crop((x1,y1,x2,y2))

        # Prepare per-line texts from TSV
        line_cols = ["page_num","block_num","par_num","line_num"]
        line_records = []
        for keys, grp in tsv_df.groupby(line_cols):
            text = " ".join([str(t) for t in grp["text"].fillna("").tolist()]).strip()
            if not text:
                continue
            x1 = int(grp["left"].min()); y1 = int(grp["top"].min())
            x2 = int(grp["right"].max()); y2 = int(grp["bottom"].max())
            bbox = (x1,y1,x2,y2)
            line_records.append({"bbox": bbox, "text": text})

        def nearest_line_text(bbox: Tuple[int,int,int,int]) -> Optional[str]:
            bx1,by1,bx2,by2 = bbox
            bc = ((bx1+bx2)/2, (by1+by2)/2)
            best = None; bestd = 1e18
            for rec in line_records:
                rx1,ry1,rx2,ry2 = rec["bbox"]
                rc = ((rx1+rx2)/2, (ry1+ry2)/2)
                d = (bc[0]-rc[0])**2 + (bc[1]-rc[1])**2
                if d < bestd:
                    bestd = d; best = rec["text"]
            return best

        replacements = []  # list of (original_text_line, latex_string, is_block)
        to_process = [(b, False) for b in inline_boxes] + [(b, True) for b in block_boxes]

        for bbox, is_block in to_process:
            crop_img = crop(bbox)
            latex = self._pix2tex(crop_img)
            if not latex:
                continue

            tgt_line = nearest_line_text(bbox)
            if not tgt_line:
                continue

            wrapped = f"\\[{latex}\\]" if is_block else f"\\({latex}\\)"
            replacements.append((tgt_line, wrapped, is_block))

        # Apply replacements: blocks first (longer), then inline
        used_lines = set()
        for orig, repl, is_block in sorted(replacements, key=lambda x: (not x[2], -len(x[0]))):
            if orig in used_lines:
                continue
            if orig in page_text:
                page_text = page_text.replace(orig, repl, 1)
                used_lines.add(orig)

        return page_text

    def run(self):
        try:
            if self.cfg.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.cfg.tesseract_cmd

            if not os.path.exists(self.cfg.pdf_path):
                raise FileNotFoundError(f"PDF not found: {self.cfg.pdf_path}")

            doc = fitz.open(self.cfg.pdf_path)
            start_i = max(0, self.cfg.start_page - 1)
            end_i = min(len(doc) - 1, self.cfg.end_page - 1)
            if start_i > end_i:
                raise ValueError("Start page must be <= end page and within document length.")

            total_pages = (end_i - start_i + 1)
            chunks: List[str] = []
            for idx, i in enumerate(range(start_i, end_i + 1), 1):
                self.log.emit(f"OCR: page {i+1}...")
                text, tsv_df, page_img = self.extract_text_or_ocr(doc[i])

                if self.cfg.use_math_ocr and page_img is not None and tsv_df is not None:
                    text = self._inject_math_latex(text, page_img, tsv_df)

                chunks.append(f"\n--- Page {i+1} ---\n{text}")
                self.progress.emit(int(idx * 100 / total_pages))

            all_text = "".join(chunks)
            with open(self.cfg.out_path, "w", encoding="utf-8") as f:
                f.write(all_text)

            # quick heading guess (unchanged)
            headings = []
            for line in all_text.splitlines():
                s = line.strip()
                if 1 <= len(s.split()) <= 10 and s and s[0].isalpha():
                    if s.isupper() or s.istitle():
                        headings.append(s)
            seen = set()
            headings = [h for h in headings if not (h in seen or seen.add(h))]

            self.done.emit(self.cfg.out_path, headings)

        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")


# -------------------------
# Main Window
# -------------------------

class OCRGui(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDF OCR (Persian + Math via pix2tex) — PySide6")
        self.setMinimumWidth(820)
        self.worker_thread: Optional[QThread] = None

        # widgets
        self.pdf_edit = QLineEdit()
        self.pdf_browse = QPushButton("Browse PDF")
        self.out_edit = QLineEdit()
        self.out_browse = QPushButton("Save As...")
        self.start_spin = QSpinBox()
        self.end_spin = QSpinBox()
        self.dpi_spin = QSpinBox()
        # Default to Persian + English
        self.lang_edit = QLineEdit("fas+eng")
        # Better defaults for layout text; keep editable
        self.cfg_edit = QLineEdit("--oem 1 --psm 6 -c preserve_interword_spaces=1")
        self.tess_edit = QLineEdit()
        self.try_embedded_cb = QCheckBox("Try embedded text first")
        self.try_embedded_cb.setChecked(True)

        # Math group (pix2tex)
        self.use_math_cb = QCheckBox("Use Math OCR (pix2tex)")
        self.inline_thr_edit = QLineEdit("0.28")
        self.block_thr_edit = QLineEdit("0.40")

        self.run_btn = QPushButton("Run OCR")
        self.progress = QProgressBar()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.headings_view = QPlainTextEdit()
        self.headings_view.setReadOnly(True)

        # ranges
        self.start_spin.setRange(1, 99999)
        self.end_spin.setRange(1, 99999)
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(220)

        # layout
        grid = QGridLayout()
        r = 0
        grid.addWidget(QLabel("PDF:"), r, 0)
        grid.addWidget(self.pdf_edit, r, 1)
        grid.addWidget(self.pdf_browse, r, 2); r += 1

        grid.addWidget(QLabel("Output .txt:"), r, 0)
        grid.addWidget(self.out_edit, r, 1)
        grid.addWidget(self.out_browse, r, 2); r += 1

        grid.addWidget(QLabel("Start page (1-based):"), r, 0)
        grid.addWidget(self.start_spin, r, 1); r += 1

        grid.addWidget(QLabel("End page (1-based):"), r, 0)
        grid.addWidget(self.end_spin, r, 1); r += 1

        grid.addWidget(QLabel("DPI:"), r, 0)
        grid.addWidget(self.dpi_spin, r, 1); r += 1

        grid.addWidget(QLabel("Language:"), r, 0)
        grid.addWidget(self.lang_edit, r, 1); r += 1

        grid.addWidget(QLabel("Tesseract config:"), r, 0)
        grid.addWidget(self.cfg_edit, r, 1); r += 1

        grid.addWidget(QLabel("Tesseract path (optional, Windows):"), r, 0)
        grid.addWidget(self.tess_edit, r, 1); r += 1

        grid.addWidget(self.try_embedded_cb, r, 1); r += 1

        # Math settings group
        math_box = QGroupBox("Math OCR")
        form = QFormLayout(math_box)
        form.addRow(self.use_math_cb)
        form.addRow(QLabel("Inline threshold (0–1):"), self.inline_thr_edit)
        form.addRow(QLabel("Block threshold (0–1):"), self.block_thr_edit)

        top = QVBoxLayout(self)
        top.addLayout(grid)
        top.addWidget(math_box)

        btn_bar = QHBoxLayout()
        btn_bar.addWidget(self.run_btn)
        btn_bar.addWidget(self.progress)
        top.addLayout(btn_bar)

        views = QHBoxLayout()
        left_col = QVBoxLayout()
        left_col.addWidget(QLabel("Log"))
        left_col.addWidget(self.log_view)
        right_col = QVBoxLayout()
        right_col.addWidget(QLabel("Possible Headings (rough)"))
        right_col.addWidget(self.headings_view)
        views.addLayout(left_col, 1)
        views.addLayout(right_col, 1)
        top.addLayout(views)

        # signals
        self.pdf_browse.clicked.connect(self.on_browse_pdf)
        self.out_browse.clicked.connect(self.on_browse_out)
        self.run_btn.clicked.connect(self.on_run_clicked)

    def on_browse_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select PDF", "", "PDF Files (*.pdf);;All Files (*)")
        if path:
            self.pdf_edit.setText(path)
            try:
                doc = fitz.open(path)
                self.start_spin.setMaximum(len(doc))
                self.end_spin.setMaximum(len(doc))
                self.end_spin.setValue(len(doc))
                self.start_spin.setValue(1)
            except Exception:
                pass

    def on_browse_out(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save OCR Text As", "", "Text Files (*.txt);;All Files (*)")
        if path:
            self.out_edit.setText(path)

    def on_run_clicked(self):
        pdf = self.pdf_edit.text().strip()
        outp = self.out_edit.text().strip()
        if not pdf:
            QMessageBox.warning(self, "Missing PDF", "Please choose a PDF file.")
            return
        if not outp:
            base = os.path.splitext(pdf)[0] + "_ocr.txt"
            self.out_edit.setText(base)
            outp = base

        start_p = self.start_spin.value()
        end_p = self.end_spin.value()
        if start_p > end_p:
            QMessageBox.warning(self, "Page range", "Start page must be <= end page.")
            return

        dpi = self.dpi_spin.value()
        lang = self.lang_edit.text().strip() or "fas+eng"
        cfg = self.cfg_edit.text().strip() or "--oem 1 --psm 6 -c preserve_interword_spaces=1"
        tess = self.tess_edit.text().strip() or None
        try_emb = self.try_embedded_cb.isChecked()

        # Math settings
        use_math = self.use_math_cb.isChecked()
        try:
            inline_thr = float(self.inline_thr_edit.text().strip())
            block_thr = float(self.block_thr_edit.text().strip())
        except Exception:
            inline_thr, block_thr = 0.28, 0.40

        self.log_view.clear()
        self.headings_view.clear()
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)

        cfg_obj = OCRConfig(
            pdf_path=pdf,
            out_path=outp,
            start_page=start_p,
            end_page=end_p,
            dpi=dpi,
            lang=lang,
            config=cfg,
            tesseract_cmd=tess,
            try_embedded_text_first=try_emb,
            use_math_ocr=use_math,
            math_inline_threshold=inline_thr,
            math_block_threshold=block_thr
        )

        # spin up worker
        self.worker_thread = QThread()
        self.worker = OCRWorker(cfg_obj)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.append_log)
        self.worker.done.connect(self.on_done)
        self.worker.error.connect(self.on_error)

        # cleanup
        self.worker.done.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.on_thread_finished)

        self.worker_thread.start()

    def append_log(self, msg: str):
        self.log_view.appendPlainText(msg)

    def on_done(self, out_path: str, headings: List[str]):
        self.append_log(f"Saved text to: {out_path}")
        if headings:
            self.headings_view.appendPlainText("\n".join(headings))
        else:
            self.headings_view.appendPlainText("(no headings detected with the rough heuristic)")

    def on_error(self, err: str):
        self.append_log("ERROR:\n" + err)
        QMessageBox.critical(self, "OCR Error", err)

    def on_thread_finished(self):
        self.run_btn.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    w = OCRGui()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
