"""Read an exam sheet from PDF, RTF or text, with OCR for pages that have no text layer.

Page by page: the text layer (pdftotext) is used when it holds real text; a page that is empty or
nearly empty (a scan) is rendered to an image and read by Tesseract with the Polish model. A mixed
PDF, part text and part scan, is handled the same way.

OCR needs, in order of preference:
    renderer  PyMuPDF (pip install pymupdf), or pdftoppm from poppler
    engine    Tesseract (tesseract.exe on PATH or in its default install folder)
    language  pol.traineddata, in Tesseract's tessdata or in data/tessdata/ of this repository
"""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import config
from .rtf import rtf_to_text

MIN_LETTERS = 25  # a scan's text layer is empty or holds a page number; a short task page has more
_TESSERACT_PATHS = [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"]
LOCAL_TESSDATA = config.ROOT / "data" / "tessdata"


def _letters(text):
    return len(re.findall(r"[^\W\d_]", text))


def text_layer_pages(path):
    exe = shutil.which("pdftotext")
    if not exe:
        return []
    out = subprocess.run([exe, "-enc", "UTF-8", str(path), "-"], capture_output=True)
    text = out.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")
    pages = text.split("\f")
    return pages[:-1] if pages and not pages[-1].strip() else pages


def page_count(path):
    try:
        import fitz  # PyMuPDF
        with fitz.open(str(path)) as doc:
            return doc.page_count
    except ImportError:
        pass
    exe = shutil.which("pdfinfo")
    if exe:
        out = subprocess.run([exe, str(path)], capture_output=True, text=True).stdout
        m = re.search(r"Pages:\s+(\d+)", out)
        if m:
            return int(m.group(1))
    return None


def tesseract():
    return shutil.which("tesseract") or next((p for p in _TESSERACT_PATHS if Path(p).exists()), None)


def _tessdata_env():
    env = dict(os.environ)
    if (LOCAL_TESSDATA / "pol.traineddata").exists():
        env["TESSDATA_PREFIX"] = str(LOCAL_TESSDATA)
    return env


def ocr_status():
    """What OCR can do on this machine, as a short report (also used by --check-ocr)."""
    renderer = "PyMuPDF" if _has_fitz() else ("pdftoppm" if shutil.which("pdftoppm") else None)
    engine = tesseract()
    langs = []
    if engine:
        out = subprocess.run([engine, "--list-langs"], capture_output=True, text=True, env=_tessdata_env())
        langs = [l.strip() for l in out.stdout.splitlines()[1:] if l.strip()]
    return {"renderer": renderer, "tesseract": engine, "polish": "pol" in langs, "langs": langs}


def _has_fitz():
    try:
        import fitz  # noqa: F401
        return True
    except ImportError:
        return False


def _render(path, page_no, out_dir, dpi=300):
    """Render one page (1-based) to PNG; returns the image path."""
    target = Path(out_dir) / f"page{page_no:03d}.png"
    if _has_fitz():
        import fitz
        with fitz.open(str(path)) as doc:
            doc[page_no - 1].get_pixmap(dpi=dpi).save(str(target))
        return target
    exe = shutil.which("pdftoppm")
    if not exe:
        raise RuntimeError("no PDF renderer: pip install pymupdf (or install poppler's pdftoppm)")
    prefix = Path(out_dir) / f"p{page_no:03d}"
    subprocess.run([exe, "-r", str(dpi), "-f", str(page_no), "-l", str(page_no), "-png", str(path), str(prefix)],
                   check=True, capture_output=True)
    return next(Path(out_dir).glob(f"p{page_no:03d}*.png"))


def ocr_page(path, page_no, out_dir):
    engine = tesseract()
    if not engine:
        raise RuntimeError("no OCR engine: install Tesseract (winget install UB-Mannheim.TesseractOCR)")
    image = _render(path, page_no, out_dir)
    lang = "pol" if ocr_status()["polish"] else "eng"
    out = subprocess.run([engine, str(image), "stdout", "-l", lang, "--psm", "3"],
                         capture_output=True, env=_tessdata_env())
    return out.stdout.decode("utf-8", errors="replace").replace("\r\n", "\n")


def read_pdf(path, force_ocr=False):
    """Return (text, report). Pages with a usable text layer are kept; the rest are OCR'd."""
    pages = [] if force_ocr else text_layer_pages(path)
    total = page_count(path) or len(pages)
    pages += [""] * max(0, total - len(pages))
    report = {"pages": total, "text_layer": 0, "ocr": 0, "failed": [], "ocr_error": None}
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, page in enumerate(pages, 1):
            if _letters(page) >= MIN_LETTERS:
                report["text_layer"] += 1
                out.append(page)
                continue
            try:
                text = ocr_page(path, i, tmp)
                report["ocr"] += 1
                out.append(text)
            except Exception as exc:  # keep going: one bad page must not lose the whole sheet
                report["failed"].append(i)
                report["ocr_error"] = str(exc)
                out.append(page)
    return "\n\f\n".join(out), report


def read_exam(path, force_ocr=False):
    """Any supported file -> (text, report)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path, force_ocr=force_ocr)
    if suffix == ".rtf":
        return rtf_to_text(path.read_text(encoding="latin-1")), {"source": "rtf"}
    return path.read_text(encoding="utf-8", errors="replace"), {"source": "text"}
