"""
extractor_router.py – roteador que decide a estratégia de extração.

• text-only   → Docling / PyPDFLoader
• image-only  → OCR completo (layout_ocr_from_pdf)
• híbrido     → HybridPDFLoader  (texto nativo + OCR seletivo)

A lógica de classificação replica a que funciona no seu RAG Jurídico,
mas lendo todos os parâmetros a partir de lang_hybrid_pdf.settings.OCR.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import fitz                       # PyMuPDF
import pytesseract
from PIL import Image

from lang_hybrid_pdf.settings import OCR
from .text_docling import load_with_docling
from .image_layout_ocr import layout_ocr_from_pdf
from .hybrid_pdf_loader import HybridPDFLoader

logger = logging.getLogger(__name__)

CENTRAL_CROP_PCT = 0.25           # 25 % de cada borda → crop central


# ────────────────────────────────────────────────────────────────────
def _quick_ocr(page: fitz.Page, *, central_crop: bool = False) -> int:
    """OCR rápido (baixa DPI) só para contagem de caracteres."""
    if central_crop:
        w, h = page.rect.width, page.rect.height
        mx, my = w * CENTRAL_CROP_PCT, h * CENTRAL_CROP_PCT
        clip = fitz.Rect(mx, my, w - mx, h - my)
        pix  = page.get_pixmap(dpi=OCR.quick_ocr_dpi, clip=clip)
    else:
        pix  = page.get_pixmap(dpi=OCR.quick_ocr_dpi)

    img  = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return len(pytesseract.image_to_string(img, lang=OCR.lang).strip())


# ────────────────────────────────────────────────────────────────────
def fast_classify(file_path: str) -> Tuple[str, List[int], List[int]]:
    """
    Retorna:
        kind        : 'text' | 'image' | 'hybrid'
        text_pages  : páginas 1-based com texto nativo “longo”
        ocr_pages   : páginas onde o OCR rápido detectou texto relevante
    """
    doc = fitz.open(file_path)
    try:
        text_pages: list[int] = []
        ocr_pages:  list[int] = []

        # 1) Varredura página-a-página -------------------------------------
        for idx, page in enumerate(doc, 1):
            has_text = len(page.get_text().strip()) >= OCR.min_native_chars
            has_ocr  = _quick_ocr(page)             >= OCR.min_ocr_chars

            if has_text:
                text_pages.append(idx)
            if has_ocr:
                ocr_pages.append(idx)               # ⚠️ mesmo que haja texto!

            # Conclusão antecipada a cada *batch_size*
            if idx % OCR.batch_size == 0 and text_pages and ocr_pages:
                kind = "hybrid"
                break
        else:
            # Se TODAS as páginas têm texto nativo, é 'text'
            if len(text_pages) == doc.page_count:
                kind, ocr_pages = "text", []
            else:
                kind = ("hybrid" if text_pages and ocr_pages
                        else "text" if text_pages else "image")

        # 2) Sanity-check (remove falso híbrido por cabeçalhos vetoriais)
        if kind == "hybrid" and ocr_pages:
            first_img = ocr_pages[0]
            page      = doc.load_page(first_img - 1)
            if _quick_ocr(page, central_crop=True) < OCR.min_ocr_chars:
                kind, ocr_pages = "text", []

        return kind, text_pages, ocr_pages

    finally:
        doc.close()


# ────────────────────────────────────────────────────────────────────
def extract_text(file_path: str):
    """
    Entry-point público – delega ao loader adequado.
    Falhas inesperadas são capturadas para não quebrar aplicações.
    """
    try:
        kind, text_pages, ocr_pages = fast_classify(file_path)

        if kind == "text":
            return load_with_docling(file_path)

        if kind == "image":
            return layout_ocr_from_pdf(file_path)

        # kind == "hybrid"
        loader = HybridPDFLoader(
            file_path,
            text_pages=text_pages,
            ocr_pages=ocr_pages,
        )
        return loader.load()

    except Exception as exc:                # pragma: no cover
        logger.exception("Erro ao processar %s: %s", file_path, exc)
        return []                           # fallback seguro
