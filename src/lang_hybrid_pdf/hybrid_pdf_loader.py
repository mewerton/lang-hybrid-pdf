#lang-hybrid-pdf/src/lang_hybrid_pdf/hybrid_pdf_loader.py
"""
hybrid_pdf_loader.py
--------------------
Loader híbrido: extrai texto nativo via Docling quando disponível
e aplica OCR seletivo (bloco-a-bloco) em páginas ou PDFs onde o
texto está “embutido” em imagens. Mantém deduplicação barata e
pode receber as páginas já classificadas pelo extractor_router.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import fitz                       # PyMuPDF
import pytesseract
from PIL import Image
from rapidfuzz import fuzz
from pdf2image import convert_from_path
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from lang_hybrid_pdf.settings import OCR

from .helpers import (
    bbox_sort_key,
    boxes_iou,
    fingerprint,
    adjust_chunks_to_token_limit,
)
from .text_docling import load_with_docling
from .image_layout_ocr import layout_ocr_from_pdf  # caso router peça fallback

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Parâmetros default (podem ser sobreescritos via __init__)
_MIN_NATIVE_CHARS = OCR.min_native_chars
_MIN_OCR_CHARS    = OCR.min_ocr_chars
_DPI_PAGE_IMAGE   = OCR.page_image_dpi
_DPI_BLOCK_IMAGE  = int(OCR.quick_ocr_dpi * 2.5)

# ----------------------------------------------------------------------
class HybridPDFLoader(BaseLoader):
    """
    Extrai texto nativo + aplica OCR seletivo em PDFs híbridos.
    Se `text_pages`/`ocr_pages` forem passados (via router), pula
    a classificação local e usa esses rótulos diretamente.
    """

    def __init__(
        self,
        file_path: str,
        *,
        # --- overrides de OCR ---
        ocr_lang: str = "por+eng",
        min_native_chars: int = _MIN_NATIVE_CHARS,
        min_ocr_chars: int = _MIN_OCR_CHARS,
        dpi_page_image: int = _DPI_PAGE_IMAGE,
        dpi_block_image: int = int(_DPI_BLOCK_IMAGE),
        # --- controle de chunk ---
        token_limit: int | None = None,
        # --- rótulos vindos de fora (opcional) ---
        text_pages: list[int] | None = None,
        ocr_pages:  list[int] | None = None,
    ):
        self.file_path = file_path

        self.ocr_lang          = ocr_lang
        self.MIN_NATIVE_CHARS  = min_native_chars
        self.MIN_OCR_CHARS     = min_ocr_chars
        self.DPI_PAGE_IMAGE    = dpi_page_image
        self.DPI_BLOCK_IMAGE   = dpi_block_image
        self.token_limit       = token_limit

        self._text_pages_in = text_pages
        self._ocr_pages_in  = ocr_pages

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------
    def _classify_pages(self) -> Tuple[str, list[int], list[int]]:
        """Rótula cada página como text / ocr e devolve o tipo global."""
        doc = fitz.open(self.file_path)
        text_pages, ocr_pages = [], []
        for pno, page in enumerate(doc, 1):
            if len(page.get_text().strip()) >= self.MIN_NATIVE_CHARS:
                text_pages.append(pno)
            else:
                ocr_pages.append(pno)

        kind = (
            "hybrid" if text_pages and ocr_pages
            else "text" if text_pages
            else "image"
        )
        logger.info("📑 %s | text=%s | ocr=%s", kind, text_pages, ocr_pages)
        return kind, text_pages, ocr_pages

    def _extract_blocks_hybrid(self, page: fitz.Page, pno: int) -> List[Document]:
        """Executa OCR seletivo bloco-a-bloco numa página híbrida."""
        collected: list[tuple[Tuple[float, ...], str]] = []
        docs: List[Document] = []

        blocks = sorted(
            page.get_text("dict")["blocks"],
            key=lambda b: bbox_sort_key(tuple(b["bbox"])),
        )

        for blk in blocks:
            bbox, btype = blk["bbox"], blk["type"]
            meta = {"page": pno, "bbox": bbox}

            # ---- A. texto nativo ---------------------------------
            if btype == 0:
                line = " ".join(
                    w["text"] for l in blk["lines"] for w in l["spans"]
                ).strip()
                if line:
                    collected.append((bbox, line))
                    docs.append(Document(line, metadata=meta))

            # ---- B. imagem → OCR --------------------------------
            elif btype == 1:
                # se a imagem cobre >=50 % de bloco com texto já capturado → pula
                if any(boxes_iou(bbox, tbx) >= 0.5 for tbx, _ in collected):
                    continue

                pix = page.get_pixmap(dpi=self.DPI_BLOCK_IMAGE,
                                      clip=fitz.Rect(*bbox))
                img = Image.frombytes("RGB", (pix.width, pix.height),
                                      pix.samples)
                ocr = pytesseract.image_to_string(img,
                                                  lang=self.ocr_lang).strip()

                if len(ocr) < self.MIN_OCR_CHARS:
                    continue
                if any(fuzz.ratio(ocr, txt) >= 90 for _, txt in collected):
                    continue  # muito parecido com algo já guardado

                collected.append((bbox, ocr))
                docs.append(Document(ocr, metadata=meta))

        return docs

    # ------------------------------------------------------------------
    # API pública (BaseLoader)
    # ------------------------------------------------------------------
    def load(self) -> List[Document]:
        # 0) Usa rótulos do router caso venham preenchidos
        if self._text_pages_in is not None and self._ocr_pages_in is not None:
            kind = "hybrid"
            text_pages, ocr_pages = self._text_pages_in, self._ocr_pages_in
        else:
            kind, text_pages, ocr_pages = self._classify_pages()

        all_docs: List[Document] = []

        # 1) Texto nativo ---------------------------------------------
        if text_pages:
            docs_text = load_with_docling(self.file_path)

            # --- CORREÇÃO 0-based → 1-based ---------------------------------
            def _page1(d):
                """Retorna página 1-based mesmo quando o loader devolve 0-based."""
                return (d.metadata.get("page") or 0) + 1

            docs_text = [d for d in docs_text if _page1(d) in text_pages]
            # ----------------------------------------------------------------

            all_docs.extend(docs_text)

        # 2) PDF 100 % imagem -----------------------------------------
        if kind == "image" and not text_pages:
            for i, img in enumerate(
                convert_from_path(self.file_path, dpi=self.DPI_PAGE_IMAGE), 1
            ):
                txt = pytesseract.image_to_string(img,
                                                  lang=self.ocr_lang).strip()
                if txt:
                    all_docs.append(Document(txt, metadata={"page": i}))

        # 3) Páginas híbridas -----------------------------------------
        if ocr_pages:
            with fitz.open(self.file_path) as doc:
                for pno in ocr_pages:
                    all_docs.extend(
                        self._extract_blocks_hybrid(
                            doc.load_page(pno - 1), pno
                        )
                    )

        # 4) Token-limit + deduplicação ------------------------------
        chunks = adjust_chunks_to_token_limit(all_docs, self.token_limit)
        seen, dedup = set(), []
        for d in chunks:
            fp = fingerprint(d.page_content)
            if fp not in seen:
                dedup.append(d)
                seen.add(fp)

        return dedup
