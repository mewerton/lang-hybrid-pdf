#lang-hybrid-pdf/src/lang_hybrid_pdf/helpers.py
"""
helpers.py – utilidades leves, sem dependências pesadas.
"""

from __future__ import annotations
import re, unicodedata, hashlib
from typing import Tuple, List
from langchain_core.documents import Document

# ----------------------------------------------------------------------
# Sort key para blocos (top-left → bottom-right)
def bbox_sort_key(bbox: Tuple[float, float, float, float]):
    x0, y0, *_ = bbox
    return (round(y0, 1), round(x0, 1))

# IoU simples entre dois bboxes
def boxes_iou(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter_h = max(0, min(ay1, by1) - max(ay0, by0))
    inter   = inter_w * inter_h
    area_a  = (ax1 - ax0) * (ay1 - ay0)
    area_b  = (bx1 - bx0) * (by1 - by0)
    return inter / max(area_a + area_b - inter, 1)

# “Fingerprint” de texto para deduplicação barata
def fingerprint(text: str, n: int = 120) -> str:
    txt = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    txt = re.sub(r"[\s\-_–—]+", "", txt.lower())
    txt = re.sub(r"[^a-z0-9]", "", txt)
    return hashlib.md5(txt[:n].encode()).hexdigest()

# Stub simples: mantém chunks se não houver token_limit
def adjust_chunks_to_token_limit(
    docs: List[Document], token_limit: int | None = None
) -> List[Document]:
    return docs if token_limit else docs
