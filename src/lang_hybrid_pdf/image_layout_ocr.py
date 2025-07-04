#lang-hybrid-pdf/src/lang_hybrid_pdf/image_layout_ocr.py 
"""
image_layout_ocr.py – OCR fallback para PDFs 100 % imagem.
• pipeline LEVE  = OCR → regex → SBERT (opcional extra [layout])
• pipeline PRECISO = gera PDF OCR + Docling (extra [docling])
"""

from __future__ import annotations
import os, re, io, tempfile, logging, gc
from typing import List
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from langchain_core.documents import Document
from .helpers import adjust_chunks_to_token_limit

logger   = logging.getLogger(__name__)
OCR_LANG = "por+eng"

# ------------- helpers para lazy import ----------------------------
def _require(pkg: str, extra: str):
    try:
        return __import__(pkg, fromlist=["*"])
    except ImportError as err:
        raise ImportError(
            f'Funcionalidade opcional ausente – instale com:\n'
            f'  pip install "lang-hybrid-pdf[{extra}]"'
        ) from err

# ------------- pipeline leve (LayoutLMv2 + SBERT) ------------------
def _load_layout_models():
    tr  = _require("transformers", "layout")
    st  = _require("sentence_transformers", "layout")
    processor = tr.LayoutLMv2Processor.from_pretrained(
        "microsoft/layoutlmv2-base-uncased", apply_ocr=False
    )
    semantic_model = st.SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    return processor, semantic_model

def _ocr_to_lines(img: Image.Image) -> List[tuple[str, int]]:
    data = pytesseract.image_to_data(
        img, output_type=pytesseract.Output.DICT, lang=OCR_LANG
    )
    lines: dict[int, list[str]] = {}
    for txt, ln in zip(data["text"], data["line_num"]):
        if txt.strip():
            lines.setdefault(ln, []).append(txt)
    return [(" ".join(ws), ln) for ln, ws in lines.items()]

# regex p/ cláusulas
_PAT = re.compile(
    r"(CLÁUSULA\s+\w+.*?|Art\.\s*\d+º?.*?|§\s*\d+º?.*?|Parágrafo\s+único.*?"
    r"|[IVXLCDM]+\s*-\s+.*?|[a-zA-Z]\))",
    flags=re.I,
)

def _split_juridico(txt: str) -> List[str]:
    m = list(_PAT.finditer(txt))
    if not m:
        return [txt]
    out = []
    for i, mm in enumerate(m):
        s = mm.start()
        e = m[i + 1].start() if i + 1 < len(m) else len(txt)
        seg = txt[s:e].strip()
        if len(seg) > 20:
            out.append(seg)
    return out

# agrupa chunks semânticos similares
def _group_similar(chunks: List[str], model) -> List[str]:
    if not chunks:
        return []
    import torch
    out, cur = [], chunks[0]
    cur_emb = model.encode(cur, convert_to_tensor=True)
    for nxt in chunks[1:]:
        nxt_emb = model.encode(nxt, convert_to_tensor=True)
        thr = 0.8 if len(cur) < 300 else 0.7
        if torch.cosine_similarity(cur_emb, nxt_emb, dim=0).item() >= thr:
            cur += "\n" + nxt
        else:
            out.append(cur)
            cur, cur_emb = nxt, nxt_emb
    out.append(cur)
    return out

def _layout_pipeline(pages, embedding_limit):
    processor, semantic_model = _load_layout_models()
    docs = []
    for i, img in enumerate(pages, 1):
        lines = _ocr_to_lines(img)
        for txt, ln in lines:
            for chunk in _split_juridico(txt):
                docs.append(Document(chunk, metadata={"page": i, "line": ln}))
    chunks = _group_similar([d.page_content for d in docs], semantic_model)
    docs_out = [Document(c, metadata={}) for c in chunks]
    return adjust_chunks_to_token_limit(docs_out, embedding_limit)

# ------------- pipeline Docling (gera PDF OCR) ----------------------
def _docling_pipeline(pages, embedding_limit):
    PdfWriter = _require("PyPDF2", "docling").PdfWriter
    PdfReader = _require("PyPDF2", "docling").PdfReader
    from .text_docling import load_with_docling

    writer = PdfWriter()
    for img in pages:
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(
            img, extension="pdf", lang=OCR_LANG, config="--psm 6"
        )
        writer.add_page(PdfReader(io.BytesIO(pdf_bytes)).pages[0])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        writer.write(tmp_pdf)
        tmp_path = tmp_pdf.name

    try:
        docs = load_with_docling(tmp_path)
    finally:
        writer.close()
        gc.collect()
        os.remove(tmp_path)

    return adjust_chunks_to_token_limit(docs, embedding_limit)

# ------------- API pública -----------------------------------------
def layout_ocr_from_pdf(file_path: str, embedding_limit: int | None = None) -> List[Document]:
    pages = convert_from_path(file_path, dpi=300)

    # 1) tenta pipeline preciso
    try:
        return _docling_pipeline(pages, embedding_limit)
    except ImportError:
        logger.info("Docling não instalado – tentando pipeline LayoutLMv2…")

    # 2) tenta LayoutLMv2 leve
    try:
        return _layout_pipeline(pages, embedding_limit)
    except ImportError:
        logger.info("transformers não instalado – fallback OCR simples.")

    # 3) OCR plano (mínimo)
    docs: List[Document] = []
    for i, pg in enumerate(pages, 1):
        text = pytesseract.image_to_string(pg, lang=OCR_LANG).strip()
        if text:
            docs.append(Document(text, metadata={"page": i}))
    return docs
