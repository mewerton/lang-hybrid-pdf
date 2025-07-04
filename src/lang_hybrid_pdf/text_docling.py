#lang-hybrid-pdf/src/lang_hybrid_pdf/text_docling.py 

from __future__ import annotations
import logging
from typing import List, Any
from langchain_core.documents import Document
from .helpers import adjust_chunks_to_token_limit        # se quiser usar
# Se você tiver um decorator log_time comum ao pacote,
# faça from .helpers import log_time

logger = logging.getLogger(__name__)

def _require(pkg: str, extra: str):
    try:
        return __import__(pkg, fromlist=["*"])
    except ImportError as err:
        raise ImportError(
            f'Funcionalidade opcional ausente. Instale com:\n'
            f'  pip install "lang-hybrid-pdf[{extra}]"'
        ) from err

def load_with_docling(
    file_path: str,
    export_type: Any | None = None,
) -> List[Document]:
    """
    Tenta carregar via Docling; se não disponível, usa PyPDFLoader.
    """
    try:
        lc = _require("langchain_docling", "docling")
        ExportType = lc.loader.ExportType
        if export_type is None:
            export_type = ExportType.DOC_CHUNKS
        loader = lc.DoclingLoader(file_path=file_path, export_type=export_type)
        docs = loader.load()
        logger.info("✅ Docling retornou %d chunks.", len(docs))
        return docs

    except ImportError:
        from langchain_community.document_loaders import PyPDFLoader
        logger.warning("Docling não instalado – usando PyPDFLoader.")
        return PyPDFLoader(file_path).load()
