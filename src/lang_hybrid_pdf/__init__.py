#lang-hybrid-pdf/src/lang_hybrid_pdf/__init__.py
"""
lang_hybrid_pdf
~~~~~~~~~~~~~~~
Pacote que fornece:

• HybridPDFLoader – carrega PDFs híbridos (texto + imagem)
• extract_text     – roteador que escolhe loader adequado

Instalação completa (Docling + LayoutLMv2):
    pip install "lang-hybrid-pdf[full]"
"""

from importlib.metadata import version, PackageNotFoundError

# ---------------------------------------------------------------
# Versão do pacote (lida do pyproject/dist-info)
try:
    __version__: str = version(__name__.replace('.', '-'))
except PackageNotFoundError:  # editable mode antes de build
    __version__ = "0.0.0.dev0"

# ---------------------------------------------------------------
# APIs públicas
from .hybrid_pdf_loader import HybridPDFLoader  # noqa: F401
from .extractor_router  import extract_text     # noqa: F401

__all__ = [
    "HybridPDFLoader",
    "extract_text",
    "__version__",
]
