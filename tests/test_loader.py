"""
Testes de sanidade para o HybridPDFLoader / extractor_router.

Requisitos:
• Os PDFs de exemplo devem estar em tests/data/
    ├─ texto_total.pdf   (100 % texto nativo)
    ├─ ocr_total.pdf     (100 % imagem)
    └─ ocr_e_texto.pdf   (texto + imagem)
"""
from pathlib import Path
import pytest

from lang_hybrid_pdf.extractor_router import extract_text, fast_classify
from lang_hybrid_pdf.hybrid_pdf_loader import HybridPDFLoader

DATA_DIR = Path(__file__).parent / "data"

# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "fname, expected_kind",
    [
        ("texto_total.pdf",   "text"),
        ("ocr_total.pdf",     "image"),
        ("ocr_e_texto.pdf",   "hybrid"),
    ],
)
def test_extract_text_end_to_end(fname, expected_kind):
    pdf_path = DATA_DIR / fname

    # 0) Verifica se a classificação rápida bate com o esperado
    kind, *_ = fast_classify(str(pdf_path))
    assert kind == expected_kind, f"{fname} deveria ser '{expected_kind}', mas retornou '{kind}'"

    # 1) Extrai chunks via router
    chunks = extract_text(str(pdf_path))
    assert len(chunks) > 0, f"{fname} não gerou chunks"
    assert all(c.page_content.strip() for c in chunks), "Chunk vazio detectado"

# ----------------------------------------------------------------------
def test_loader_direct_use():
    """Garante que HybridPDFLoader sozinho funciona (sem router)."""
    pdf_path = DATA_DIR / "ocr_e_texto.pdf"
    loader = HybridPDFLoader(str(pdf_path))
    docs = loader.load()
    assert len(docs) > 0, "Loader não retornou chunks"
    # Não deve haver duplicatas pelo fingerprint (prefixo de 120 chars)
    prefixes = [d.page_content[:120] for d in docs]
    assert len(prefixes) == len(set(prefixes)), "Deduplicação falhou"
