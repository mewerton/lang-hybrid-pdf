# lang-hybrid-pdf

> **Hybrid PDF loader** for LangChain that combines native‑text extraction with *selective* OCR, giving you the **best of both worlds** when dealing with real‑world PDFs that mix searchable text, scanned pages, stamps and signatures.

---

<div align="center">

| | |
|---|---|
| **Status** | ![Tests](https://img.shields.io/badge/tests-passing-brightgreen) |
| **License** | MIT |
| **Python** | 3.12+ |
| **Requires** | [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) 5 + `por` & `eng` language data |

</div>

---

## ✨ Why another PDF loader?

Most single‑shot PDF loaders force you to choose **between** native‑text extraction **or** full‑page OCR.  
Legal and corporate documents, however, are often *hybrids* – some pages have selectable text, others are scanned images.

`lang‑hybrid‑pdf` :

* **Detects** automatically whether each page already has text.
* **Runs OCR only where needed**, block‑by‑block, avoiding duplicate content.
* Keeps **layout‑aware metadata** (`page`, `bbox`) so you can trace answers back.
* Works out‑of‑the‑box with [LangChain](https://python.langchain.com).

---

## 📋 Prerequisites

Before installing, make sure you have **Tesseract OCR 5+** *and* the Portuguese + English language data:

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr tesseract-ocr-por tesseract-ocr-eng

# macOS (Homebrew)
brew install tesseract tesseract-lang
```

> Check with `tesseract --version`. If it fails, ensure the binary is in your `PATH`.

---

## 🏃 Quick start

```bash
# 1. Clone
git clone https://github.com/mewerton/lang-hybrid-pdf.git
cd lang-hybrid-pdf

# 2. Optional: create a virtual‑env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install **core** dependencies
pip install -e .

# 4. Run unit tests (≈ 3 min on CPU)
pytest -q
```

### Extras

| Extra            | Installs…                               | When to use                                                         |
|------------------|-----------------------------------------|---------------------------------------------------------------------|
| `docling`        | [`langchain-docling`](https://github.com/docling-ai/langchain-docling) + **PyPDF2** | High‑fidelity parsing of native‑text PDFs; keeps headings & chunks. |
| `layout`         | **transformers** + **sentence‑transformers** | Lightweight semantic grouping for 100 % image PDFs.                 |
| `full`           | *docling* + *layout*                    | All features – recommended for production.                          |

```bash
# Full installation
pip install -e .[full]
```

---

## 🔧 Usage inside your project

```python
from lang_hybrid_pdf import extract_text, HybridPDFLoader

# Simple one‑liner
docs = extract_text("mixed_document.pdf")
print(f"Extracted {len(docs)} chunks")

# Advanced usage with custom settings
loader = HybridPDFLoader(
    "contract.pdf",
    ocr_lang="por+eng",
    min_native_chars=60,
    token_limit=1_000,
)
docs = loader.load()

for doc in docs[:3]:  # first 3 chunks
    print(f"Page {doc.metadata['page']}: {doc.page_content[:100]}…")
```

Each item is a **LangChain `Document`** ready for chunking, embedding or RAG.

---

## 📂 Project layout

```
lang-hybrid-pdf/
├─ src/lang_hybrid_pdf/
│  ├─ extractor_router.py   # fast classifier (text / image / hybrid)
│  ├─ hybrid_pdf_loader.py  # native text + selective OCR
│  ├─ image_layout_ocr.py   # fallback for 100 % image PDFs
│  ├─ helpers.py            # IoU, fingerprint, token utils
│  ├─ settings.py           # central OCR config dataclass
│  └─ text_docling.py       # lazy import wrapper around Docling
└─ tests/
   ├─ data/                 # 3 tiny sample PDFs
   └─ test_loader.py        # end‑to‑end sanity tests
```

---

## 🔍 Troubleshooting

| Problem | Fix |
|---------|-----|
| **“Tesseract not found”** | Confirm `tesseract --version` works and the binary is in your `PATH`. |
| **Poor OCR quality / missing accents** | Install additional language packs and consider increasing `dpi_block_image`. |
| **Out‑of‑memory on huge PDFs** | Set `token_limit` in `HybridPDFLoader` or process the document in page batches. |

---

## 🧪 Testing

```bash
pytest           # run all tests
pytest -q        # quiet mode
pytest tests/test_loader.py::test_loader_direct_use
```

All four tests should pass:

| PDF              | Expected kind | Status |
|------------------|---------------|--------|
| `texto_total.pdf`| text          | ✅ |
| `ocr_total.pdf`  | image         | ✅ |
| `ocr_e_texto.pdf`| hybrid        | ✅ |

> On CPU‑only laptops, tests take ~3 minutes because Tesseract runs for each sample.

---

## 🚧 Limitations & roadmap

* Currently tuned for **Portuguese + English** (`por+eng`). Adjust `settings.OCR.lang` for others.
* Table reconstruction is **not** implemented (PRs welcome!).
* Multiprocessing for very large PDFs is under evaluation.

---

## 🤝 Contributing

1. Fork → commit → PR.  
2. Make sure `pytest -q` is green and `black .` yields no diff.  
3. For major ideas, open an issue so we can discuss design first.

Star ⭐ the repo if you find it useful!

---

## 📜 License

Distributed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

<div align="center"><sub>© 2025 Mewerton de Melo Silva — Happy Parsing!</sub></div>
