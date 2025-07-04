from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class OCRCfg:
    min_native_chars: int = 60          #OCR.min_native_chars
    min_ocr_chars:   int = 30           #OCR.min_ocr_chars
    batch_size:      int = 3            #OCR.batch_size
    quick_ocr_dpi:   int = 120          #OCR.quick_ocr_dpi
    lang:            str = "por+eng"    #OCR.lang
    page_image_dpi:  int = 300          #OCR.page_image_dpi

OCR = OCRCfg()          # uso: OCR.min_native_chars, etc.

#OCR.page_image_dpi
