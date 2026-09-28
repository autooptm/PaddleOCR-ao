from paddleocr import PaddleOCR
ocr = PaddleOCR(use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False)
for img in __import__("sys").argv[1:] or ["general_ocr_002.png"]:
    result = ocr.ocr(img)
