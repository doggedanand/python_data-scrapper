import fitz
import base64
import io
from PIL import Image

PDF_PATH = "SSC-CGL-Tier-1-Question-Paper-9-September-2024-Shift-1.pdf"

def to_base64(img_bytes):
    return base64.b64encode(img_bytes).decode("utf-8")

doc = fitz.open(PDF_PATH)
with open("output.txt", "w", encoding="utf-8") as out:
    for page in doc:
        if page.number == 3:
            break
        page_items = []

        # --- text blocks ---
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = block
            if text.strip():
                page_items.append({
                    "type": "text",
                    "bbox": (x0, y0, x1, y1),
                    "data": text.strip()
                })

        # --- images ---
        for img in page.get_images(full=True):
            xref = img[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            rect = rects[0]
            base_img = doc.extract_image(xref)
            img_bytes = base_img["image"]
            img_b64 = to_base64(img_bytes)
            page_items.append({
                "type": "image",
                "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
                "data": img_b64
            })

        # --- sort top-to-bottom, left-to-right ---
        page_items.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

        # --- write output ---
        for item in page_items:
            if item["type"] == "text":
                out.write(item["data"] + "\n")
            elif item["type"] == "image":
                out.write("[IMAGE_BASE64]\n") 
                # out.write(item["data"] + "\n")

        out.write("\n" + "="*50 + "\n")   
        # break    
