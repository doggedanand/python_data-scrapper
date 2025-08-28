import fitz  
import pdfplumber
import base64
import io
from PIL import Image
import re
import json
import pandas as pd
import os


def image_to_base64(image_bytes, ext):
    buffer = io.BytesIO(image_bytes)
    img = Image.open(buffer)
    output_buffer = io.BytesIO()
    img.save(output_buffer, format=ext.upper())
    return f"data:image/{ext};base64," + base64.b64encode(output_buffer.getvalue()).decode()

def extract_questions(pdf_file):
    doc = fitz.open(pdf_file)
    section_pattern = re.compile(r"^(Section|SECTION|Part|PART)\s*[:\-]?\s*(.*)", re.IGNORECASE)
    question_pattern = re.compile(r"^Q\.(\d+)")
    
    current_section = "Unknown Section"
    sectioned_questions = {current_section: []}

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        current_question = None
        for b in blocks:
            text = b[4].strip()
            if not text:
                continue

            # Detect section headers dynamically
            sec_match = section_pattern.match(text)
            if sec_match:
                section_title = sec_match.group(2).strip() if sec_match.group(2) else sec_match.group(1)
                current_section = section_title
                if current_section not in sectioned_questions:
                    sectioned_questions[current_section] = []
                continue

            # Detect question start
            q_match = question_pattern.match(text)
            if q_match:
                if current_question:
                    sectioned_questions[current_section].append(current_question)

                current_question = {
                    "question_number": int(q_match.group(1)),
                    "text": text,
                    "options": [],
                    "page": page_num,
                    "bbox": b[:4],
                    "images": [],
                    "tables": []
                }
            else:
                if current_question:
                    if re.match(r"^(Ans\s*)?\d+\.", text):
                        current_question["options"].append(text)
                    else:
                        current_question["text"] += " " + text

        if current_question:
            sectioned_questions[current_section].append(current_question)

    return sectioned_questions

def extract_images(pdf_file):
    doc = fitz.open(pdf_file)
    images = []

    for page_num, page in enumerate(doc, start=1):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            if not base_image:
                continue

            img_bytes = base_image["image"]
            ext = base_image["ext"]
            b64_img = image_to_base64(img_bytes, ext)
            rect = page.get_image_bbox(img)

            images.append({
                "page": page_num,
                "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
                "base64": b64_img
            })

    return images

def extract_tables(pdf_file):
    tables = []
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            extracted_tables = page.find_tables()
            for t in extracted_tables:
                df = pd.DataFrame(t.extract())
                tables.append({
                    "page": page_num,
                    "bbox": t.bbox,
                    "headers": list(df.iloc[0]) if not df.empty else [],
                    "rows": df.iloc[1:].values.tolist() if len(df) > 1 else []
                })
    return tables

def merge_data(sectioned_questions, images, tables):
    for section, qs in sectioned_questions.items():
        for q in qs:
            q_page, (x0, y0, x1, y1) = q["page"], q["bbox"]

            # Attach images
            for img in images:
                if img["page"] == q_page and img["bbox"][1] >= y0 and img["bbox"][1] <= y0 + 300:
                    q["images"].append(img["base64"])

            # Attach tables
            for tbl in tables:
                if tbl["page"] == q_page and tbl["bbox"][1] >= y0 and tbl["bbox"][1] <= y0 + 300:
                    q["tables"].append({
                        "headers": tbl["headers"],
                        "rows": tbl["rows"]
                    })

            # Clean
            del q["bbox"]
            del q["page"]

    # Format JSON with sections
    return {"sections": [{"section_name": s, "questions": qs} for s, qs in sectioned_questions.items()]}

def parse_pdf_to_json(pdf_file, output_file="output.json"):
    # Auto-delete old JSON
    if os.path.exists(output_file):
        os.remove(output_file)

    sectioned_questions = extract_questions(pdf_file)
    images = extract_images(pdf_file)
    tables = extract_tables(pdf_file)
    merged = merge_data(sectioned_questions, images, tables)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"JSON saved to {output_file}")


pdf_path = "SSC-CGL-Tier-1-Question-Paper-9-September-2024-Shift-1.pdf"
parse_pdf_to_json(pdf_path, "output.json")
