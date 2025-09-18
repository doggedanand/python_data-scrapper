# PDF processing
from pydoc import text
import fitz
import pdfplumber
# Image encoding
import base64
# JSON handling
import json
# Regex patterns
import re
# Type hints
from typing import Any, Dict, List, Tuple, Optional
import os
# Input PDF file path
PDF_PATH = "SSC-CGL-Tier-1-Question-Paper-9-September-2024-Shift-1.pdf"
# Output JSON file path
OUTPUT_JSON = "output.json"
# Max width/height for small icons
ICON_MAX_WH = 35
# Pattern to detect question numbers (e.g., Q. 1)
Q_START = re.compile(r"^\s*Q\.\s*(\d+)", re.IGNORECASE)
# Pattern to detect full section titles
SECTION_FULL = re.compile(r"^\s*Section\s*[:\.]\s*(.+)$", re.IGNORECASE)
# Pattern to detect section prefix without title
SECTION_PREFIX = re.compile(r"^\s*Section\s*[:\.]?\s*$", re.IGNORECASE)
# Pattern to detect answer lines
ANS = re.compile(r"^\s*Ans[\s\.:]*([1-9])?(.*)$", re.IGNORECASE)
# Pattern to detect option markers (1., A), etc.)
OPT_MARK = re.compile(r"^\s*([1-4a-dA-D])[\.\)]\s*", re.IGNORECASE)
# Patterns to identify and skip headers/footers
HEADER_FOOTER_PATTERNS = [
    re.compile(r"SSC", re.IGNORECASE),
    re.compile(r"Tier", re.IGNORECASE),
    re.compile(r"Shift", re.IGNORECASE),
    re.compile(r"Page \d+", re.IGNORECASE),
]
# Check if text matches header/footer patterns
def is_header_footer(text: str) -> bool:
    # Return True if any pattern matches
    return any(p.search(text) for p in HEADER_FOOTER_PATTERNS)
# Encode image bytes to base64 string
def to_base64(img_bytes: bytes) -> str:
    # Convert bytes to base64 and decode to string
    return base64.b64encode(img_bytes).decode("utf-8")
# Convert color value to RGB tuple
def parse_color(col: Any) -> Tuple[int, int, int]:
    # If color is integer, extract RGB from bits
    if isinstance(col, int):
        return ((col >> 16) & 255, (col >> 8) & 255, col & 255)
    # If color is list/tuple, handle float (0-1) or int values
    if isinstance(col, (list, tuple)):
        vals = list(col)
        # Convert float (0-1) to 0-255 range
        if all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in vals[:3]):
            return tuple(int(v * 255) for v in vals[:3])
        # Otherwise take first 3 values as integers
        return tuple(int(vals[i]) for i in range(3))
    # If color is single float, map to gray RGB
    if isinstance(col, float):
        v = int(col * 255)
        return (v, v, v)
    # Default black if unknown type
    return (0, 0, 0)
# Check if RGB color is red
def is_red(rgb: Tuple[int, int, int]) -> bool:
    # Unpack RGB values
    r, g, b = rgb
    # Red must be dominant and >150
    return (r > 150) and (r >= g + 30) and (r >= b + 30)
# Check if RGB color is green
def is_green(rgb: Tuple[int, int, int]) -> bool:
    # Unpack RGB values
    r, g, b = rgb
    # Green must be dominant and >150
    return (g > 150) and (g >= r + 30) and (g >= b + 30)
# Collect text and images from a PDF page
def collect_page_items(page) -> List[Dict[str, Any]]:
    # Store collected items
    items: List[Dict[str, Any]] = []
    # Extract page content as dict
    d = page.get_text("dict")
    # Loop through all blocks
    for block in d.get("blocks", []):
        # Skip non-text blocks
        if block.get("type", 0) != 0:
            continue
        # Loop through lines in block
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text is None:
                    continue
                text_stripped = text.strip()
                if not text_stripped:
                    continue
                bbox = span.get("bbox", (0, 0, 0, 0))
                color_raw = span.get("color", None)
                color = parse_color(color_raw) if color_raw is not None else (0, 0, 0)
                items.append({
                    "type": "text",
                    "text": text_stripped,
                    "bbox": bbox,
                    "color": color
                })
    # Loop through all images on page
    for img in page.get_images(full=True):
        xref = img[0]
        # Extract base image
        info = page.parent.extract_image(xref)
        img_bytes = info["image"]
        img_ext = info["ext"]
        smask = img[1] if len(img) > 1 else None
        if smask:
            try:
                smask_info = page.parent.extract_image(smask)
                if smask_info:
                    from PIL import Image
                    import io
                    base_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    mask_img = Image.open(io.BytesIO(smask_info["image"])).convert("L")
                    base_img.putalpha(mask_img)
                    buf = io.BytesIO()
                    base_img.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                    img_ext = "png"
            except Exception as e:
                print(f"Error applying smask: {e}")
        # Get width and height
        w, h = info.get("width", 0), info.get("height", 0)
        page_w = page.rect.width
        page_h = page.rect.height
        # Skip very large images (possible watermark)
        if w > page_w * 0.7 and h > page_h * 0.7:
            continue
        # Get image rectangle
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        r = rects[0]
        # Skip very small icons
        if w < ICON_MAX_WH and h < ICON_MAX_WH and r.x0 < page_w * 0.125:
            continue
        # Add image item
        items.append(
            {
                "type": "image",
                "data": to_base64(img_bytes),
                "bbox": (r.x0, r.y0, r.x1, r.y1),
                "w": w,
                "h": h,
                "is_small": (w <= ICON_MAX_WH and h <= ICON_MAX_WH),
            }
        )
    # Sort items in reading order
    def sort_key(item):
        x0, y0, x1, y1 = item["bbox"]
        y_center = (y0 + y1) / 2
        y_grouped = round(y_center / 5) * 5
        return (y_grouped, x0)
    items.sort(key=sort_key)
    return items
# Remove option markers (A., 1., etc.) from text
def clean_option_text(text: str) -> str:
    return OPT_MARK.sub("", text).strip()
# Convert option marker to index (0-based)
def get_option_index(marker: str) -> int:
    # If numeric option (1,2,3,4)
    if marker.isdigit():
        return int(marker) - 1
    # If alphabetic option (a,b,c,d)
    else:
        return ord(marker.lower()) - ord("a")
def extract_all_tables(pdf_path: str) -> Dict[int, List[Dict]]:
    # Helper function to filter out false positives
    # Only keep tables that look like real structured tables
    def looks_like_real_table(tbl):
        extracted = tbl.extract()
        if not extracted or len(extracted) < 3:
            return False
        # Count non-empty cells per column
        col_counts = [sum(1 for row in extracted if row and i < len(row) and row[i]) 
                      for i in range(len(extracted[0]))]
        # A "real" table should have multiple dense columns
        dense_cols = sum(1 for c in col_counts if c >= len(extracted) // 2)
        return dense_cols >= 3
    # Dictionary to hold extracted tables page-wise
    all_tables = {}
    # Table detection settings for pdfplumber
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 50,
    }
    # Open PDF with pdfplumber
    with pdfplumber.open(pdf_path) as plumber_doc:
        # Iterate through all pages
        for page_num, page in enumerate(plumber_doc.pages, start=1):
            try:
                # Detect tables on current page
                tables = page.find_tables(table_settings)
                extracted_tables = []
                for tbl in tables:
                    # Only keep valid tables
                    if looks_like_real_table(tbl):
                        data = tbl.extract()
                        extracted_tables.append({
                            "bbox": tbl.bbox,   
                            "data": data         
                        })
                # Save tables if any found on this page
                if extracted_tables:
                    all_tables[page_num] = extracted_tables
            except Exception as e:
                print(f"Warning: Failed to extract tables on page {page_num}: {e}")
    # Return all tables from the PDF, organized by page number
    return all_tables
# Define function to parse PDF and extract structured questions
def parse_pdf(pdf_path: str) -> Dict[str, Any]:
    # Open the PDF document using fitz
    doc = fitz.open(pdf_path)
    # Initialize result dictionary with empty sections list
    result = {"sections": []}
    # Initialize variable to track current section
    current_section = None
    # Initialize variable to track current question
    current_question = None
    # Initialize flag to track if processing options
    in_options = False
    # Extract all tables from the PDF
    all_tables = extract_all_tables(PDF_PATH)
    # Iterate through each page in the PDF
    for page in doc:
        # Collect text and image items from the page
        items = collect_page_items(page)
        # Check if page number is 5 for debugging
        if page.number == 5:
            # Open file to save page items as JSON
            with open("pageitems.json", "w", encoding="utf-8") as f:
                # Write items to JSON file with proper encoding
                json.dump(items, f, ensure_ascii=False, indent=4)
        # Initialize index for processing items
        idx = 0
        # Process each item on the page
        while idx < len(items):
            # Get the current item
            item = items[idx]
            # Check if item is not text (i.e., image)
            if item["type"] != "text":
                # Check if there is a current question
                if current_question:
                    # Check if in options mode and options exist
                    if in_options and current_question["options"]:
                        # Append image to the last option
                        current_question["options"][-1].append({"type": "image", "data": item["data"]})
                    # If not in options, add image to question
                    else:
                        current_question["question"].append({"type": "image", "data": item["data"]})
                # Increment index to next item
                idx += 1
                # Continue to next iteration
                continue
            # Extract text from the item
            text = item["text"]
            # Extract color from the item, default to black
            color = item.get("color", (0, 0, 0))
            # Check if text starts with "section :"
            if text.lower().startswith("section :"):
                # Calculate index of next item
                next_idx = idx + 1
                # Get section title from next item or use default
                title = items[next_idx]["text"].strip() if next_idx < len(items) else "Untitled"
                # Create new section dictionary
                current_section = {"title": title, "questions": []}
                # Append section to result
                result["sections"].append(current_section)
                # Update index to skip title item
                idx = next_idx + 1
                # Continue to next iteration
                continue
            # Check if text starts with "comprehension:"
            if text.lower().startswith("comprehension:"):
                # Create new question dictionary
                current_question = {"question": [], "options": [], "correctOption": None}
                # Check if current section exists
                if not current_section:
                    # Create default section if none exists
                    current_section = {"title": "Uncategorized", "questions": []}
                    # Append default section to result
                    result["sections"].append(current_section)
                # Append question to current section
                current_section["questions"].append(current_question)
                # Add comprehension text to question
                current_question["question"].append({"type": "text", "data": "Comprehension:"})
                # Reset options flag
                in_options = False
                # Increment index to next item
                idx += 1
                # Collect items until answer marker is found
                while idx < len(items) and not ANS.match(items[idx]["text"]):
                    # Get current comprehension item
                    comp_item = items[idx]
                    # Check if item is text
                    if comp_item["type"] == "text":
                        # Append text to question
                        current_question["question"].append({"type": "text", "data": comp_item["text"]})
                    # If item is image
                    else:
                        # Append image to question
                        current_question["question"].append({"type": "image", "data": comp_item["data"]})
                    # Increment index
                    idx += 1
                # Continue to next iteration
                continue
            # Check if text matches question number pattern
            if Q_START.match(text):
                # Create new question dictionary
                current_question = {"question": [], "options": [], "correctOption": None}
                # Check if current section exists
                if not current_section:
                    # Create default section if none exists
                    current_section = {"title": "Uncategorized", "questions": []}
                    # Append default section to result
                    result["sections"].append(current_section)
                # Append question to current section
                current_section["questions"].append(current_question)
                # Reset options flag
                in_options = False
                # Increment index
                idx += 1
                # Continue to next iteration
                continue
            # Check if text matches answer marker
            if ANS.match(text):
                # Set flag to indicate options processing
                in_options = True
                # Increment index
                idx += 1
                # Continue to next iteration
                continue
            # Check if text matches option marker and in options mode
            opt = OPT_MARK.match(text)
            if in_options and opt and current_question:
                # Extract option marker
                marker = opt.group(1)
                # Convert marker to 0-based index
                opt_idx = int(marker) if marker.isdigit() else (ord(marker.lower()) - ord("a") + 1)
                # Ensure options list has enough slots
                while len(current_question["options"]) < opt_idx:
                    # Append empty option list
                    current_question["options"].append([])
                # Clean option text
                cleaned = clean_option_text(text)
                # Append option text with color
                current_question["options"][opt_idx - 1].append({"type": "text", "data": cleaned, "_color": color})
                # Check if option is marked correct (green)
                if is_green(color):
                    # Set correct option index
                    current_question["correctOption"] = opt_idx
                # Increment index
                idx += 1
                # Continue to next iteration
                continue
            # Check if there is a current question
            if current_question:
                # Check if in options mode with existing options
                if in_options and current_question["options"]:
                    # Append text to last option with color
                    current_question["options"][-1].append({"type": "text", "data": text, "_color": color})
                # If not in options, append text to question
                else:
                    current_question["question"].append({"type": "text", "data": text})
            # Increment index
            idx += 1
    all_tables = extract_all_tables(PDF_PATH)
    # print("Extracted tables from PDF:", all_tables)
    # Iterate through all sections
    for section in result["sections"]:
        # Iterate through all questions in section
        for q in section["questions"]:
            # Iterate through all options in question
            for opt in q["options"]:
                # Iterate through all pieces in option
                for piece in opt:
                    # Remove temporary color field
                    piece.pop("_color", None)
    # Merge lines in questions based on length threshold
    result = merge_lines(result, line_length_threshold=111)
    # Return the parsed result
    return result
# Define function to merge text lines in questions based on length threshold
def merge_lines(result: Dict[str, Any], line_length_threshold: int) -> Dict[str, Any]:
    # Iterate through each section in the result
    for section in result["sections"]:
        # Iterate through each question in the section
        for question in section["questions"]:
            # Initialize list for merged question content
            merged_question = []
            # Initialize index for processing question items
            i = 0
            # Process each item in the question
            while i < len(question["question"]):
                # Get the current item
                current_item = question["question"][i]
                # Check if the item is an image
                if current_item["type"] == "image":
                    # Add image item as-is to merged list
                    merged_question.append(current_item)
                    # Increment index
                    i += 1
                    # Continue to next item
                    continue
                # Check if the item is text
                if current_item["type"] == "text":
                    # Get the text data from the item
                    current_text = current_item["data"]
                    # Check if text length meets or exceeds threshold
                    if len(current_text) >= line_length_threshold:
                        # Start with current text
                        merged_text = current_text
                        # Set index for checking next items
                        j = i + 1
                        # Merge consecutive text lines that meet threshold
                        while (j < len(question["question"]) and 
                               question["question"][j]["type"] == "text" and
                               len(question["question"][j]["data"]) >= line_length_threshold):
                            # Append next text with a space
                            merged_text += " " + question["question"][j]["data"]
                            # Increment inner index
                            j += 1
                        # Merge first text line that doesn't meet threshold, if it exists
                        if (j < len(question["question"]) and 
                            question["question"][j]["type"] == "text"):
                            # Append next text with a space
                            merged_text += " " + question["question"][j]["data"]
                            # Increment inner index
                            j += 1
                        # Add merged text as a single item
                        merged_question.append({
                            "type": "text",
                            "data": merged_text
                        })
                        # Update index to next unprocessed item
                        i = j
                    else:
                        # Add text item as-is if below threshold
                        merged_question.append(current_item)
                        # Increment index
                        i += 1
            # Update question with merged content
            question["question"] = merged_question
    # Return the modified result
    return result
# Standard Python entry point check
if __name__ == "__main__":
    # Call parse_pdf function with the given PDF path
    parsed = parse_pdf(PDF_PATH)
    # Open the output JSON file in write mode with UTF-8 encoding
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        # Dump the parsed result dictionary into the JSON file
        json.dump(parsed, fh, ensure_ascii=False, indent=2)
    # Print confirmation message with file name
    print("saved", OUTPUT_JSON)
