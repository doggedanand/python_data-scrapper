# PDF processing
from pydoc import text
import fitz
# Image encoding
import base64
# JSON handling
import json
# Regex patterns
import re
# Type hints
from typing import Any, Dict, List, Tuple, Optional
# Input PDF file path
PDF_PATH = "SSC-CGL-Question-Paper-10-September-2024-Shift-3.pdf"
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
    all_texts_for_testing = []
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
                all_texts_for_testing.append({"text": text_stripped, "bbox": bbox})
    # Loop through all images on page
    all_images_for_testing = []
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
        # Skip very small icons
        if w < ICON_MAX_WH and h < ICON_MAX_WH:
            continue
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
        if not (w <= ICON_MAX_WH and h <= ICON_MAX_WH):
            all_images_for_testing.append({
                "data": to_base64(img_bytes),
                "bbox": (r.x0, r.y0, r.x1, r.y1)
            })
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
# Parse PDF and extract structured questions
def parse_pdf(pdf_path: str) -> Dict[str, Any]:
    # Open PDF document
    doc = fitz.open(pdf_path)
    # Initialize result structure with empty sections list
    result = {"sections": []}
    # Track current section being processed
    current_section = None
    # Track current question being processed
    current_question = None
    # Flag to indicate if we're currently processing options
    in_options = False
    # Loop through each page in the document
    for page in doc:
        # Skip page 500 (seems to be a specific exclusion)
        if page.number != 500:
            # Collect all text and image items from current page
            items = collect_page_items(page) 
            # Save page items to JSON file for last page (debugging purpose)
            if page.number == len(doc) - 1:
                with open("pageitems.json", "w", encoding="utf-8") as f:
                    json.dump(items, f, ensure_ascii=False, indent=4)
        # Process each item (text or image) on the page
        for idx, item in enumerate(items):
            # Handle non-text items (images)
            if item["type"] != "text":
                # Only add images if we have a current question
                if current_question:
                    # If processing options, add image to last option
                    if in_options and current_question["options"]:
                        current_question["options"][-1].append({"type": "image", "data": item["data"]})
                    # Otherwise add image to question content
                    else:
                        current_question["question"].append({"type": "image", "data": item["data"]})
                # Skip to next item
                continue
            # Extract text content and color from current item
            text, color = item["text"], item.get("color", (0, 0, 0))
            # Check if text indicates start of new section
            if text.lower().startswith("section"):
                # Get section title from next item
                next_idx = idx + 1
                title = items[next_idx]["text"].strip() if next_idx < len(items) else "Untitled"
                # Create new section with title and empty questions list
                current_section = {"title": title, "questions": []}
                # Add section to result
                result["sections"].append(current_section)
                # Move to next item
                continue
            # Check if text matches question number pattern (Q. 1, Q. 2, etc.)
            if Q_START.match(text):
                # Create new question structure
                current_question = {"question": [], "options": [], "correctOption": None}
                # Create default section if none exists
                if not current_section:
                    current_section = {"title": "Uncategorized", "questions": []}
                    result["sections"].append(current_section)
                # Add question to current section
                current_section["questions"].append(current_question)
                # Reset options flag
                in_options = False
                # Move to next item
                continue
            # Check if text matches answer block pattern
            if ANS.match(text):
                # Set flag to indicate we're now processing options
                in_options = True
                # Move to next item
                continue
            # Check if text matches option marker pattern (1., A), etc.)
            opt = OPT_MARK.match(text)
            if in_options and opt and current_question:
                # Extract option marker (1, 2, A, B, etc.)
                marker = opt.group(1)
                # Convert marker to option index (1-based)
                opt_idx = int(marker) if marker.isdigit() else (ord(marker.lower()) - ord("a") + 1)
                # Ensure options list has enough empty slots
                while len(current_question["options"]) < opt_idx:
                    current_question["options"].append([])
                # Clean option text by removing marker
                cleaned = clean_option_text(text)
                # Add option text to appropriate index
                current_question["options"][opt_idx - 1].append(
                    {"type": "text", "data": cleaned, "_color": color}
                )
                # Check if option text is green (indicates correct answer)
                if is_green(color):
                    current_question["correctOption"] = opt_idx
                # Move to next item
                continue
            # Handle regular text that doesn't match special patterns
            if current_question:
                # If processing options, add to last option
                if in_options and current_question["options"]:
                    current_question["options"][-1].append({"type": "text", "data": text, "_color": color})
                # Otherwise add to question content
                else:
                    current_question["question"].append({"type": "text", "data": text})
    # Remove temporary color fields from all options
    for section in result["sections"]:
        for q in section["questions"]:
            for opt in q["options"]:
                for piece in opt:
                    piece.pop("_color", None)
    # Return the structured result
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

