# PDF processing
import fitz
# Image encoding
import base64
# JSON handling
import json
# Regex patterns
import re
# Stats functions
import statistics
# Type hints
from typing import Any, Dict, List, Tuple, Optional
# Input PDF file path
PDF_PATH = "SSC-CGL-Question-Paper-10-September-2024-Shift-3_removed.pdf"
# Output JSON file path
OUTPUT_JSON = "output.json"
# Max width/height for small icons
ICON_MAX_WH = 40  
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
            # Loop through spans in line
            for span in line.get("spans", []):
                # Extract text from span
                text = span.get("text", "")
                # Skip if text missing
                if text is None:
                    continue
                # Remove extra spaces
                text_stripped = text.strip()
                # Skip if empty text
                if not text_stripped:
                    continue
                # Get bounding box of text
                bbox = span.get("bbox", (0, 0, 0, 0))
                # Raw color value
                color_raw = span.get("color", None)
                # Parse RGB color if available else default black
                color = parse_color(color_raw) if color_raw is not None else (0, 0, 0)
                # Add text item
                items.append({"type": "text", "text": text_stripped, "bbox": bbox, "color": color})
    # Loop through all images on page
    for img in page.get_images(full=True):
        # Image reference id
        xref = img[0]
        # Extract image info
        info = page.parent.extract_image(xref)
        # Get width and height
        w, h = info.get("width", 0), info.get("height", 0)
        # Skip very small icons
        if w < ICON_MAX_WH and h < ICON_MAX_WH:
            continue
        # Extract image bytes
        img_bytes = info["image"]
        # Get image rectangle
        rects = page.get_image_rects(xref)
        # Skip if no rect found
        if not rects:
            continue
        # Take first rect
        r = rects[0]
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
    # Sort items by vertical (y) then horizontal (x)
    items.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))
    # Return all collected items
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
    # Initialize result with sections list
    result = {"sections": []}
    # Current section being processed
    current_section: Optional[Dict[str, Any]] = None
    # Store all items from pages
    all_items: List[Dict[str, Any]] = []
    # Check if document has pages
    if doc.page_count > 0:
        # Get page height for offset
        page_height = doc[0].bound()[3]
        # Loop through all pages
        for pagenum, page in enumerate(doc):
            # Collect items from page
            page_items = collect_page_items(page)
            # Calculate offset based on page number
            offset = pagenum * page_height
            # Adjust bbox with offset
            for it in page_items:
                b = it["bbox"]
                it["bbox"] = (b[0], b[1] + offset, b[2], b[3] + offset)
            # Add items to global list
            all_items.extend(page_items)
    # Sort all items by vertical then horizontal position
    all_items.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))
    # Collect text heights for median
    heights = [(it["bbox"][3] - it["bbox"][1]) for it in all_items if it["type"] == "text"]
    # Median height fallback to 12.0
    median_height = statistics.median(heights) if heights else 12.0
    # Threshold to detect paragraph breaks
    paragraph_threshold = median_height * 1.6
    # Start index
    i = 0
    # Total number of items
    N = len(all_items)
    # Loop through items
    while i < N:
        # Current item
        it = all_items[i]
        # Skip if not text
        if it["type"] != "text":
            i += 1
            continue
        # Extract text
        text = it["text"]
        # Skip if header/footer
        if is_header_footer(text):
            i += 1
            continue
        # Match full section pattern
        mfull = SECTION_FULL.match(text)
        # If section found
        if mfull:
            # Extract section title
            title = mfull.group(1).strip()
            # Reuse last section if same title
            if result["sections"] and result["sections"][-1]["title"] == title:
                current_section = result["sections"][-1]
            # Otherwise create new section
            else:
                current_section = {"title": title, "questions": []}
                result["sections"].append(current_section)
            i += 1
            continue
        # Match section prefix only
        mpref = SECTION_PREFIX.match(text)
        # If prefix found
        if mpref:
            # Start from next item
            j = i + 1
            # Collect possible title parts
            title_parts: List[str] = []
            # Scan forward for title
            while j < N and len(title_parts) < 6:
                nxt = all_items[j]
                if nxt["type"] != "text":
                    break
                if is_header_footer(nxt["text"]):
                    break
                if Q_START.match(nxt["text"]):
                    break
                if ANS.match(nxt["text"]):
                    break
                if OPT_MARK.match(nxt["text"]):
                    break
                title_parts.append(nxt["text"])
                j += 1
            # If title found
            if title_parts:
                title = " ".join(title_parts).strip()
                if result["sections"] and result["sections"][-1]["title"] == title:
                    current_section = result["sections"][-1]
                else:
                    current_section = {"title": title, "questions": []}
                    result["sections"].append(current_section)
                i = j
                continue
            else:
                i += 1
                continue
        # Match question start
        mq = Q_START.match(text)
        if mq:
            # If no section, create default
            if current_section is None:
                current_section = {"title": "Uncategorized", "questions": []}
                result["sections"].append(current_section)
            # Initialize question object
            question: Dict[str, Any] = {"question": [], "options": [], "correctOption": None}
            current_section["questions"].append(question)
            # Start scanning from next item
            start = i + 1
            j = start
            # Find question end
            while j < N:
                nxt = all_items[j]
                if nxt["type"] == "text" and (
                    SECTION_FULL.match(nxt["text"]) or SECTION_PREFIX.match(nxt["text"]) or Q_START.match(nxt["text"])
                ):
                    break
                j += 1
            # Copy items for question
            q_items = []
            for k in range(start, j):
                entry = dict(all_items[k])
                entry["_assigned"] = False
                entry["_assigned_to"] = None 
                q_items.append(entry)
            # Remove answer text
            q_items = [itq for itq in q_items if not (itq["type"] == "text" and ANS.match(itq["text"]))]
            # Remove header/footer
            q_items = [itq for itq in q_items if not (itq["type"] == "text" and is_header_footer(itq["text"]))]
            # Separate text items
            text_items = [itq for itq in q_items if itq["type"] == "text"]
            # Separate image items
            image_items = [itq for itq in q_items if itq["type"] == "image"]
            # Map option labels
            label_map: Dict[int, Dict[str, Any]] = {}
            for t in text_items:
                m = OPT_MARK.match(t["text"])
                if m:
                    idx = get_option_index(m.group(1))
                    label_map[idx] = t
            # If no labels found
            if not label_map:
                buf_texts: List[str] = []
                last_bottom = None
                for itq in q_items:
                    if itq["type"] == "text":
                        cur_top = itq["bbox"][1]; cur_bottom = itq["bbox"][3]
                        if last_bottom is not None and (cur_top - last_bottom) > paragraph_threshold:
                            buf_texts.append("")  
                        buf_texts.append(itq["text"])
                        last_bottom = cur_bottom
                    else:
                        if buf_texts:
                            question["question"].append({"type": "text", "data": "\n".join(buf_texts)})
                            buf_texts = []
                        question["question"].append({"type": "image", "data": itq["data"]})
                if buf_texts:
                    question["question"].append({"type": "text", "data": "\n".join(buf_texts)})
                i = j
                continue 
            # Sort labels by vertical position
            labels_sorted = sorted(label_map.items(), key=lambda kv: kv[1]["bbox"][1])  
            # Get first label positions
            first_label_top = labels_sorted[0][1]["bbox"][1]
            first_label_bottom = labels_sorted[0][1]["bbox"][3]
            # Collect texts above first option
            above_label_texts = [t for t in text_items if t["bbox"][1] < first_label_top and not OPT_MARK.match(t["text"])]
            # Get last bottom position of text above
            last_text_bottom = max((t["bbox"][3] for t in above_label_texts), default=None)
            # Bands for each option
            bands: Dict[int, Tuple[float, float]] = {}
            for idx_pos, (opt_idx, label_item) in enumerate(labels_sorted):
                top = label_item["bbox"][1] - paragraph_threshold * 0.5
                if idx_pos + 1 < len(labels_sorted):
                    next_top = labels_sorted[idx_pos + 1][1]["bbox"][1]
                    bottom = (label_item["bbox"][1] + next_top) / 2.0
                else:
                    bottom = label_item["bbox"][3] + paragraph_threshold * 2.0
                bands[opt_idx] = (top, bottom)
            # Get maximum label index
            max_label_idx = max(label_map.keys())
            # Create options list
            options: List[List[Dict[str, Any]]] = [[] for _ in range(max_label_idx + 1)]
            # Assign labels as option starters
            for opt_idx, label_item in label_map.items():
                label_item["_assigned"] = True
                label_item["_assigned_to"] = opt_idx
                cleaned = clean_option_text(label_item["text"])
                if cleaned:
                    options[opt_idx].append({"type": "text", "data": cleaned, "_color": label_item.get("color")})
            # Assign text pieces to options
            for t in text_items:
                if OPT_MARK.match(t["text"]):
                    continue
                cy = (t["bbox"][1] + t["bbox"][3]) / 2.0
                assigned_flag = False
                for opt_idx, (top, bottom) in bands.items():
                    if cy >= top and cy < bottom:
                        t["_assigned"] = True
                        t["_assigned_to"] = opt_idx
                        options[opt_idx].append({"type": "text", "data": t["text"], "_color": t.get("color")})
                        assigned_flag = True
                        break
                if not assigned_flag:
                    continue
            # Assign images to options
            for im in image_items:
                if im.get("_assigned"):
                    continue
                cy = (im["bbox"][1] + im["bbox"][3]) / 2.0
                cx = (im["bbox"][0] + im["bbox"][2]) / 2.0
                if last_text_bottom is not None and cy < first_label_top:
                    dist_to_text = abs(cy - last_text_bottom)
                    label_center_y = (label_map[labels_sorted[0][0]]["bbox"][1] + label_map[labels_sorted[0][0]]["bbox"][3]) / 2.0
                    dist_to_label = abs(cy - label_center_y)
                    if dist_to_text <= dist_to_label + (median_height * 0.6):
                        im["_assigned"] = False
                        im["_assigned_to"] = None
                        continue 
                best_idx = None
                best_dist = 1e9
                for opt_idx, label_item in label_map.items():
                    label_cy = (label_item["bbox"][1] + label_item["bbox"][3]) / 2.0
                    d = abs(cy - label_cy)
                    if d < best_dist:
                        best_dist = d
                        best_idx = opt_idx
                if best_idx is not None:
                    im["_assigned"] = True
                    im["_assigned_to"] = best_idx
                    options[best_idx].append({"type": "image", "data": im["data"], "_bbox": im.get("bbox")})
            # Count options with text
            has_text_counts = sum(1 for opt in options if any(p.get("type") == "text" for p in opt))
            # Count options with images
            has_image_counts = sum(1 for opt in options if any(p.get("type") == "image" for p in opt))
            # If enough text options exist
            if has_text_counts >= max(1, (len(options) // 2)):
                for opt_idx in range(len(options)):
                    imgs = [p for p in options[opt_idx] if p.get("type") == "image"]
                    if imgs:
                        options[opt_idx] = [p for p in options[opt_idx] if p.get("type") != "image"]
                        for im in imgs:
                            for itq in q_items:
                                if itq["type"] == "image" and itq.get("data") == im.get("data"):
                                    itq["_assigned"] = False
                                    itq["_assigned_to"] = None
                                    break
            # If more images than texts
            else:
                for opt_idx in range(len(options)):
                    texts = [p for p in options[opt_idx] if p.get("type") == "text"]
                    if texts:
                        options[opt_idx] = [p for p in options[opt_idx] if p.get("type") != "text"]
                        for tx in texts:
                            for itq in q_items:
                                if itq["type"] == "text" and itq.get("text") == tx.get("data"):
                                    itq["_assigned"] = False
                                    itq["_assigned_to"] = None
                                    break
            # Handle extra images
            if any(any(p["type"] == "image" for p in opt) for opt in options):
                extras: List[Tuple[int, Dict[str, Any]]] = [] 
                empty_image_slots: List[int] = []
                for idx_opt, opt in enumerate(options):
                    imgs = [p for p in opt if p.get("type") == "image"]
                    if len(imgs) == 0:
                        empty_image_slots.append(idx_opt)
                    elif len(imgs) > 1:
                        for extra in imgs[1:]:
                            extras.append((idx_opt, extra))
                        first_img = imgs[0]
                        options[idx_opt] = [p for p in opt if not (p.get("type") == "image" and p is not first_img)]
                for from_idx, image_piece in extras:
                    if not empty_image_slots:
                        for itq in q_items:
                            if itq["type"] == "image" and itq.get("data") == image_piece.get("data"):
                                itq["_assigned"] = False
                                itq["_assigned_to"] = None
                                break
                        continue
                    best_empty = min(empty_image_slots, key=lambda x: abs(x - from_idx))
                    empty_image_slots.remove(best_empty)
                    options[best_empty].append(image_piece)
                    for itq in q_items:
                        if itq["type"] == "image" and itq.get("data") == image_piece.get("data"):
                            itq["_assigned"] = True
                            itq["_assigned_to"] = best_empty
                            break
            # Reassign unassigned items
            for itq in q_items:
                if itq["_assigned_to"] is None:
                    if itq["type"] == "image":
                        for opt_idx in range(len(options)):
                            for p in options[opt_idx]:
                                if p.get("type") == "image" and p.get("data") == itq.get("data"):
                                    itq["_assigned"] = True
                                    itq["_assigned_to"] = opt_idx
                                    break
                            if itq["_assigned"]:
                                break
                    elif itq["type"] == "text":
                        for opt_idx in range(len(options)):
                            for p in options[opt_idx]:
                                if p.get("type") == "text" and p.get("data") == itq.get("text"):
                                    itq["_assigned"] = True
                                    itq["_assigned_to"] = opt_idx
                                    break
                            if itq["_assigned"]:
                                break
            # Buffer for question text lines
            buf_lines: List[str] = []
            last_bottom = None
            # Helper function to flush buffer
            def flush_buf():
                nonlocal buf_lines
                if buf_lines:
                    question["question"].append({"type": "text", "data": "\n".join(buf_lines)})
                    buf_lines = []
            # Collect remaining unassigned question parts
            for itq in q_items:
                if itq.get("_assigned"):
                    continue
                if itq["type"] == "text":
                    cur_top = itq["bbox"][1]; cur_bottom = itq["bbox"][3]
                    if last_bottom is not None and (cur_top - last_bottom) > paragraph_threshold:
                        buf_lines.append("") 
                    buf_lines.append(itq["text"])
                    last_bottom = cur_bottom
                else:
                    flush_buf()
                    question["question"].append({"type": "image", "data": itq["data"]})
            flush_buf()
            # Assign options to question
            question["options"] = []
            for opt_idx in range(len(options)):
                opt_pieces = []
                for itq in q_items:
                    if itq.get("_assigned_to") == opt_idx:
                        if itq["type"] == "text":
                            opt_pieces.append({"type": "text", "data": itq["text"], "_color": itq.get("color")})
                        else:
                            opt_pieces.append({"type": "image", "data": itq["data"]})
                if not opt_pieces and options[opt_idx]:
                    for p in options[opt_idx]:
                        opt_pieces.append(p)
                question["options"].append(opt_pieces)
            i = j
            continue
        i += 1
    # Final pass for each section
    for section in result["sections"]:
        for q in section["questions"]:
            if q["options"]:
                for idx, opt in enumerate(q["options"], start=1):
                    first_txt = next((p for p in opt if p.get("type") == "text" and "_color" in p), None)
                    if first_txt and not is_red(first_txt["_color"]):
                        q["correctOption"] = idx
                        break
            for opt in q["options"]:
                for piece in opt:
                    if isinstance(piece, dict):
                        piece.pop("_color", None)
                        piece.pop("_bbox", None)
                        piece.pop("is_small", None)
            for piece in q["question"]:
                if isinstance(piece, dict):
                    piece.pop("_bbox", None)
                    piece.pop("is_small", None)
    # Return structured result
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

