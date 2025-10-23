# PDF processing
from pydoc import text
from venv import logger
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
PDF_PATH = "./pdfs/SSC-CGL-17-September-2024-Shift-1.pdf"
# PDF_PATH = "SSC-CGL-24-September-2024-Shift-1.pdf"
# Output JSON file path
OUTPUT_JSON = "output.json"
# Max width/height for small icons
DEFAULT_ICON_MAX_WH = 35
# Pattern to detect question numbers (e.g., Q. 1)
DEFAULT_Q_START = re.compile(r"^\s*Q\.\s*(\d+)", re.IGNORECASE)
# Pattern to detect full section titles
DEFAULT_SECTION_PATTERN = re.compile(r"^\s*Section\s*[:\.]\s*(.+)$", re.IGNORECASE)
# Pattern to detect answer lines
DEFAULT_ANS = re.compile(r"^\s*Ans[\s\.:]*([1-9])?(.*)$", re.IGNORECASE)
# Pattern to detect option markers (1., A), etc.)
DEFAULT_OPT_MARK = re.compile(r"^\s*([1-4a-dA-D])[\.\)]\s*", re.IGNORECASE)
# Patterns to identify and skip headers/footers
DEFAULT_GLOBAL_IGNORE_PATTERNS = [
    re.compile(r"SSC", re.IGNORECASE),
    re.compile(r"Tier", re.IGNORECASE),
    re.compile(r"Shift", re.IGNORECASE),
    re.compile(r"Page \d+", re.IGNORECASE),
]
# Default line length threshold for merging lines (characters)
DEFAULT_MERGE_LINE_LENGTH_THRESHOLD = 111
# Global variables for settings
Q_START = [DEFAULT_Q_START]
SECTION_PATTERNS = [DEFAULT_SECTION_PATTERN]
ANS_PATTERN = DEFAULT_ANS
OPT_MARK = DEFAULT_OPT_MARK
AFTER_OPTION_IGNORE_PATTERNS = []
GLOBAL_IGNORE_PATTERNS = DEFAULT_GLOBAL_IGNORE_PATTERNS
ICON_MAX_WH = DEFAULT_ICON_MAX_WH
PAGE_WIDTH = 0
MERGE_LINE_LENGTH_THRESHOLD = DEFAULT_MERGE_LINE_LENGTH_THRESHOLD
HAS_SECTIONS = False  
QUESTION_FORMAT = 'mcq'  
QUESTION_TYPES = set(['text'])  
# Initialize settings from provided config or defaults
def initialize_settings(settings: Dict[str, Any] = None):
    global Q_START, SECTION_PATTERNS, ANS_PATTERN, OPT_MARK, PAGE_WIDTH
    global GLOBAL_IGNORE_PATTERNS, ICON_MAX_WH, MERGE_LINE_LENGTH_THRESHOLD, HAS_SECTIONS, QUESTION_FORMAT,QUESTION_TYPES
    parsing_config = settings.get("attributes", {}).get("parsingConfig", {}) if settings else {}
    # Helper function to safely compile regex patterns
    def compile_regex(pattern, default):
        if isinstance(pattern, str) and pattern.strip():
            try:
                return re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                return default
        return default
    # Set question start patterns (list)
    question_patterns = parsing_config.get("question_start_pattern", [])
    if isinstance(question_patterns, str):  # Backward compatibility
        question_patterns = [question_patterns]
    Q_START = [compile_regex(p, DEFAULT_Q_START) for p in question_patterns if isinstance(p, str)] or [DEFAULT_Q_START]
    # Set section patterns (list)
    section_patterns = parsing_config.get("section_pattern", [])
    if isinstance(section_patterns, str):  # Backward compatibility
        section_patterns = [section_patterns]
    SECTION_PATTERNS = [compile_regex(p, DEFAULT_SECTION_PATTERN) for p in section_patterns if isinstance(p, str)] or [DEFAULT_SECTION_PATTERN]
    # Set other regex patterns
    ANS_PATTERN = compile_regex(parsing_config.get("answer_pattern"), DEFAULT_ANS)
    OPT_MARK = compile_regex(parsing_config.get("option_pattern"), DEFAULT_OPT_MARK)
    # Set after_option_ignore_patterns
    after_option_patterns = parsing_config.get("after_option_ignore_pattern", [])
    global AFTER_OPTION_IGNORE_PATTERNS
    AFTER_OPTION_IGNORE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in after_option_patterns if isinstance(p, str)]
    # Set global_ignore_patterns
    header_patterns = parsing_config.get("global_ignore_patterns", [])
    GLOBAL_IGNORE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in header_patterns if isinstance(p, str)] or DEFAULT_GLOBAL_IGNORE_PATTERNS
    # Set numeric settings
    ICON_MAX_WH =  int(parsing_config.get("icon_max_wh", DEFAULT_ICON_MAX_WH)) if isinstance(parsing_config.get("icon_max_wh"), (int, float, str)) and str(parsing_config.get("icon_max_wh")).isdigit() else DEFAULT_ICON_MAX_WH
    MERGE_LINE_LENGTH_THRESHOLD = int(parsing_config.get("merge_line_length_threshold", DEFAULT_MERGE_LINE_LENGTH_THRESHOLD)) if isinstance(parsing_config.get("merge_line_length_threshold"), (int, float, str)) and str(parsing_config.get("merge_line_length_threshold")).isdigit() else DEFAULT_MERGE_LINE_LENGTH_THRESHOLD
    PAGE_WIDTH = float(parsing_config.get("pdf_page_wh", 0)) if isinstance(parsing_config.get("pdf_page_wh"), (int, float, str)) and str(parsing_config.get("pdf_page_wh")) else 0
    # Set HAS_SECTIONS from attributes.has_section, fallback to expectedOutputData
    HAS_SECTIONS = settings.get("attributes", {}).get("has_section", False)
    expected = settings.get("attributes", {}).get("expectedOutputData", {})
    if not HAS_SECTIONS:
        HAS_SECTIONS = 'sections' in expected  # Fallback for older records
    # Infer QUESTION_TYPES and QUESTION_FORMAT
    if HAS_SECTIONS:
        sample_q = expected['sections'][0]['questions'][0] if expected.get('sections') and expected['sections'][0].get('questions') else {}
    else:
        sample_q = expected.get('questions', [{}])[0] if expected.get('questions') else {}
    if sample_q:
        QUESTION_TYPES = set(item['type'] for item in sample_q.get('question', []) if isinstance(item, dict) and 'type' in item) or set(['text'])
        num_options = len(sample_q.get('options', []))
        if num_options == 4:
            QUESTION_FORMAT = 'mcq'
        elif num_options == 2:
            QUESTION_FORMAT = 'true_false'
        elif num_options == 0:
            QUESTION_FORMAT = 'descriptive'
        else:
            QUESTION_FORMAT = 'mcq'
# Check if text matches header/footer patterns
def is_ignored_text(text: str) -> bool:
    # Return True if any pattern matches
    return any(p.search(text) for p in GLOBAL_IGNORE_PATTERNS)
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
# Define function to check if two bounding boxes overlap with a given tolerance
def bbox_overlaps(b1, b2, tol=0.5) -> bool:
    # Unpack first bounding box coordinates
    x0, y0, x1, y1 = b1
    # Unpack second bounding box coordinates
    X0, Y0, X1, Y1 = b2
    # Return True if bboxes overlap within tolerance, False otherwise
    return not (x1 < X0 - tol or x0 > X1 + tol or y1 < Y0 - tol or y0 > Y1 + tol)
def match_question_start(text):
    for pattern in Q_START:
        match = pattern.match(text)
        if match:
            return match
    return None
# Collect text and images from a PDF page
def collect_page_items(page) -> List[Dict[str, Any]]:
    # print('page width', PAGE_WIDTH)
    # Store collected items
    items: List[Dict[str, Any]] = []
    # Extract page content as dict
    d = page.get_text("dict")
    # Get current page number (1-based)
    page_num = page.number + 1
    # Extract tables on this page
    tables_for_page = extract_all_tables(PDF_PATH, page_num).get(page_num, [])
    # Get bounding boxes of all tables on the page
    table_bboxes = [tbl["bbox"] for tbl in tables_for_page]
    # Loop through all blocks
    for block in d.get("blocks", []):
        # Skip non-text blocks
        if block.get("type", 0) != 0:
            continue
        # Iterate through each line in the block
        for line in block.get("lines", []):
            # Iterate through each span in the line
            for span in line.get("spans", []):
                # Extract text from the span, default to empty string if not found
                text = span.get("text", "")
                # Skip if text is None
                if text is None:
                    continue
                # Strip whitespace from text
                text_stripped = text.strip()
                # Skip if text is empty after stripping
                if not text_stripped:
                    continue
                # Extract bounding box coordinates, default to (0, 0, 0, 0) if not found
                bbox = span.get("bbox", (0, 0, 0, 0))
                # skip if span falls inside/overlaps a table
                if any(bbox_overlaps(bbox, tb) for tb in table_bboxes):
                    continue
                # Extract raw color value, default to None if not found
                color_raw = span.get("color", None)
                # Parse color to RGB tuple, default to black (0, 0, 0) if no color
                color = parse_color(color_raw) if color_raw is not None else (0, 0, 0)
                # Append text item with its properties to the items list
                items.append({
                    # Set item type to text
                    "type": "text",
                    # Store stripped text content
                    "text": text_stripped,
                    # Store bounding box coordinates
                    "bbox": bbox,
                    # Store RGB color tuple
                    "color": color
                })
    # Iterate through all images on the page with full details
    for img in page.get_images(full=True):
        # Extract the image's XREF (cross-reference) identifier
        xref = img[0]
        # Extract image information using the XREF
        info = page.parent.extract_image(xref)
        # Get the image's raw bytes
        img_bytes = info["image"]
        # Get the image's file extension
        img_ext = info["ext"]
        # Check for soft mask (smask) if image tuple has more than one element
        smask = img[1] if len(img) > 1 else None
        # Process soft mask if present
        if smask:
            # Attempt to apply the soft mask to the image
            try:
                # Extract soft mask image information
                smask_info = page.parent.extract_image(smask)
                # Proceed if soft mask information is available
                if smask_info:
                    # Import PIL for image processing
                    from PIL import Image
                    # Import io for handling byte streams
                    import io
                    # Open base image and convert to RGB
                    base_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    # Open soft mask image and convert to grayscale
                    mask_img = Image.open(io.BytesIO(smask_info["image"])).convert("L")
                    # Apply the soft mask as an alpha channel
                    base_img.putalpha(mask_img)
                    # Create a buffer for saving the processed image
                    buf = io.BytesIO()
                    # Save the image with alpha channel as PNG
                    base_img.save(buf, format="PNG")
                    # Update image bytes with the processed image
                    img_bytes = buf.getvalue()
                    # Update file extension to PNG
                    img_ext = "png"
            # Handle any errors during soft mask processing
            except Exception as e:
                # Print error message for soft mask processing failure
                print(f"Error applying smask: {e}")
        # Extract image width, default to 0 if not found
        w, h = info.get("width", 0), info.get("height", 0)
        # Get the page width
        page_w = page.rect.width
        # Get the page height
        page_h = page.rect.height
        # Skip images that are too large (likely watermarks)
        if w > page_w * 0.7 and h > page_h * 0.7:
            # Continue to next image
            continue
        # Get the image's bounding rectangles
        rects = page.get_image_rects(xref)
        # Skip if no rectangles are found
        if not rects:
            # Continue to next image
            continue
        # Use the first rectangle for bounding box
        r = rects[0]
        # Skip small icons located near the page's left margin
        if w < ICON_MAX_WH and h < ICON_MAX_WH and r.x0 < page_w * PAGE_WIDTH:
            # Continue to next image
            continue
        # Append image item to the items list
        items.append(
            {
                # Set item type to image
                "type": "image",
                # Convert image bytes to base64 string
                "data": 'to_base64(img_bytes)',
                # Store bounding box coordinates
                "bbox": (r.x0, r.y0, r.x1, r.y1),
                # Store image width
                "w": w,
                # Store image height
                "h": h,
                # Indicate if the image is small (icon-sized)
                "is_small": (w <= ICON_MAX_WH and h <= ICON_MAX_WH),
            }
        )
    # Check if tables were found on the page
    if tables_for_page:
        # Extract bounding boxes for all tables on the page
        all_bboxes = [tbl["bbox"] for tbl in tables_for_page]
        # Find minimum x-coordinate (left edge) across all table bounding boxes
        x0 = min(b[0] for b in all_bboxes)
        # Find minimum y-coordinate (top edge) across all table bounding boxes
        y0 = min(b[1] for b in all_bboxes)
        # Find maximum x-coordinate (right edge) across all table bounding boxes
        x1 = max(b[2] for b in all_bboxes)
        # Find maximum y-coordinate (bottom edge) across all table bounding boxes
        y1 = max(b[3] for b in all_bboxes)
        # Append a single table item to the items list
        items.append({
            # Set item type to table
            "type": "table",
            # Collect data from all tables on the page
            "data": [tbl["data"] for tbl in tables_for_page],
            # Define a single bounding box encompassing all tables
            "bbox": (x0, y0, x1, y1)  # single bbox for all
        })
    # Define function to sort items in reading order
    def sort_key(item):
        # Extract bounding box coordinates from the item
        x0, y0, x1, y1 = item["bbox"]
        # Calculate the vertical center of the bounding box
        y_center = (y0 + y1) / 2
        # Group y-center into 5-unit intervals for consistent sorting
        y_grouped = round(y_center / 5) * 5
        # Return a tuple for sorting by grouped y-coordinate and x-coordinate
        return (y_grouped, x0)
    # Sort the items list using the sort_key function
    items.sort(key=sort_key)
    # Return the sorted items list
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
# Define function to extract tables from a PDF, optionally for a specific page
def extract_all_tables(pdf_path: str, page_num: Optional[int] = None) -> Dict[int, List[Dict]]:
    # Define helper function to filter out false positive tables
    def looks_like_real_table(tbl):
        # Extract table content as a list of rows
        extracted = tbl.extract()
        # Check if table is empty or has fewer than 3 rows
        if not extracted or len(extracted) < 3:
            # Return False if table is not valid
            return False
        # Count non-empty cells per column
        col_counts = [sum(1 for row in extracted if row and i < len(row) and row[i]) 
                      for i in range(len(extracted[0]))]
        # Count columns with at least half the rows having non-empty cells
        dense_cols = sum(1 for c in col_counts if c >= len(extracted) // 2)
        # Return True if table has at least 3 dense columns
        return dense_cols >= 3
    # Initialize dictionary to store tables by page number
    all_tables = {}
    # Define table detection settings for pdfplumber
    table_settings = {
        # Use lines for vertical table boundaries
        "vertical_strategy": "lines",
        # Use lines for horizontal table boundaries
        "horizontal_strategy": "lines",
        # Set tolerance for snapping lines to table edges
        "snap_tolerance": 3,
        # Set tolerance for joining nearby lines
        "join_tolerance": 3,
        # Set minimum length for table edges
        "edge_min_length": 50,
    }
    # Open PDF file using pdfplumber
    with pdfplumber.open(pdf_path) as plumber_doc:
        # Determine pages to process: single page if specified, else all pages
        pages_to_process = [page_num] if page_num else range(1, len(plumber_doc.pages) + 1)
        # Iterate through pages to process
        for pnum in pages_to_process:
            # Get page object from pdfplumber (0-based index)
            page = plumber_doc.pages[pnum - 1]
            # Begin try block to handle potential errors during table extraction
            try:
                # Find tables on the page using specified settings
                tables = page.find_tables(table_settings)
                # Initialize list to store valid tables for this page
                extracted_tables = []
                # Iterate through detected tables
                for tbl in tables:
                    # Check if table is valid using helper function
                    if looks_like_real_table(tbl):
                        # Extract table data as a list of rows
                        data = tbl.extract()
                        # Append table data and bounding box to list
                        extracted_tables.append({"bbox": tbl.bbox, "data": data})
                # Check if any valid tables were found
                if extracted_tables:
                    # Store extracted tables in dictionary under page number
                    all_tables[pnum] = extracted_tables
            # Catch any exceptions during table extraction
            except Exception as e:
                # Print warning message with page number and error
                print(f"Warning: Failed to extract tables on page {pnum}: {e}")
    # Return dictionary of extracted tables organized by page number
    return all_tables
# Define function to parse PDF and extract structured questions
def parse_pdf(pdf_path: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    # Open the PDF document using fitz
    doc = fitz.open(pdf_path)
    # Initialize result dictionary with empty sections list
    result = {"sections": []} if HAS_SECTIONS else {"questions": []}
    # Initialize variable to track current section
    current_section = None
    # Initialize variable to track current question
    current_question = None
    # Initialize flag to track if processing options
    in_options = False
    num_options = 4 if QUESTION_FORMAT == 'mcq' else 2 if QUESTION_FORMAT == 'true_false' else 0
    # Iterate through each page in the PDF
    for page in doc:
        # Collect text and image items from the page
        items = collect_page_items(page)
        # Check if page number is 5 for debugging
        if page.number == 0:
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
            if item["type"] == "image":
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
            # Check if the current item is a table
            if item["type"] == "table":
                # Check if there is an active question being processed
                if current_question:
                    # Check if currently processing options and options list is not empty
                    if in_options and current_question["options"]:
                        # Append table data to the last option of the current question
                        current_question["options"][-1].append({"type": "table", "data": item["data"]})
                    # If not in options mode, append table data to the question body
                    else:
                        current_question["question"].append({"type": "table", "data": item["data"]})
                # If no active question, store table separately
                else:
                    # Add table to result's tables list, initializing list if needed
                    result.setdefault("tables", []).append(item)
                # Increment index to move to the next item
                idx += 1
                # Skip to the next iteration of the loop
                continue
            # Extract text from the item
            text = item["text"]
            # Extract color from the item, default to black
            color = item.get("color", (0, 0, 0))
            # Check if text starts with "section :"
            if any(p.search(text) for p in SECTION_PATTERNS):
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
            # Check if text matches question number pattern
            q_start = match_question_start(text)
            if q_start:
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
                # Add the question start text to question content
                current_question["question"].append({"type": "text", "data": text})
                # Increment index
                idx += 1
                # Collect all items (text, images, tables) until answer pattern is found
                while idx < len(items):
                    next_item = items[idx]
                    # Check if item is text and matches answer pattern
                    if next_item["type"] == "text" and ANS_PATTERN.match(next_item["text"]):
                        # Exit loop if answer pattern is found
                        break 
                    # Handle text item
                    if next_item["type"] == "text":
                        current_question["question"].append({"type": "text", "data": next_item["text"]})
                    # Handle image item
                    elif next_item["type"] == "image":
                        current_question["question"].append({"type": "image", "data": next_item["data"]})
                    # Handle table item
                    elif next_item["type"] == "table":
                        current_question["question"].append({"type": "table", "data": next_item["data"]})
                    idx += 1
                # Continue to next iteration
                continue
            # Check if text matches answer marker
            if ANS_PATTERN.match(text):
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
                    if any(p.search(text) for p in AFTER_OPTION_IGNORE_PATTERNS):
                        # print('Skipping line due to AFTER_OPTION_IGNORE_PATTERNS:', text)
                        idx += 1
                        in_options = False
                        current_question = None
                        continue
                    # Append text to last option with color
                    current_question["options"][-1].append({"type": "text", "data": text, "_color": color})
                # If not in options, append text to question
                else:
                    current_question["question"].append({"type": "text", "data": text})
            # Increment index
            idx += 1
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
    if MERGE_LINE_LENGTH_THRESHOLD is not None:
        result = merge_lines(result, MERGE_LINE_LENGTH_THRESHOLD)
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
                if current_item["type"] == "image" or current_item["type"] == "table":
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
# Main function to handle uploaded PDF and process it
def handle_uploaded_pdf(pdf_file_path: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    # Initialize global settings
    initialize_settings(settings)
    print('global ignore pattern', GLOBAL_IGNORE_PATTERNS)
    global PDF_PATH
    PDF_PATH = pdf_file_path
    # Call the processing function
    result = parse_pdf(pdf_file_path, settings)
    # Open the output JSON file in write mode with UTF-8 encoding
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        # Dump the parsed result dictionary into the JSON file
        json.dump(result, fh, ensure_ascii=False, indent=2)
    # Print confirmation message with file name
    print("saved", OUTPUT_JSON)
    # Return the result
    return result
def handle_test_pdf(pdf_file_path: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    # Initialize global settings
    initialize_settings(settings)
    global PDF_PATH
    PDF_PATH = pdf_file_path
    doc = fitz.open(pdf_file_path)
    num_pages = int(settings['attributes'].get('test_page', 1))
    data = detect_pdf_pattern(doc, num_pages)
    # print('data',data)
    return data
# Function to detect the pdf pattern
def detect_pdf_pattern(doc, num_pages: int) -> Dict[str, Any]:
    # Initialize flags and variables
    has_question = False
    has_option = False
    has_answer = False
    has_section = False
    correct_answer = None
    # Initialize result dictionary
    result = {"sections": []}
    print('-------------result---------------', result)
    current_section = None
    current_question = None
    in_options = False
    # Process specified number of pages (e.g., page 0 if num_pages=1)
    for i in range(min(num_pages, len(doc))):
        # Collect items from the page
        items = collect_page_items(doc[i])
        # print('items', len(items))
        idx = 0
        # Skip header/footer items at the top
        while idx < len(items) and items[idx]["type"] == "text" and is_ignored_text(items[idx]["text"]):
            idx += 1
        # Process remaining items
        while idx < len(items):
            item = items[idx]
            # if item["type"] == "text":
            #     print('next==========item',idx, 'is idx =>', item["text"])
            # Handle section pattern
            if item["type"] == "text" and any(p.search(item["text"]) for p in SECTION_PATTERNS):
                has_section = True
                next_idx = idx + 1
                title = items[next_idx]["text"].strip() if next_idx < len(items) else "Untitled"
                # print('title name is ', title)
                current_section = {"title": title, "questions": []}
                result["sections"].append(current_section)
                idx = next_idx + 1
                continue
            # Handle question start
            q_start = match_question_start(item["text"]) if item["type"] == "text" else None
            if q_start:
                # print('got question start pattern')
                has_question = True
                # Initialize new question
                current_question = {"question": [], "options": [], "correctOption": None}
                if not current_section:
                    current_section = {"title": "Uncategorized", "questions": []}
                    result["sections"].append(current_section)
                current_section["questions"].append(current_question)
                in_options = False
                # Add question start item (text)
                if item["type"] == "text":
                    current_question["question"].append({"type": "text", "data": item["text"]})
                # Collect items until next question start
                idx += 1
                while idx < len(items):
                    next_item = items[idx]
                    # New: Check for section pattern inside question parsing loop
                    if next_item["type"] == "text" and any(p.search(next_item["text"]) for p in SECTION_PATTERNS):
                        has_section = True
                        next_idx = idx + 1
                        new_title = items[next_idx]["text"].strip() if next_idx < len(items) else "Untitled"
                        if result["sections"] and result["sections"][0]["title"] == "Uncategorized":
                            result["sections"][0]["title"] = new_title
                        else:
                            if not result["sections"] or result["sections"][-1]["title"] != new_title:
                                new_section = {
                                    "title": new_title,
                                    "questions": []
                                }
                                result["sections"].append(new_section)
                        idx = next_idx + 1 
                        continue
                    # if next_item["type"] == "text":
                    #     print('next==========item', next_item["text"])
                    # Check for next question start
                    if next_item["type"] == "text" and match_question_start(next_item["text"]):
                        break
                    # Handle answer pattern (text)
                    if next_item["type"] == "text" and ANS_PATTERN.match(next_item["text"]):
                        has_answer = True
                        match = ANS_PATTERN.match(next_item["text"])
                        if match.group(1) and current_question["correctOption"] is None:
                            correct_answer = int(match.group(1))
                            current_question["correctOption"] = correct_answer
                        current_question["question"].append({"type": "text", "data": next_item["text"]})
                        idx += 1
                        in_options = True  
                        continue
                    # Handle option pattern (text or image)
                    opt = OPT_MARK.match(next_item["text"]) if next_item["type"] == "text" else None
                    if opt and (next_item["type"] == "text" or next_item["type"] == "image"):
                        has_option = True
                        in_options = True
                        marker = opt.group(1)
                        opt_idx = get_option_index(marker)
                        while len(current_question["options"]) <= opt_idx:
                            current_question["options"].append([])
                        if next_item["type"] == "text":
                            cleaned = clean_option_text(next_item["text"])
                            current_question["options"][opt_idx].append({"type": "text", "data": cleaned})
                        elif next_item["type"] == "image":
                            current_question["options"][opt_idx].append({"type": "image", "data": next_item["data"]})
                        # Check for correct option via color (only if no explicit answer)
                        color = next_item.get("color", (0, 0, 0))
                        if is_green(color) and current_question["correctOption"] is None:
                            current_question["correctOption"] = opt_idx + 1
                        idx += 1
                        continue
                    # Handle other items (text, image, table for question; text, image for options)
                    if in_options and current_question["options"]:
                        if next_item["type"] == "text" and any(p.search(next_item["text"]) for p in AFTER_OPTION_IGNORE_PATTERNS):
                            idx += 1
                            continue
                        if next_item["type"] in ["text", "image"]:
                            current_question["options"][-1].append({"type": next_item["type"], "data": next_item["text"] if next_item["type"] == "text" else next_item["data"]})
                            # Check for correct answer via color (only if no explicit answer)
                            if is_green(next_item.get("color", (0, 0, 0))) and current_question["correctOption"] is None:
                                current_question["correctOption"] = len(current_question["options"])
                        idx += 1
                        continue
                    # Add to question (text, image, table)
                    if next_item["type"] in ["text", "image", "table"]:
                        current_question["question"].append({"type": next_item["type"], "data": next_item["text"] if next_item["type"] == "text" else next_item["data"]})
                        # Check for correct answer via color (only if no explicit answer)
                        if is_green(next_item.get("color", (0, 0, 0))) and current_question["correctOption"] is None and next_item["type"] != "table":
                            current_question["correctOption"] = 0  # Indicates answer in question
                    # Handle between section
                    # print('section pattern', SECTION_PATTERNS, item["text"])
                    # if next_item["type"] == "text" and any(p.search(next_item["text"]) for p in SECTION_PATTERNS):
                    #     print('---section detected--')

                    #     has_section = True
                    #     next_idx = idx + 1
                    #     if result["sections"][0]["title"] == "Uncategorized":
                    #         new_title = items[next_idx]["text"].strip()
                    #         print('section====is=====uncategorized', new_title)
                    #         result["sections"][0]["title"] = new_title
                    #         print("after sec tittle", result["sections"][0]["title"])
                    #     else:
                    #         print('new section got')
                    #         new_title = items[next_idx]["text"].strip()
                    #         if result["sections"][-1]["title"] != new_title:
                    #             new_section = {
                    #                 "title": new_title,
                    #                 "questions": []
                    #             }
                    #             result["sections"].append(new_section)
                    #             print("Created/*/*/*/*/*/*/*/*/*/*/*/*/*/ new/*/*/*/*/*/*/*/*/*/*/*/*/*/ section:", new_title)
                    idx += 1
                continue
            # Add item to current question if exists
            if current_question:
                if in_options and current_question["options"]:
                    if item["type"] == "text" and any(p.search(item["text"]) for p in AFTER_OPTION_IGNORE_PATTERNS):
                        idx += 1
                        continue
                    if item["type"] in ["text", "image"]:
                        current_question["options"][-1].append({"type": item["type"], "data": item["text"] if item["type"] == "text" else item["data"]})
                        if is_green(item.get("color", (0, 0, 0))) and current_question["correctOption"] is None:
                            current_question["correctOption"] = len(current_question["options"])
                else:
                    if item["type"] in ["text", "image", "table"]:
                        current_question["question"].append({"type": item["type"], "data": item["text"] if item["type"] == "text" else item["data"]})
                        if is_green(item.get("color", (0, 0, 0))) and current_question["correctOption"] is None and item["type"] != "table":
                            current_question["correctOption"] = 0
            idx += 1
    # Clean up temporary fields
    # for section in result.get("sections", []):
    #     for q in section["questions"]:
    #         for opt in q["options"]:
    #             for piece in opt:
    #                 piece.pop("_color", None)
    
    # Merge lines if threshold is set
    # if MERGE_LINE_LENGTH_THRESHOLD is not None:
    #     result = merge_lines(result, MERGE_LINE_LENGTH_THRESHOLD)
    
    # Return structured result with detection flags
    return {
        "pattern_detected": {
            "has_question": has_question,
            "has_option": has_option,
            "has_answer": has_answer,
            "has_section": has_section
        },
        "data": result
    }