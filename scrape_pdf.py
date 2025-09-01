import fitz
import base64
import json
import re
import statistics
from typing import Any, Dict, List, Tuple, Optional

PDF_PATH = "SSC-CGL-Question-Paper-10-September-2024-Shift-3_removed.pdf"
OUTPUT_JSON = "output.json"

ICON_MAX_WH = 40  
Q_START = re.compile(r"^\s*Q\.\s*(\d+)", re.IGNORECASE)
SECTION_FULL = re.compile(r"^\s*Section\s*[:\.]\s*(.+)$", re.IGNORECASE)
SECTION_PREFIX = re.compile(r"^\s*Section\s*[:\.]?\s*$", re.IGNORECASE)
ANS = re.compile(r"^\s*Ans[\s\.:]*([1-9])?(.*)$", re.IGNORECASE)
OPT_MARK = re.compile(r"^\s*([1-4a-dA-D])[\.\)]\s*", re.IGNORECASE)

HEADER_FOOTER_PATTERNS = [
    re.compile(r"SSC", re.IGNORECASE),
    re.compile(r"Tier", re.IGNORECASE),
    re.compile(r"Shift", re.IGNORECASE),
    re.compile(r"Page \d+", re.IGNORECASE),
]


def is_header_footer(text: str) -> bool:
    return any(p.search(text) for p in HEADER_FOOTER_PATTERNS)


def to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


def parse_color(col: Any) -> Tuple[int, int, int]:
    if isinstance(col, int):
        return ((col >> 16) & 255, (col >> 8) & 255, col & 255)
    if isinstance(col, (list, tuple)):
        vals = list(col)
        if all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in vals[:3]):
            return tuple(int(v * 255) for v in vals[:3])
        return tuple(int(vals[i]) for i in range(3))
    if isinstance(col, float):
        v = int(col * 255)
        return (v, v, v)
    return (0, 0, 0)


def is_red(rgb: Tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return (r > 150) and (r >= g + 30) and (r >= b + 30)


def is_green(rgb: Tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return (g > 150) and (g >= r + 30) and (g >= b + 30)


def collect_page_items(page) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
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
                items.append({"type": "text", "text": text_stripped, "bbox": bbox, "color": color})

    for img in page.get_images(full=True):
        xref = img[0]
        info = page.parent.extract_image(xref)
        w, h = info.get("width", 0), info.get("height", 0)
        # skip tiny icons (tick/cross)
        if w < ICON_MAX_WH and h < ICON_MAX_WH:
            continue
        img_bytes = info["image"]
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        r = rects[0]
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
    items.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))
    return items


def clean_option_text(text: str) -> str:
    return OPT_MARK.sub("", text).strip()


def get_option_index(marker: str) -> int:
    if marker.isdigit():
        return int(marker) - 1
    else:
        return ord(marker.lower()) - ord("a")


def parse_pdf(pdf_path: str) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    result = {"sections": []}
    current_section: Optional[Dict[str, Any]] = None
    all_items: List[Dict[str, Any]] = []
    if doc.page_count > 0:
        page_height = doc[0].bound()[3]
        for pagenum, page in enumerate(doc):
            page_items = collect_page_items(page)
            offset = pagenum * page_height
            for it in page_items:
                b = it["bbox"]
                it["bbox"] = (b[0], b[1] + offset, b[2], b[3] + offset)
            all_items.extend(page_items)

    all_items.sort(key=lambda it: (it["bbox"][1], it["bbox"][0]))

    heights = [(it["bbox"][3] - it["bbox"][1]) for it in all_items if it["type"] == "text"]
    median_height = statistics.median(heights) if heights else 12.0
    paragraph_threshold = median_height * 1.6

    i = 0
    N = len(all_items)

    while i < N:
        it = all_items[i]
        if it["type"] != "text":
            i += 1
            continue
        text = it["text"]

        if is_header_footer(text):
            i += 1
            continue

        mfull = SECTION_FULL.match(text)
        if mfull:
            title = mfull.group(1).strip()
            if result["sections"] and result["sections"][-1]["title"] == title:
                current_section = result["sections"][-1]
            else:
                current_section = {"title": title, "questions": []}
                result["sections"].append(current_section)
            i += 1
            continue

        mpref = SECTION_PREFIX.match(text)
        if mpref:
            j = i + 1
            title_parts: List[str] = []
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

        mq = Q_START.match(text)
        if mq:
            if current_section is None:
                current_section = {"title": "Uncategorized", "questions": []}
                result["sections"].append(current_section)
            question: Dict[str, Any] = {"question": [], "options": [], "correctOption": None}
            current_section["questions"].append(question)
            start = i + 1
            j = start
            while j < N:
                nxt = all_items[j]
                if nxt["type"] == "text" and (
                    SECTION_FULL.match(nxt["text"]) or SECTION_PREFIX.match(nxt["text"]) or Q_START.match(nxt["text"])
                ):
                    break
                j += 1

            q_items = []
            for k in range(start, j):
                entry = dict(all_items[k])
                entry["_assigned"] = False
                entry["_assigned_to"] = None 
                q_items.append(entry)
            q_items = [itq for itq in q_items if not (itq["type"] == "text" and ANS.match(itq["text"]))]
            q_items = [itq for itq in q_items if not (itq["type"] == "text" and is_header_footer(itq["text"]))]
            text_items = [itq for itq in q_items if itq["type"] == "text"]
            image_items = [itq for itq in q_items if itq["type"] == "image"]
            label_map: Dict[int, Dict[str, Any]] = {}
            for t in text_items:
                m = OPT_MARK.match(t["text"])
                if m:
                    idx = get_option_index(m.group(1))
                    label_map[idx] = t
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
            labels_sorted = sorted(label_map.items(), key=lambda kv: kv[1]["bbox"][1])  
            first_label_top = labels_sorted[0][1]["bbox"][1]
            first_label_bottom = labels_sorted[0][1]["bbox"][3]
            above_label_texts = [t for t in text_items if t["bbox"][1] < first_label_top and not OPT_MARK.match(t["text"])]
            last_text_bottom = max((t["bbox"][3] for t in above_label_texts), default=None)
            bands: Dict[int, Tuple[float, float]] = {}
            for idx_pos, (opt_idx, label_item) in enumerate(labels_sorted):
                top = label_item["bbox"][1] - paragraph_threshold * 0.5
                if idx_pos + 1 < len(labels_sorted):
                    next_top = labels_sorted[idx_pos + 1][1]["bbox"][1]
                    bottom = (label_item["bbox"][1] + next_top) / 2.0
                else:
                    bottom = label_item["bbox"][3] + paragraph_threshold * 2.0
                bands[opt_idx] = (top, bottom)
            max_label_idx = max(label_map.keys())
            options: List[List[Dict[str, Any]]] = [[] for _ in range(max_label_idx + 1)]
            for opt_idx, label_item in label_map.items():
                label_item["_assigned"] = True
                label_item["_assigned_to"] = opt_idx
                cleaned = clean_option_text(label_item["text"])
                if cleaned:
                    options[opt_idx].append({"type": "text", "data": cleaned, "_color": label_item.get("color")})
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
            has_text_counts = sum(1 for opt in options if any(p.get("type") == "text" for p in opt))
            has_image_counts = sum(1 for opt in options if any(p.get("type") == "image" for p in opt))
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
            buf_lines: List[str] = []
            last_bottom = None
            def flush_buf():
                nonlocal buf_lines
                if buf_lines:
                    question["question"].append({"type": "text", "data": "\n".join(buf_lines)})
                    buf_lines = []

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
    return result


if __name__ == "__main__":
    parsed = parse_pdf(PDF_PATH)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(parsed, fh, ensure_ascii=False, indent=2)
    print("saved", OUTPUT_JSON)
