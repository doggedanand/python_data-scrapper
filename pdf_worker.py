import pymupdf
import os
import shutil
import pdfplumber
# def extract_text_from_pdf(pdf_file):
#     text = ""
#     with pymupdf.open(pdf_file) as pdf:
#         for page_num in range(len(pdf)):
#             page = pdf[page_num]
#             text += f"\n--- Page {page_num+1} ---\n"
#             blocks = page.get_text("blocks")
            
#             for b in blocks:
#                 text += b[4]
#                 text += "\n"
#     return text


def extract_images_from_pdf(pdf_file, output_folder="images"):

    # 🔹 Clear output folder if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)   
    os.makedirs(output_folder, exist_ok=True)  

    image_count = 0

    with pymupdf.open(pdf_file) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            images = page.get_images(full=False)   
            for img_index, img in enumerate(images):
                xref = img[0]   
                pix = pdf.extract_image(xref)
                img_bytes = pix["image"]
                img_ext = pix["ext"]

                img_filename = os.path.join(output_folder, f"page{page_num+1}_img{img_index+1}.{img_ext}")
                with open(img_filename, "wb") as img_file:
                    img_file.write(img_bytes)

                image_count += 1
                print(f"Saved image: {img_filename}")

    if image_count == 0:
        print("No images found in PDF.")
    else:
        print(f"Extracted {image_count} images into '{output_folder}/'")


def extract_images_with_coordinates(pdf_file, output_folder="images_with_coords"):
    # Clear/create folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    image_count = 0

    with pymupdf.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            # Get all embedded images
            images = page.get_images(full=True)
            for img_index, img in enumerate(images, start=1):
                xref = img[0]
                # Extract image
                pix = pdf.extract_image(xref)
                img_bytes = pix["image"]
                img_ext = pix["ext"]

                # Image coordinates on page
                try:
                    rect = page.get_image_bbox(xref)  # Returns fitz.Rect
                    print(f"Page {page_num}, Image {img_index} coordinates: {rect}")
                except:
                    rect = None
                    print(f"Page {page_num}, Image {img_index} coordinates not found")

                # Save original image
                img_filename = os.path.join(output_folder, f"page{page_num}_img{img_index}.{img_ext}")
                with open(img_filename, "wb") as f:
                    f.write(img_bytes)
                
                image_count += 1
                print(f"Saved image: {img_filename}")

                # Optional: save cropped version using coordinates
                if rect:
                    pixmap = page.get_pixmap(clip=rect)
                    cropped_filename = os.path.join(output_folder, f"page{page_num}_img{img_index}_cropped.png")
                    pixmap.save(cropped_filename)
                    print(f"Saved cropped image: {cropped_filename}")

    if image_count == 0:
        print("No images found in PDF.")
    else:
        print(f"Extracted {image_count} images into '{output_folder}/'")

# # ---- Run Test ----
pdf_path = "SSC-CGL-Tier-1-Question-Paper-10-September-2024-Shift-1.pdf"

# # Extract text
# # extracted_text = extract_text_from_pdf(pdf_path)
# # print(extracted_text)

# # Extract images
# extract_images_from_pdf(pdf_path)

# # Extract images with coordinates
# # extract_images_with_coordinates(pdf_path)

def extract_images_with_pdfplumber(pdf_file, output_folder="images_pdfplumber"):
    # Clear/create output folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    image_count = 0

    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Access page's images
            images = page.images
            for img_index, img in enumerate(images, start=1):
                # The image object contains x0, y0, x1, y1 and raw stream
                if "stream" in img:
                    # Save the image bytes
                    img_bytes = img["stream"].get_data()
                    img_filename = os.path.join(output_folder, f"page{page_num}_img{img_index}.jpg")
                    with open(img_filename, "wb") as f:
                        f.write(img_bytes)
                    image_count += 1
                    print(f"Saved image: {img_filename}")

    if image_count == 0:
        print("No images found in PDF.")
    else:
        print(f"Extracted {image_count} images into '{output_folder}/'")


# # extract_images_with_pdfplumber(pdf_path)


import fitz, os, shutil
from PIL import Image
import io

def extract_images_with_smask(pdf_file, output_folder="pdf_images"):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    doc = fitz.open(pdf_file)
    image_count = 0

    for page_num, page in enumerate(doc, start=1):
        print(f"\n--- Processing Page {page_num} ---")

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            print('xref', xref);
            base_image = doc.extract_image(xref)

            if not base_image:
                continue

            img_bytes = base_image["image"]
            img_ext = base_image["ext"]

            # check if image has smask (soft mask / alpha channel)
            smask = img[1] if len(img) > 1 else None
            if smask:
                try:
                    smask_image = doc.extract_image(smask)
                    if smask_image:
                        # merge mask with main image
                        base_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        mask_img = Image.open(io.BytesIO(smask_image["image"])).convert("L")
                        base_img.putalpha(mask_img)
                        img_ext = "png" 
                        img_bytes = io.BytesIO()
                        base_img.save(img_bytes, format="PNG")
                        img_bytes = img_bytes.getvalue()
                        print(f"🎭 Applied smask to image {img_index+1}")
                except Exception as e:
                    print(f"Error applying smask: {e}")

            img_path = os.path.join(output_folder, f"page{page_num}_img{img_index+1}.{img_ext}")
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            print(f"Saved image: {img_path}")
            image_count += 1

    print(f"\nTotal images extracted: {image_count}")


# 🔹 Example usage
pdf_path = "SSC-CGL-Tier-1-Question-Paper-9-September-2024-Shift-1.pdf"
# extract_images_with_smask(pdf_path, output_folder="pdf_images")





import camelot, os

def extract_tables(pdf_file, output_folder="pdf_tables"):
    # Try with stream first (less false positives)
    tables = camelot.read_pdf(pdf_file, pages="all", flavor="lattice")

    print(f"🔍 Total raw tables detected: {len(tables)}")

    # Filter out noise (keep only real tables)
    valid_tables = [t for t in tables if t.shape[1] > 1 and t.shape[0] > 2]

    print(f"✅ After filtering, tables kept: {len(valid_tables)}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for i, table in enumerate(valid_tables):
        csv_path = f"{output_folder}/table_{i+1}.csv"
        table.to_csv(csv_path)
        print(f"📄 Saved table {i+1} → {csv_path}")

 
# extract_tables(pdf_path, output_folder="pdf_tables")