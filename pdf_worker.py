
print('Hello World! \n from pdf_worker.py')
import fitz
import os
import shutil

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text += f"\n--- Page {page_num+1} ---\n"
            text += page.get_text()
    return text


def extract_images_from_pdf(pdf_file, output_folder="images"):

    # 🔹 Clear output folder if it exists
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)   
    os.makedirs(output_folder, exist_ok=True)  

    image_count = 0

    with fitz.open(pdf_file) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            images = page.get_images(full=True)   
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


# ---- Run Test ----
pdf_path = "SSC-CGL-Question-Paper-10-September-2024-Shift-3.pdf"

# Extract text
extracted_text = extract_text_from_pdf(pdf_path)
print(extracted_text)

# Extract images
extract_images_from_pdf(pdf_path)

