import shutil
import os
import pikepdf
from pikepdf import PdfImage


def extract_images_from_pdf(pdf_path, output_dir="images_pikepdf"):
   
    try:
         # If directory exists, clear it
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        pdf = pikepdf.open(pdf_path)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        image_count = 0
        for page_num, page in enumerate(pdf.pages, start=1):
            for name, raw_image in page.images.items():
                try:
                    pdf_image = PdfImage(raw_image)

                    filter_type = raw_image.get("/Filter", None)
                    if filter_type == "/DCTDecode":
                        ext = "jpg"
                    elif filter_type == "/JPXDecode":
                        ext = "jp2"
                    else:
                        ext = "png"

                    output_filename = os.path.join(
                        output_dir, f"page{page_num}_img{image_count}.{ext}"
                    )

                    pdf_image.extract_to(fileprefix=output_filename[:-4])
                    print(f"Extracted: {output_filename}")

                    image_count += 1
                except Exception as e:
                    print(f"Error extracting image on page {page_num}: {e}")

        pdf.close()

        if image_count == 0:
            print("No images found in PDF.")
        else:
            print(f"Done! Extracted {image_count} images into '{output_dir}/'")

    except Exception as e:
        print(f"Error opening or processing PDF: {e}")


# Call the function
extract_images_from_pdf("abhishek_bsb-udn.pdf")


