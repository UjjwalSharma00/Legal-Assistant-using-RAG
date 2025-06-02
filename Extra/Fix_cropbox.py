import sys
import os
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import NameObject

# Always use the preloaded_pdfs/merged.pdf file
pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'preloaded_pdfs', 'merged.pdf')
print(f"Checking and fixing file: {os.path.abspath(pdf_path)}")

reader = PdfReader(pdf_path)
writer = PdfWriter()

fixed = False
for i, page in enumerate(reader.pages):
    # If /CropBox is missing, set it to /MediaBox
    if NameObject("/CropBox") not in page:
        print(f"Page {i+1}: CropBox missing, setting to MediaBox.")
        page[NameObject("/CropBox")] = page[NameObject("/MediaBox")]
        fixed = True
    else:
        print(f"Page {i+1}: CropBox present.")
    writer.add_page(page)

if fixed:
    out_path = os.path.splitext(pdf_path)[0] + "_fixed.pdf"
    with open(out_path, "wb") as f:
        writer.write(f)
    print(f"Fixed PDF saved as: {out_path}")
else:
    print("No missing CropBox found. No new file created.")
print("Done.")
