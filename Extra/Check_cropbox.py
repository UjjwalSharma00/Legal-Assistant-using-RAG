import sys
import pdfplumber
import os

if len(sys.argv) != 2:
    print("Usage: python check_cropbox.py <pdf_path>")
    print("Example:")
    print("  python check_cropbox.py \"Z:/OneDrive/Desktop/new-code/preloaded_pdfs/1_113_1_The_Patents_Act__1970___incorporating_all_amendments_till_1-08-2024.pdf\"")
    sys.exit(1)

pdf_path = sys.argv[1]
print(f"Checking file: {os.path.abspath(pdf_path)}")

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        if page.cropbox is None:
            print(f"Page {i+1}: CropBox missing, setting to MediaBox.")
            page.cropbox = page.mediabox
        else:
            print(f"Page {i+1}: CropBox present.")
print("Done checking CropBox for all pages.")
