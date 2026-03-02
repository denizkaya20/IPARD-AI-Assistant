import fitz  # PyMuPDF
from pathlib import Path

# Define the source path for the PDF file
# Note: Keeping the original path as requested
pdf_path = Path("data/raw_pdfs/IPARDIII_T101_S1_v9.0_AKTIF.pdf")


def inspect_pdf_content(file_path, preview_limit=3, char_limit=500):
    """
    Opens a PDF file and prints the total page count along with
    a text preview of the initial pages.
    """
    try:
        # Open the document using PyMuPDF
        with fitz.open(file_path) as doc:
            total_pages = len(doc)
            print(f"Total Pages: {total_pages}")
            print(f"\n--- Text Preview (First {preview_limit} pages) ---\n")

            # Iterate through the document up to the specified preview limit
            for i in range(min(preview_limit, total_pages)):
                page = doc[i]
                text = page.get_text()

                print(f"=== Page {i + 1} ===")
                # Print a snippet of the text to keep the output concise
                print(text[:char_limit].strip())
                print("-" * 20)

    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")


if __name__ == "__main__":
    inspect_pdf_content(pdf_path)