from pathlib import Path
from collections import Counter

# --- Configuration ---
pdf_dir = Path('data/unique_pdfs')
doc_types_counter = Counter()

# --- Classification Loop ---
for file in pdf_dir.glob('*.pdf'):
    # Extract filename without extension
    filename_stem = file.stem

    # Classify documents based on naming conventions
    # Example format: IPARDIII_T101_S1_basvuru_paketi_hazirlama_00
    if 'hazirlama' in filename_stem:
        doc_types_counter['application_package_preparation'] += 1
    elif 'basvuru_paketi' in filename_stem:
        doc_types_counter['application_package'] += 1
    elif 'bilgilendirme' in filename_stem:
        doc_types_counter['information_notes'] += 1
    elif 'AKTIF' in filename_stem or 'ARSIV' in filename_stem:
        doc_types_counter['application_call_guide'] += 1
    else:
        doc_types_counter['other'] += 1

# --- Reporting Statistics ---
print("Document Type Statistics:")
for doc_type, count in sorted(doc_types_counter.items()):
    print(f'  {doc_type:35}: {count}')

print("-" * 40)
print(f'  TOTAL DOCUMENTS: {sum(doc_types_counter.values())}')