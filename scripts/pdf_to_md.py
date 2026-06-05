import os
import glob
import pymupdf4llm
from tqdm import tqdm

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    out_dir = os.path.join(data_dir, "markdown")
    os.makedirs(out_dir, exist_ok=True)

    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    print(f"Found {len(pdf_files)} PDFs.")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        md_filename = filename.replace(".pdf", ".md")
        out_path = os.path.join(out_dir, md_filename)

        if os.path.exists(out_path):
            print(f"Skipping {filename}, already exists.")
            continue

        print(f"Converting {filename} to Markdown...")
        try:
            md_text = pymupdf4llm.to_markdown(pdf_path)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md_text)
            print(f"✅ Saved to {out_path}")
        except Exception as e:
            print(f"❌ Failed to convert {filename}: {e}")

if __name__ == "__main__":
    main()
