import os
import glob
import re

def clean_markdown_files():
    md_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "markdown")
    clean_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "markdown_clean")
    os.makedirs(clean_dir, exist_ok=True)
    md_files = glob.glob(os.path.join(md_dir, "*.md"))
    
    if not md_files:
        print("No Markdown files found. Have you run pdf_to_md.py yet?")
        return

    print(f"🧹 Found {len(md_files)} markdown files to clean.")
    
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        original_len = len(content)
        
        # 1. Clean Table of Contents lines with dots and page numbers
        # Matches: |Skylines ................................................................................ 28|
        # Or: Chapter 1 ......... 12
        content = re.sub(r'\|?.*\.{5,}\s*\d+\s*\|?\n', '', content)
        
        # 2. Clean standalone page numbers or Roman numerals
        # Matches: 142, iv, ix, Page 12 of 233
        content = re.sub(r'(?i)^\s*\|?(page\s+\d+\s+of\s+\d+|[ivxlcdm]+|\d+)\|?\s*$\n', '', content, flags=re.MULTILINE)
        
        # 3. Clean specific recurring headers/footers in military manuals
        # Matches: B-GL-392-009/FP-000, FM 21-76, etc.
        content = re.sub(r'(?i)^\s*(B-GL-\d+.*|FM \d+-\d+.*|ATP \d+-\d+.*)\s*$\n', '', content, flags=re.MULTILINE)
        
        # 4. Clean empty table blocks
        content = re.sub(r'^\|\s*\|$\n', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\|---\|$\n', '', content, flags=re.MULTILINE)
        
        # 5. Condense excessive empty lines into a single blank line
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        new_len = len(content)
        removed_chars = original_len - new_len
        
        with open(os.path.join(clean_dir, os.path.basename(md_file)), "w", encoding="utf-8") as f:
            f.write(content.strip())
            
        print(f"✅ Cleaned {os.path.basename(md_file)}: Removed {removed_chars} characters.")

if __name__ == "__main__":
    clean_markdown_files()
