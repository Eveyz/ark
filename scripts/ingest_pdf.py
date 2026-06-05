import glob
import os

import fitz  # PyMuPDF
import lancedb
import pyarrow as pa
from fastembed import TextEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "ark_db.lance")
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "model_cache", "fastembed")

def ingest_pdf(
    input_path: str, db_path: str = DEFAULT_DB_PATH, table_name: str = "survival_guide"
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Path not found: {input_path}")

    pdf_files = []
    if os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        pdf_files = [input_path]
    elif os.path.isdir(input_path):
        pdf_files = glob.glob(os.path.join(input_path, "**/*.pdf"), recursive=True)
        if not pdf_files:
            print(f"No PDF files found in directory: {input_path}")
            return
    else:
        raise ValueError(f"Provided path is neither a PDF file nor a directory containing PDFs: {input_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )

    chunks_data = []
    total_pages_parsed = 0

    print(f"Found {len(pdf_files)} PDF(s). Extracting and splitting text...")
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file}")
        doc = fitz.open(pdf_file)

        for page_num in tqdm(range(len(doc)), desc=f"Pages in {os.path.basename(pdf_file)}"):
            page = doc.load_page(page_num)
            text = page.get_text("text")

            # Clean text: remove invalid newlines and extra spaces
            text = text.replace("\n", " ").strip()
            text = " ".join(text.split())

            if not text:
                continue

            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                chunks_data.append(
                    {
                        "page": page_num + 1,  # 1-based indexing for better readability
                        "text": chunk,
                    }
                )

        total_pages_parsed += len(doc)

    num_chunks = len(chunks_data)
    print(
        f"\n✅ Extraction complete: 共计 {total_pages_parsed} 页解析完毕，生成了 {num_chunks} 个 Chunk。"
    )

    if num_chunks == 0:
        print("No text extracted. Exiting.")
        return

    print(f"\nInitializing embedding model (BAAI/bge-small-en-v1.5) with cache: {DEFAULT_CACHE_DIR}...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=DEFAULT_CACHE_DIR)

    texts_to_embed = [item["text"] for item in chunks_data]
    print("Generating embeddings...")
    # fastembed.embed returns a generator
    embeddings_generator = embedding_model.embed(texts_to_embed)
    embeddings = list(
        tqdm(embeddings_generator, total=len(texts_to_embed), desc="Embedding Chunks")
    )

    # Assemble Data for PyArrow Table
    ids = list(range(1, num_chunks + 1))
    pages = [item["page"] for item in chunks_data]
    texts = [item["text"] for item in chunks_data]

    # Define Schema strictly matching downstream Rust requirements
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("page", pa.int32()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 384)),
        ]
    )

    # Create Arrow Table
    table = pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.int32()),
            pa.array(pages, type=pa.int32()),
            pa.array(texts, type=pa.string()),
            pa.array(embeddings, type=pa.list_(pa.float32(), 384)),
        ],
        schema=schema,
    )

    print(f"\nConnecting to LanceDB at {db_path}...")
    db = lancedb.connect(db_path)

    print(f"Writing data to table '{table_name}' (Overwrite mode)...")
    db.create_table(table_name, data=table, schema=schema, mode="overwrite")

    print("\n✅ 数据已成功写入 LanceDB！")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDF into LanceDB.")
    parser.add_argument(
        "input_path", type=str, help="Path to a PDF file or a folder containing PDF files"
    )
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH, help="LanceDB path")
    args = parser.parse_args()

    ingest_pdf(args.input_path, db_path=args.db)
