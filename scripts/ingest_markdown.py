import glob
import os
import lancedb
import pyarrow as pa
from fastembed import TextEmbedding
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "ark_db.lance")
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "model_cache", "fastembed")
MD_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "markdown_clean")
TABLE_NAME = "survival_guide_md"

def ingest_markdown():
    if not os.path.exists(MD_DATA_DIR):
        print(f"Markdown directory not found: {MD_DATA_DIR}")
        return

    md_files = glob.glob(os.path.join(MD_DATA_DIR, "*.md"))
    if not md_files:
        print("No markdown files found to ingest.")
        return

    print(f"Found {len(md_files)} Markdown file(s).")

    # Define how to split headers
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    
    # Fallback character splitter for very long sections under a single header
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    chunks_data = []
    total_docs = 0

    for md_file in tqdm(md_files, desc="Parsing Markdown Files"):
        source_name = os.path.basename(md_file).replace(".md", "")
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        if not content.strip():
            continue

        # Split by markdown headers
        md_header_splits = markdown_splitter.split_text(content)
        
        # Further split long sections
        final_splits = char_splitter.split_documents(md_header_splits)

        for split in final_splits:
            text = split.page_content.strip()
            if not text:
                continue
            
            # Extract metadata
            metadata = split.metadata
            h1 = metadata.get("h1", "")
            h2 = metadata.get("h2", "")
            if not h1 and not h2:
                h3 = metadata.get("h3", "")
                h2 = h3 # Fallback if h2 is missing but h3 exists

            # Add structured context to the text itself to help the embedding model
            enriched_text = f"Source: {source_name}\n"
            if h1: enriched_text += f"Chapter: {h1}\n"
            if h2: enriched_text += f"Section: {h2}\n"
            enriched_text += f"\n{text}"

            chunks_data.append({
                "source": source_name,
                "h1": h1,
                "h2": h2,
                "text": enriched_text,
            })
            
        total_docs += 1

    num_chunks = len(chunks_data)
    print(f"\n✅ Extraction complete: {total_docs} files parsed into {num_chunks} structured chunks.")

    if num_chunks == 0:
        return

    print(f"\nInitializing embedding model (BAAI/bge-small-en-v1.5)...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=DEFAULT_CACHE_DIR)

    texts_to_embed = [item["text"] for item in chunks_data]
    print("Generating embeddings...")
    embeddings_generator = embedding_model.embed(texts_to_embed)
    embeddings = list(tqdm(embeddings_generator, total=len(texts_to_embed), desc="Embedding Chunks"))

    # Assemble PyArrow Data
    ids = list(range(1, num_chunks + 1))
    sources = [item["source"] for item in chunks_data]
    h1s = [item["h1"] for item in chunks_data]
    h2s = [item["h2"] for item in chunks_data]
    texts = [item["text"] for item in chunks_data]

    schema = pa.schema([
        pa.field("id", pa.int32()),
        pa.field("source", pa.string()),
        pa.field("h1", pa.string()),
        pa.field("h2", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 384)),
    ])

    data = [
        pa.array(ids, type=pa.int32()),
        pa.array(sources, type=pa.string()),
        pa.array(h1s, type=pa.string()),
        pa.array(h2s, type=pa.string()),
        pa.array(texts, type=pa.string()),
        pa.array(embeddings, type=pa.list_(pa.float32(), 384)),
    ]
    batch = pa.RecordBatch.from_arrays(data, schema=schema)

    # Save to LanceDB
    print(f"\nConnecting to LanceDB at {DEFAULT_DB_PATH}...")
    db = lancedb.connect(DEFAULT_DB_PATH)
    
    if TABLE_NAME in db.table_names():
        print(f"Table '{TABLE_NAME}' already exists. Dropping and recreating...")
        db.drop_table(TABLE_NAME)
        
    table = db.create_table(TABLE_NAME, data=batch)
    print(f"✅ Successfully created table '{TABLE_NAME}' with {num_chunks} rows.")
    print("Now you can update your Rust backend to query 'survival_guide_md'!")

if __name__ == "__main__":
    ingest_markdown()
