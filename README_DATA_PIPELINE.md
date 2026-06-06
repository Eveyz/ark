# Ark-1 知识库数据流水线指南 (Data Pipeline Guide)

为了保证 RAG 检索的极高质量，我们不直接将生硬的 PDF 喂给向量数据库，而是采用**“清洗、结构化、灌库”三步走的 ETL 流水线**。

当你以后下载了新的生存手册 PDF，并希望把它加入到你的知识库中时，请按照以下三个步骤操作。

## 前置环境准备
确保你在终端中激活了 Python 环境：
```bash
conda activate google-adk
```

---

## 步骤 1：PDF 转 Markdown (`pdf_to_md.py`)
**用途**：利用 `pymupdf4llm` 的版面分析能力，将 PDF 转换为带有 `##` 标题和列表的 Markdown 文件，最大程度保留原本的文档结构，同时剥离无效的排版格式。
**操作**：
1. 把下载的 PDF 放入 `data/` 目录。
2. 运行转换脚本：
```bash
python scripts/pdf_to_md.py
```
**结果**：脚本会自动在 `data/markdown/` 下生成同名的 `.md` 文件。

---

## 步骤 2：二次清洗脏数据 (`clean_markdown.py`)
**用途**：军用或扫描版的 PDF 在转换后往往会留下大量无关的 OCR 脏数据（例如目录的 `... 28`、页码 `142`、页眉 `FM 21-76` 以及无意义的空表格和过长的空行）。这个脚本通过正则极速清洗这些内容，帮你省下大量无关 Token，大幅提升 Embedding 的精确度。
**操作**：
```bash
python scripts/clean_markdown.py
```
**结果**：脚本会将清洗后的“纯净版” Markdown 文件保存到全新的 `data/markdown_clean/` 目录下，不会破坏原始的转换数据。

---

## 步骤 3：结构化切分与灌库 (`ingest_markdown.py`)
**用途**：将清洗干净的 Markdown 文本进行结构化入库。
- 使用 `MarkdownHeaderTextSplitter` 按照 `h1`, `h2` 提取元数据。
- 把元数据强行前置注入到每一个 Text Chunk 中（例如 `Source: 某某手册, Chapter: 第三章`），让语义向量更饱满。
- 调用 `BAAI/bge-small-en-v1.5` 生成向量。
- 存入本地的 LanceDB 数据库 `ark_db.lance` 的 `survival_guide_md` 表中。
**操作**：
```bash
python scripts/ingest_markdown.py
```

## 最后
启动 Rust 系统！Agent 会自动拥有挂载了分类层级的 `Check_Index` 和更精准的检索能力。
```bash
cargo run
```

pip install pymupdf langchain-text-splitters fastembed lancedb pyarrow tqdm