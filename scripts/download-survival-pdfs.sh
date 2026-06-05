#!/bin/bash
# Survival Pod - 核心知识库 PDF 批量下载脚本
# 使用 curl（系统自带，无需安装 wget）
# 在 Orange Pi 5 / Linux / macOS 上直接运行
# 总大小约 500MB，建议至少 2GB 可用空间

set -e

BASE_DIR="${1:-./survival-knowledge}"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

echo "=== Survival Pod Knowledge Base Downloader ==="
echo "Target: $BASE_DIR"
echo ""

# curl helper: -L 跟踪重定向, -f 失败时返回非0, -# 显示进度条
download() {
  local url="$1"
  local filename="$2"
  echo "  → $filename"
  curl -L -f -# -o "$filename" "$url" 2>&1 || echo "  ⚠  FAILED: $filename"
}

# ---------- 综合基础手册 (必下) ----------
echo "[1/7] Downloading core survival manuals..."

download "https://trueprepper.com/wp-content/uploads/2022/11/FM-21-76-US-Army-Survival-Manual.pdf" \
  "FM_21-76_US_Army_Survival_Manual.pdf"

download "https://seasonedcitizenprepper.com/wp-content/uploads/2013/02/FM-21-76-1-Survival-Evasion-Recovery.pdf" \
  "FM_21-76-1_Survival_Evasion_Recovery.pdf"

download "https://trueprepper.com/wp-content/uploads/ATP-3-50.21-Survival.pdf" \
  "ATP_3-50.21_Survival.pdf"

download "http://www.landsurvival.com/downloads/SAS%20Survival%20Handbook.pdf" \
  "SAS_Survival_Handbook.pdf"

download "https://trueprepper.com/wp-content/uploads/Canadian-Military-Fieldcraft.pdf" \
  "Canadian_Military_Fieldcraft.pdf"

# ---------- 1. 急救与创伤 ----------
echo "[2/7] Downloading medical & first aid..."

download "https://trueprepper.com/wp-content/uploads/FM-4-25-11-First_Aid.pdf" \
  "FM_4-25.11_First_Aid.pdf"

download "https://trueprepper.com/wp-content/uploads/2022/12/ST-31-91B-US-Army-Special-Forces-Medical-Handbook.pdf" \
  "ST_31-91B_SF_Medical_Handbook.pdf"

download "https://trueprepper.com/wp-content/uploads/2022/11/Where-There-is-no-Doctor-a-Village-Health-Care-Handbook.pdf" \
  "Where_There_Is_No_Doctor.pdf"

download "https://seasonedcitizenprepper.com/wp-content/uploads/2013/01/Survival-and-Austere-Medicine.pdf" \
  "Survival_Austere_Medicine.pdf"

# ---------- 3. 水源净化 ----------
echo "[3/7] Downloading water purification..."

download "https://seasonedcitizenprepper.com/wp-content/uploads/2014/02/SODIS-manual.pdf" \
  "SODIS_Safe_Water_Manual.pdf"

# ---------- 4. 野外生火 ----------
echo "[4/7] Downloading fire craft..."

download "https://trueprepper.com/wp-content/uploads/2022/12/FM-21-76-Appendix-F-Fire.pdf" \
  "FM_21-76_Appendix_Fire.pdf"

# ---------- 6. 可食用植物 ----------
echo "[5/7] Downloading plant identification..."

download "https://docs.google.com/open?id=0B6GE42-kvADvMjJhNmM4ODEtNmU3Yy00OTJkLWFkMTAtZTU1NzQ2MmE4ZmI1" \
  "Edible_Wild_Plants_Guide.pdf"

# ---------- 8. 狩猎与陷阱 ----------
download "https://ia801406.us.archive.org/13/items/deadfallssnaresb00harduoft/deadfallssnaresb00harduoft.pdf" \
  "Deadfalls_and_Snares.pdf"

# ---------- 10. 极端环境 ----------
echo "[6/7] Downloading extreme environment..."

download "https://trueprepper.com/wp-content/uploads/2022/12/FM-31-70-Basic-Cold-Weather-Manual.pdf" \
  "FM_31-70_Basic_Cold_Weather.pdf"

download "https://trueprepper.com/wp-content/uploads/USMC-Winter-Survival-Course-Handbook.pdf" \
  "USMC_Winter_Survival.pdf"

download "https://trueprepper.com/wp-content/uploads/Canadian-Military-Basic-Cold-Weather-Training.pdf" \
  "Canadian_Cold_Weather.pdf"

# ---------- 11. 绳索与绳结 ----------
download "https://docs.google.com/open?id=0B6GE42-kvADvNjZmMDY0NGItZThhZS00OTcyLWJlYTktMjIxYjg2ZDc2NTQ4" \
  "Handbook_of_Knots_Splices.pdf"

# ---------- 12. 信号与救援 ----------
echo "[7/7] Downloading signaling & rescue..."

download "https://trueprepper.com/wp-content/uploads/FM-21-60-Visual-Signals.pdf" \
  "FM_21-60_Visual_Signals.pdf"

# ---------- 统计 ----------
echo ""
echo "=== Download Complete ==="
echo "Files in $BASE_DIR:"
ls -lh "$BASE_DIR"/*.pdf 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
echo ""
echo "Total size:"
du -sh "$BASE_DIR" 2>/dev/null
echo ""
echo "Next steps:"
echo "  1. Clone survivalRAG: git clone https://github.com/bdkoeh/survivalRAG.git"
echo "  2. Extract text from PDFs with Docling or PyMuPDF"
echo "  3. Build ChromaDB vector store"
echo "  4. Deploy on Orange Pi 5 with rkllm-llama.cpp + RAG pipeline"
