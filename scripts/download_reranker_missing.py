import urllib.request
import os

target_dir = os.path.expanduser(
    "~/Documents/projects/rust-projects/ark-1/model_cache/fastembed/models--BAAI--bge-reranker-base/snapshots/main"
)
os.makedirs(target_dir, exist_ok=True)

base_url = "https://huggingface.co/BAAI/bge-reranker-base/resolve/main/"
files = ["special_tokens_map.json", "tokenizer_config.json"]

print("下载 BGE-Reranker-Base 缺失文件...")
for f in files:
    out_path = os.path.join(target_dir, f)
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"  ✅ {f} 已存在 ({size:,} bytes)，跳过")
        continue
    url = base_url + f
    print(f"  ⏳ 下载 {f} ...")
    try:
        urllib.request.urlretrieve(url, out_path)
        size = os.path.getsize(out_path)
        print(f"  ✅ {f} 下载完成 ({size:,} bytes)")
    except Exception as e:
        print(f"  ❌ {f} 下载失败: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)

print("\n完成。再次运行: cargo run")
