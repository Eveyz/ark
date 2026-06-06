import urllib.request
import os
import json

target_dir = os.path.expanduser(
    "~/Documents/projects/rust-projects/ark-1/model_cache/fastembed/models--rozgo--bge-reranker-v2-m3/snapshots/main"
)
os.makedirs(target_dir, exist_ok=True)

base_url = "https://huggingface.co/rozgo/bge-reranker-v2-m3/resolve/main/"
files = ["model.onnx", "config.json", "tokenizer.json"]

print("开始下载 bge-reranker-v2-m3 模型文件...")
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

# 创建 refs 指针
refs_dir = os.path.expanduser(
    "~/Documents/projects/rust-projects/ark-1/model_cache/fastembed/models--rozgo--bge-reranker-v2-m3/refs"
)
os.makedirs(refs_dir, exist_ok=True)
with open(os.path.join(refs_dir, "main"), "w") as fh:
    fh.write("main\n")
print(f"  ✅ refs/main 已创建")

print("\n全部完成。现在可以直接运行: cargo run")
