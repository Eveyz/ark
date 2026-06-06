import urllib.request
import os

target_dir = os.path.expanduser(
    "~/Documents/projects/rust-projects/ark-1/model_cache/fastembed/models--BAAI--bge-reranker-base/snapshots/main"
)
os.makedirs(os.path.join(target_dir, "onnx"), exist_ok=True)

base_url = "https://huggingface.co/BAAI/bge-reranker-base/resolve/main/"
files = {
    "onnx/model.onnx": os.path.join(target_dir, "onnx", "model.onnx"),
    "tokenizer.json": os.path.join(target_dir, "tokenizer.json"),
    "config.json": os.path.join(target_dir, "config.json"),
}

print("开始下载 BAAI/bge-reranker-base 模型文件...")
for rel_path, out_path in files.items():
    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"  ✅ {rel_path} 已存在 ({size:,} bytes)，跳过")
        continue
    url = base_url + rel_path
    print(f"  ⏳ 下载 {rel_path} ...")
    try:
        urllib.request.urlretrieve(url, out_path)
        size = os.path.getsize(out_path)
        print(f"  ✅ {rel_path} 下载完成 ({size:,} bytes)")
    except Exception as e:
        print(f"  ❌ {rel_path} 下载失败: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)

# 创建 refs 指针
refs_dir = os.path.expanduser(
    "~/Documents/projects/rust-projects/ark-1/model_cache/fastembed/models--BAAI--bge-reranker-base/refs"
)
os.makedirs(refs_dir, exist_ok=True)
with open(os.path.join(refs_dir, "main"), "w") as fh:
    fh.write("main")
print(f"  ✅ refs/main 已创建")

print("\n全部完成。记得把 src/main.rs 里的 RerankerModel 改成 BGERerankerBase，然后 cargo run")
