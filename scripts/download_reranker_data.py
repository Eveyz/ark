import urllib.request
import os

target_dir = os.path.expanduser(
    "~/Documents/projects/rust-projects/ark-1/model_cache/fastembed/models--rozgo--bge-reranker-v2-m3/snapshots/main"
)
os.makedirs(target_dir, exist_ok=True)

url = "https://huggingface.co/rozgo/bge-reranker-v2-m3/resolve/main/model.onnx.data"
out_path = os.path.join(target_dir, "model.onnx.data")

if os.path.exists(out_path):
    size = os.path.getsize(out_path)
    print(f"model.onnx.data 已存在 ({size:,} bytes)，跳过")
else:
    print("开始下载 model.onnx.data (可能较大，请耐心等待)...")
    print(f"保存到: {out_path}")
    try:
        urllib.request.urlretrieve(url, out_path)
        size = os.path.getsize(out_path)
        print(f"下载完成: {size:,} bytes")
    except Exception as e:
        print(f"下载失败: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)
