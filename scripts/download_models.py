# download_models.py —— 六模型 GGUF 下载（ModelScope 镜像 + sha256 校验）
# GGUF 镜像下载与 checksum；ZIP 离线复现用
# 学生/作者均用此脚本：下载后统一命名 <模型名>-<量化档>.gguf 放入 model/
# 运行：python scripts/download_models.py [--model qwen3-4b]（缺省下载全部）
import argparse
import hashlib
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

# 六模型：ModelScope repo + 仓库内文件名 + 本地命名 + sha256（与 model_hashes.json 一致）
# 来源：Qwen3 系 = Qwen 官方 GGUF 仓库；gemma/Phi-4-Mini = unsloth 转换仓库
MODELS = {
    "qwen3-0.6b": {
        "repo": "Qwen/Qwen3-0.6B-GGUF", "remote": "Qwen3-0.6B-Q8_0.gguf",
        "local": "Qwen3-0.6B-Instruct-Q8_0.gguf",
        "sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
    },
    "qwen3-1.7b": {
        "repo": "Qwen/Qwen3-1.7B-GGUF", "remote": "Qwen3-1.7B-Q8_0.gguf",
        "local": "Qwen3-1.7B-Instruct-Q8_0.gguf",
        "sha256": "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
    },
    "qwen3-4b": {
        "repo": "Qwen/Qwen3-4B-GGUF", "remote": "Qwen3-4B-Q8_0.gguf",
        "local": "Qwen3-4B-Instruct-Q8_0.gguf",
        "sha256": "8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300",
    },
    "gemma-3-1b": {
        "repo": "unsloth/gemma-3-1b-it-GGUF", "remote": "gemma-3-1b-it-Q8_0.gguf",
        "local": "gemma-3-1b-it-Q8_0.gguf",
        "sha256": "616dfb049ad14288d971d96f5ca4953fdebbf1e3cd407ad159f3bfd47090201d",
    },
    "gemma-3-4b": {
        "repo": "unsloth/gemma-3-4b-it-GGUF", "remote": "gemma-3-4b-it-Q8_0.gguf",
        "local": "gemma-3-4b-it-Q8_0.gguf",
        "sha256": "81bf0583ab5bad155a5a3b15d155a880a1a1e4f7de2de5c06f10f64ac49f8336",
    },
    "phi-4-mini": {
        "repo": "unsloth/Phi-4-mini-instruct-GGUF", "remote": "Phi-4-mini-instruct.Q8_0.gguf",
        "local": "Phi-4-Mini-Instruct-Q8_0.gguf",
        "sha256": "26188c6050d525376a88b04514c236c5e28a36730f1e936f2a00314212b7ba42",
    },
}


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="六模型 GGUF 下载（ModelScope + sha256 校验）")
    ap.add_argument("--model", default=None, choices=sorted(MODELS),
                    help="只下载指定模型（缺省全部）")
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    targets = [args.model] if args.model else sorted(MODELS)
    for key in targets:
        info = MODELS[key]
        dst = os.path.join(MODEL_DIR, info["local"])
        if os.path.exists(dst):
            # 已存在：校验 sha256，一致则跳过
            if sha256_file(dst) == info["sha256"]:
                print(f"[跳过] {info['local']} 已存在且校验通过")
                continue
            print(f"[覆盖] {info['local']} 已存在但 sha256 不符，重新下载")
        print(f"[下载] {info['repo']}/{info['remote']} → {info['local']}")
        try:
            from modelscope import snapshot_download
            cache_dir = snapshot_download(info["repo"], allow_file_pattern=[info["remote"]])
            src = os.path.join(cache_dir, info["remote"])
            if not os.path.exists(src):
                # 兼容子目录结构
                for root, _, files in os.walk(cache_dir):
                    if info["remote"] in files:
                        src = os.path.join(root, info["remote"])
                        break
            if not os.path.exists(src):
                print(f"  [失败] 下载文件中未找到 {info['remote']}")
                continue
            # 移动到 model/ 并重命名
            tmp = dst + ".part"
            if os.path.exists(tmp):
                os.remove(tmp)
            os.rename(src, tmp)
            digest = sha256_file(tmp)
            if digest == info["sha256"]:
                os.rename(tmp, dst)
                print(f"  [完成] {info['local']} sha256 校验通过")
            else:
                os.remove(tmp)
                print(f"  [失败] sha256 不符：期望 {info['sha256'][:16]}… 实际 {digest[:16]}…")
        except Exception as e:
            print(f"  [失败] 下载异常：{str(e)[:150]}")
    print("全部完成")


if __name__ == "__main__":
    main()
