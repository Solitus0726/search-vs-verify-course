# download_models.py —— download GGUF for six models (ModelScope mirror + sha256 verification)
# GGUF mirror download and checksum; for ZIP offline reproduction
# Used by both students and authors: after download, files are named <model>-<quant>.gguf and placed in model/
# Run: python scripts/download_models.py [--model qwen3-4b] (downloads all by default)
import argparse
import hashlib
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

# Six models: ModelScope repo + file name in repo + local name + sha256
# Sources: Qwen3 family = Qwen official GGUF repo; gemma/Phi-4-Mini = unsloth conversion repo
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
    ap = argparse.ArgumentParser(description="download GGUF for six models (ModelScope + sha256 verification)")
    ap.add_argument("--model", default=None, choices=sorted(MODELS),
                    help="download only the specified model (all by default)")
    args = ap.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    targets = [args.model] if args.model else sorted(MODELS)
    for key in targets:
        info = MODELS[key]
        dst = os.path.join(MODEL_DIR, info["local"])
        if os.path.exists(dst):
            # Already exists: verify sha256; skip if it matches
            if sha256_file(dst) == info["sha256"]:
                print(f"[skip] {info['local']} already exists and passed verification")
                continue
            print(f"[overwrite] {info['local']} exists but sha256 mismatch, downloading again")
        print(f"[download] {info['repo']}/{info['remote']} → {info['local']}")
        try:
            from modelscope import snapshot_download
            cache_dir = snapshot_download(info["repo"], allow_file_pattern=[info["remote"]])
            src = os.path.join(cache_dir, info["remote"])
            if not os.path.exists(src):
                # tolerate subdirectory layout
                for root, _, files in os.walk(cache_dir):
                    if info["remote"] in files:
                        src = os.path.join(root, info["remote"])
                        break
            if not os.path.exists(src):
                print(f"  [failed] {info['remote']} not found among downloaded files")
                continue
            # move to model/ and rename
            tmp = dst + ".part"
            if os.path.exists(tmp):
                os.remove(tmp)
            os.rename(src, tmp)
            digest = sha256_file(tmp)
            if digest == info["sha256"]:
                os.rename(tmp, dst)
                print(f"  [done] {info['local']} sha256 verification passed")
            else:
                os.remove(tmp)
                print(f"  [failed] sha256 mismatch: expected {info['sha256'][:16]}… got {digest[:16]}…")
        except Exception as e:
            print(f"  [failed] download error: {str(e)[:150]}")
    print("all done")


if __name__ == "__main__":
    main()
