import hashlib
from pathlib import Path

MODEL_ID = "MirroS-Lab/Code-as-World-VL-9B"
MODEL_REVISION = "46b111c53f5680eb12c63a7391e5c690d1e14ab1"
CHECKPOINT_FILES = {
    "chat_template.jinja": ("git", 7585, "dad26391f4e996320915996a99c31e3de9270ec2"),
    "config.json": ("git", 2908, "be6ab37a8d620b01dcc3deb6d5f1e7a226f86699"),
    "generation_config.json": ("git", 206, "a2e5671a75d8c47fd4813058fee9d2e34363730d"),
    "model.safetensors": (
        "sha256",
        18819722392,
        "02e6483033e05b046ce909ae4bfddba3900c0e4d6c6b0b5eae95d1245adf1ec3",
    ),
    "processor_config.json": ("git", 1191, "33818c7f9e991ad735fd240209f4fa73e6c28c50"),
    "tokenizer.json": (
        "sha256",
        19989325,
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    ),
    "tokenizer_config.json": ("git", 1192, "a24f83f0b01c5220e20b7b299ee543c74433383b"),
}


def verify_checkpoint(path):
    path = Path(path)
    for name, (kind, size, expected) in CHECKPOINT_FILES.items():
        file = path / name
        if not file.is_file() or file.stat().st_size != size:
            raise ValueError(f"Missing or incorrect checkpoint file: {name}")
        digest = hashlib.sha256() if kind == "sha256" else hashlib.sha1(usedforsecurity=False)
        if kind == "git":
            digest.update(f"blob {size}\x00".encode())
        with file.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"Checkpoint content differs from the pinned revision: {name}")
    for file in path.iterdir():
        if (
            file.is_file()
            and file.name not in CHECKPOINT_FILES
            and (file.name != "pss_model_provenance.json")
        ) and file.suffix in {".json", ".jinja", ".safetensors", ".bin", ".model", ".txt", ".py"}:
            raise ValueError(f"Unpinned model configuration or weights: {file.name}")


def resolve_checkpoint(model=MODEL_ID, revision=MODEL_REVISION):
    if revision != MODEL_REVISION:
        raise ValueError("This inference interface uses the published model revision")
    path = Path(model).expanduser()
    if not path.is_dir():
        if str(model) != MODEL_ID:
            raise FileNotFoundError(f"Checkpoint directory does not exist: {model}")
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=MODEL_ID, revision=revision))
    path = path.resolve()
    verify_checkpoint(path)
    return (path, {"model_id": MODEL_ID, "revision": revision})


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="pss checkpoint",
        description="Download and verify the pinned Code-as-World checkpoint.",
    )
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args(argv)
    model = MODEL_ID
    if args.directory is not None:
        from huggingface_hub import snapshot_download

        model = snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=args.directory,
            allow_patterns=list(CHECKPOINT_FILES),
        )
    print(resolve_checkpoint(model)[0])
