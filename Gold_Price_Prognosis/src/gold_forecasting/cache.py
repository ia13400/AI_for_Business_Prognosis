"""Content-addressed artifact cache with adjacent manifests."""
import json
from pathlib import Path
from typing import Any
from .hashing import stable_hash

class ArtifactCache:
    def __init__(self, root: Path): self.root = root
    def signature(self, inputs: dict[str, Any]) -> str: return stable_hash(inputs)
    def paths(self, namespace: str, kind: str, signature: str, suffix: str) -> tuple[Path, Path]:
        directory = self.root / namespace / kind; directory.mkdir(parents=True, exist_ok=True)
        artifact = directory / f"{signature}.{suffix.lstrip('.')}"
        return artifact, artifact.with_suffix(artifact.suffix + ".manifest.json")
    def valid(self, artifact: Path, manifest: Path, signature: str) -> bool:
        if not artifact.exists() or not manifest.exists(): return False
        try: return json.loads(manifest.read_text(encoding="utf-8"))["signature"] == signature
        except (OSError, KeyError, json.JSONDecodeError): return False
    def write_manifest(self, manifest: Path, signature: str, inputs: dict[str, Any], extra: dict | None = None) -> None:
        manifest.write_text(json.dumps({"signature": signature, "inputs": inputs, **(extra or {})}, indent=2, default=str), encoding="utf-8")
