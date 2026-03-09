import json
from pathlib import Path
from typing import Any
import numpy as np


ND_TAG = "__ndarray_ref__"


def _to_jsonable(obj: Any):
    # numpy scalar -> python scalar
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def split_ndarrays_to_npz_json(data: Any, npz_path: str, json_path: str):
    """
    把 data 中所有 ndarray 抽出存成 npz，JSON 留 reference。
    """
    arrays = {}

    def walk(x: Any, path: str = "root"):
        if isinstance(x, np.ndarray):
            key = path.replace(".", "__").replace("[", "_").replace("]", "")
            arrays[key] = x
            return {
                ND_TAG: key,
                "shape": list(x.shape),
                "dtype": str(x.dtype),
            }

        if isinstance(x, dict):
            return {k: walk(v, f"{path}.{k}") for k, v in x.items()}

        if isinstance(x, list):
            return [walk(v, f"{path}[{i}]") for i, v in enumerate(x)]

        return _to_jsonable(x)

    manifest = walk(data)

    npz_path = Path(npz_path)
    json_path = Path(json_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def load_from_npz_json(npz_path: str, json_path: str):
    """
    由 npz + JSON manifest 還原原始結構（ndarray 會回來）。
    """
    manifest = json.loads(Path(json_path).read_text(encoding="utf-8"))
    npz = np.load(npz_path, allow_pickle=False)

    def walk(x: Any):
        if isinstance(x, dict) and ND_TAG in x:
            return npz[x[ND_TAG]]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x

    return walk(manifest)


if __name__ == "__main__":
    # 假設 Markovian_data 是你的原始 dict
    Markovian_data = {
        "a": np.arange(6).reshape(2, 3),
        "meta": {"name": "test"},
        "b": [np.random.randn(4), 123],
    }

    split_ndarrays_to_npz_json(
        Markovian_data,
        npz_path="markovian_arrays.npz",
        json_path="markovian_manifest.json",
    )

    restored = load_from_npz_json(
        npz_path="markovian_arrays.npz",
        json_path="markovian_manifest.json",
    )

    print(type(restored["a"]), restored["a"].shape)