import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from pycocotools import mask as mask_utils
except ImportError as exc:
    raise ImportError(
        "pycocotools is required for processing Entity masks. "
        "Install it with: pip install pycocotools"
    ) from exc


def _load_json(path):
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _decode_segmentation(segmentation, height, width):
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
    else:
        raise ValueError(f"Unknown segmentation type: {type(segmentation)}")
    return mask_utils.decode(rle)


def _build_split_sets(train_data, val_data, seed=0, val_ratio=0.1, test_ratio=0.1):
    train_files = [x["file_name"] for x in train_data["images"]]
    val_files_native = [x["file_name"] for x in val_data["images"]]

    all_files = sorted(set(train_files + val_files_native))
    train_pool = set(train_files)
    native_val = set(val_files_native)

    total = len(all_files)
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0,1), got {val_ratio}")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in [0,1), got {test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio + test_ratio must be < 1.0, got {val_ratio + test_ratio}"
        )

    target_val = int(round(val_ratio * total))
    target_test = int(round(test_ratio * total))
    target_test = max(target_test, len(native_val))

    rng = random.Random(seed)

    test_set = set(native_val)
    train_candidates_for_test = sorted(list(train_pool - test_set))
    add_to_test = max(0, target_test - len(test_set))
    if add_to_test > len(train_candidates_for_test):
        add_to_test = len(train_candidates_for_test)
    test_extra = set(rng.sample(train_candidates_for_test, add_to_test)) if add_to_test > 0 else set()
    test_set.update(test_extra)

    remaining_for_val = sorted(list(train_pool - test_set))
    add_to_val = min(target_val, len(remaining_for_val))
    val_set = set(rng.sample(remaining_for_val, add_to_val)) if add_to_val > 0 else set()

    overlap = val_set.intersection(test_set)
    if overlap:
        raise ValueError(f"Split overlap found with {len(overlap)} images")

    train_set = set(all_files) - val_set - test_set

    return {
        "all": all_files,
        "train": train_set,
        "val": val_set,
        "test": test_set,
        "native_val_count": len(native_val),
    }


def _write_split_file(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for v in sorted(values):
            f.write(v + "\n")


def _build_instance_masks(root, train_data, val_data):
    ann_per_image = defaultdict(list)

    id_to_filename = {}
    for image in train_data["images"]:
        id_to_filename[(0, image["id"])] = image["file_name"]
    for image in val_data["images"]:
        id_to_filename[(2, image["id"])] = image["file_name"]

    train_anns = train_data["annotations"]
    val_anns = val_data["annotations"]

    for ann in train_anns:
        fn = id_to_filename[(0, ann["image_id"])]
        ann_per_image[fn].append(ann)
    for ann in val_anns:
        fn = id_to_filename[(2, ann["image_id"])]
        ann_per_image[fn].append(ann)

    all_images = train_data["images"] + val_data["images"]
    image_meta = {im["file_name"]: im for im in all_images}

    for file_name in tqdm(sorted(image_meta.keys()), desc="Creating masks"):
        meta = image_meta[file_name]
        h, w = int(meta["height"]), int(meta["width"])

        mask = np.zeros((h, w), dtype=np.uint8)
        anns = ann_per_image.get(file_name, [])

        for idx, ann in enumerate(anns, start=1):
            instance_id = min(idx, 255)
            decoded = _decode_segmentation(ann["segmentation"], h, w)
            if decoded.ndim == 3:
                decoded = np.any(decoded, axis=2)
            mask[decoded > 0] = instance_id

        p = Path(file_name)
        mask_dir = root / f"{p.parts[0]}_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        out_path = mask_dir / f"{p.stem}_mask.png"
        Image.fromarray(mask, mode="L").save(out_path)


def main():
    parser = argparse.ArgumentParser(description="Process raw Entity dataset into masks and splits")
    parser.add_argument("--root", type=str, default=None, help="Path to data/entity folder")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for split sampling")
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Target validation split ratio over total samples (default: 0.1)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Target test split ratio over total samples (default: 0.1). Native val always stays in test.",
    )
    args = parser.parse_args()

    root = Path(args.root) if args.root is not None else Path(__file__).resolve().parent

    train_json = root / "entityseg_train_lr.json"
    val_json = root / "entityseg_val_lr.json"
    if not train_json.exists() or not val_json.exists():
        raise FileNotFoundError(
            "Expected entityseg_train_lr.json and entityseg_val_lr.json in data/entity/."
        )

    train_data = _load_json(train_json)
    val_data = _load_json(val_json)

    splits = _build_split_sets(
        train_data,
        val_data,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    _write_split_file(root / "val_ims.txt", splits["val"])
    _write_split_file(root / "test_ims.txt", splits["test"])

    _build_instance_masks(root, train_data, val_data)

    total = len(splits["all"])
    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])
    print(
        "Created splits and masks. "
        f"Requested val_ratio={args.val_ratio:.3f}, test_ratio={args.test_ratio:.3f}. "
        f"Total={total}, train={n_train} ({n_train/total:.3f}), "
        f"val={n_val} ({n_val/total:.3f}), test={n_test} ({n_test/total:.3f}), "
        f"native_val_in_test={splits['native_val_count']}"
    )


if __name__ == "__main__":
    main()
