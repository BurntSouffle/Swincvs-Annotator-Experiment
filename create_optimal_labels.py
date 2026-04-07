"""
Create optimal-label ds_coco JSONs for SwinCVS Model C.

For each split, creates annotation_ds_coco_optimal.json where:
  C1 = Annotator 3's binary label
  C2 = Majority vote (round of average, same as original)
  C3 = 1 if BOTH Ann2 AND Ann3 say yes, else 0
"""

import json
import ast
import copy
import os
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(os.environ.get("DATASET_DIR", str(Path(__file__).resolve().parent.parent)))
ENDOSCAPES_DIR = BASE_DIR / "endoscapes"
METADATA_CSV = ENDOSCAPES_DIR / "all_metadata.csv"


def parse_annotator_votes(vote_str):
    try:
        parsed = ast.literal_eval(vote_str)
        if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
            return [int(v) for v in parsed]
    except (ValueError, SyntaxError):
        pass
    return None


def main():
    # Load metadata with per-annotator labels
    meta = pd.read_csv(METADATA_CSV)
    meta = meta[meta["is_ds_keyframe"] == True].copy()

    # Parse annotator votes
    for ann_idx in [2, 3]:
        col = f"cvs_annotator_{ann_idx}"
        parsed = meta[col].apply(parse_annotator_votes)
        meta[f"C1_ann{ann_idx}"] = parsed.apply(lambda x: x[0] if x else np.nan)
        meta[f"C2_ann{ann_idx}"] = parsed.apply(lambda x: x[1] if x else np.nan)
        meta[f"C3_ann{ann_idx}"] = parsed.apply(lambda x: x[2] if x else np.nan)

    # Build lookup: (vid, frame) -> optimal labels
    optimal_labels = {}
    for _, row in meta.iterrows():
        vid = int(row["vid"])
        frame = int(row["frame"])
        key = f"{vid}_{frame}"

        c1_ann3 = row["C1_ann3"]
        c2_mv = round(row["C2"])  # same as original
        c3_and = 1.0 if (row["C3_ann2"] == 1 and row["C3_ann3"] == 1) else 0.0

        if np.isnan(c1_ann3):
            continue

        optimal_labels[key] = [float(c1_ann3), float(c2_mv), float(c3_and)]

    print(f"Built optimal labels for {len(optimal_labels)} keyframes")

    # Process each split
    for split in ["train", "val", "test"]:
        json_path = ENDOSCAPES_DIR / split / "annotation_ds_coco.json"
        out_path = ENDOSCAPES_DIR / split / "annotation_ds_coco_optimal.json"

        with open(json_path) as f:
            data = json.load(f)

        modified = 0
        unmatched = 0
        original_data = copy.deepcopy(data)

        for img in data["images"]:
            fname = img["file_name"].split(".")[0]  # e.g. "8_14525"
            if fname in optimal_labels:
                img["ds"] = optimal_labels[fname]
                modified += 1
            else:
                unmatched += 1

        with open(out_path, "w") as f:
            json.dump(data, f)

        print(f"\n{split}: {modified} modified, {unmatched} unmatched, total {len(data['images'])}")

        # Verify: compare positive rates
        orig_c1 = sum(1 for img in original_data["images"] if round(img["ds"][0]) == 1)
        orig_c2 = sum(1 for img in original_data["images"] if round(img["ds"][1]) == 1)
        orig_c3 = sum(1 for img in original_data["images"] if round(img["ds"][2]) == 1)
        new_c1 = sum(1 for img in data["images"] if round(img["ds"][0]) == 1)
        new_c2 = sum(1 for img in data["images"] if round(img["ds"][1]) == 1)
        new_c3 = sum(1 for img in data["images"] if round(img["ds"][2]) == 1)
        n = len(data["images"])
        print(f"  Original MV: C1={orig_c1}/{n} ({orig_c1/n*100:.1f}%), C2={orig_c2}/{n} ({orig_c2/n*100:.1f}%), C3={orig_c3}/{n} ({orig_c3/n*100:.1f}%)")
        print(f"  Optimal:     C1={new_c1}/{n} ({new_c1/n*100:.1f}%), C2={new_c2}/{n} ({new_c2/n*100:.1f}%), C3={new_c3}/{n} ({new_c3/n*100:.1f}%)")

        # Spot-check 5 frames
        print(f"  Spot check:")
        for img_orig, img_new in zip(original_data["images"][:5], data["images"][:5]):
            fname = img_orig["file_name"]
            print(f"    {fname}: MV={[round(x,2) for x in img_orig['ds']]} -> Opt={[round(x,2) for x in img_new['ds']]}")


if __name__ == "__main__":
    main()
