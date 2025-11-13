#!/usr/bin/env python3
"""Build manifest CSV from label JSONs.
Reads JSONs with WKT polygons, extracts bboxes for each damage annotation, and makoes a CSV manifest with image paths, bbox coordinates, and damage labels
"""
import argparse, json, os, csv, math
from pathlib import Path
from tqdm import tqdm
from utils_wkt import parse_wkt_polygon, bbox_from_points


DAMAGE_MAP = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}

def clip_bbox(b, w, h, pad=16):
    """Clip bounding box to image dims + padding"""

    xmin, ymin, xmax, ymax = b
    # Expand bbox by padding, but not past image boundaries
    xmin = int(math.floor(max(0, xmin - pad))) 
    ymin = int(math.floor(max(0, ymin - pad)))
    xmax = int(math.ceil(min(w - 1, xmax + pad))) 
    ymax = int(math.ceil(min(h - 1, ymax + pad)))
    
    # degenerate bbox
    if xmax <= xmin or ymax <= ymin:
        return None  
    return [xmin, ymin, xmax, ymax]

def guess_pre_name(post_name: str):
    return post_name.replace("_post_disaster", "_pre_disaster")

def build_rows_for_label(label_path: Path, images_root: Path, pad: int, local_root: str = None):
    # Load and parse the label JSON file
    j = json.loads(label_path.read_text())
    meta = j.get("metadata", {})
    
    # Extract image metadata
    img_name = meta.get("img_name")  # e.g. guatemala-volcano_00000000_post_disaster.png

    # fallback to original if width/height not present
    width = int(meta.get("width", meta.get("original_width", 0)))
    height = int(meta.get("height", meta.get("original_height", 0)))
    
    # check for invalid labels
    if not img_name or width <= 0 or height <= 0:
        return []  

    # Find post-disaster image file
    post_img_path = images_root / img_name
    if not post_img_path.exists():
        # Fallback to derive filename from label filename
        post_img_path = images_root / (label_path.stem + ".png")

    if not post_img_path.exists():
        print(f"Post image not found for {label_path}")
        return [] 

    # find corresponding pre-disaster image 
    pre_img_name = guess_pre_name(post_img_path.name)
    pre_img_path = images_root / pre_img_name
    pre_exists = pre_img_path.exists()  


    # Damage annotations from JSON
    feats = j.get("features", {})
    xy = feats.get("xy", [])  # gives list of feature annotations
    
    rows = []
    # loop thru damage annotations
    for f in xy:
        props = f.get("properties", {})
        subtype = props.get("subtype")  #category: "no-damage", "minor-damage", etc.
        uid = props.get("uid")  
        
       # if unclassified, skip
        if subtype not in DAMAGE_MAP:
            continue
        
        # WKT polygon to bounding box
        pts = parse_wkt_polygon(f.get("wkt", ""))
        bb = bbox_from_points(pts) 
        bb = clip_bbox(bb, width, height, pad=pad)

        if not bb:
            continue  # Skip invalid bboxes
        
        # Rewrite paths if local_root specified
        final_post_path = str(post_img_path)
        final_pre_path = str(pre_img_path) if pre_exists else ""
        
        if local_root:
            # Replace GCSFuse path with local path
            final_post_path = str(post_img_path).replace(str(images_root), local_root)
            if pre_exists:
                final_pre_path = str(pre_img_path).replace(str(images_root), local_root)
        
        # Create row for CSV manifest
        rows.append({
            "img_path": final_post_path,  
            "pre_img_path": final_pre_path,  
            "xmin": bb[0], "ymin": bb[1], "xmax": bb[2], "ymax": bb[3],  # Bbox coords
            "label_id": DAMAGE_MAP[subtype],  
            "label_name": subtype,  
            "uid": uid,  
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--labels_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--pad", type=int, default=16)
    ap.add_argument("--local_root", type=str, default=None)
    args = ap.parse_args()

  
    images_root = Path(args.images_root)
    labels_root = Path(args.labels_root)
    out_csv = Path(args.out_csv)
    
   
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Find all label JSONs
    label_files = list(labels_root.rglob("*.json"))
    
    # Write actual manifest
    nrows = 0
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "img_path","pre_img_path","xmin","ymin","xmax","ymax",
            "label_id","label_name","uid"
        ])
        writer.writeheader()  
    
        for lp in tqdm(label_files, desc="Labels"):
            for row in build_rows_for_label(lp, images_root, args.pad, args.local_root):
                writer.writerow(row)
                nrows += 1
    
    print(f"[wrote {nrows} rows to {out_csv}")

if __name__ == "__main__":
    main()
