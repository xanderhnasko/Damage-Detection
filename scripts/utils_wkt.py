"""Utilities for parsing WKT (Well-Known Text) polygon formats from xView2 labels.

Converts WKT polygon strings to point lists and extracts bounding boxes.
"""
def parse_wkt_polygon(wkt: str):
    # wkt should be: "POLYGON ((x1 y1, x2 y2, ...))"
    wkt = wkt.strip()

    assert wkt.startswith("POLYGON"), f"Unsupported WKT: {wkt[:20]}"
    # remove parens
    inner = wkt[wkt.find("((")+2 : wkt.rfind("))")]


    pts = []
    for pair in inner.split(","):
        x_str, y_str = pair.strip().split() # split on blank eg x1 y1
        pts.append((float(x_str), float(y_str)))
    return pts

def bbox_from_points(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)] # want all area between smallest and largest x,y