import os, sys
root = sys.argv[1] if len(sys.argv) > 1 else "."
for d, _, files in os.walk(root):
    print(d, ":", len(files), "files")
    for f in files[:3]: print("  -", f)
    print()
