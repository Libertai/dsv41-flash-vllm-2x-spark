#!/usr/bin/env python3
"""Check each safetensors shard's declared data extent against its file size."""
import glob, json, os, struct, sys
root = sys.argv[1]
bad = 0
for p in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
    size = os.path.getsize(os.path.realpath(p))
    with open(p, "rb") as fh:
        hl = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hl))
    end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
    need = 8 + hl + end
    flag = "OK " if size == need else "BAD"
    if size != need:
        bad += 1
        print(f"{flag} {os.path.basename(p)}  file={size:,}  declared={need:,}  short by {need-size:,}")
n_shards = len(glob.glob(os.path.join(root, "*.safetensors")))
# "0 bad of 0" is not a pass -- an empty or wrong directory must not look healthy.
if n_shards == 0:
    print(f"NO SHARDS FOUND in {root!r} -- wrong directory, or the download never ran")
    sys.exit(2)
print(f"{bad} bad shard(s) of {n_shards}")
sys.exit(1 if bad else 0)
