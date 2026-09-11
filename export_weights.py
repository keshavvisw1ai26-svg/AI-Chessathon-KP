"""Convert weights.npz to the flat little-endian weights.bin read by web/engine.js."""
import struct
import sys
from pathlib import Path

import numpy as np

src = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "weights.npz")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parent / "weights.bin")
z = np.load(src)
with open(dst, "wb") as f:
    f.write(struct.pack("<I", 0x4555_4E4E))  # "NNUE"
    f.write(np.ascontiguousarray(z["ft"], dtype="<i2").tobytes())
    f.write(np.ascontiguousarray(z["ftb"], dtype="<i2").tobytes())
    f.write(np.ascontiguousarray(z["ow"], dtype="<i4").tobytes())
    f.write(np.ascontiguousarray(z["ob"], dtype="<i4").tobytes())
    f.write(np.ascontiguousarray(z["flip"], dtype="<i4").tobytes())
print(f"wrote {dst} ({dst.stat().st_size} bytes)")
