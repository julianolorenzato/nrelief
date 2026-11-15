import json
from pathlib import Path

import torch


def export(checkpoint="checkpoint.pt", out_dir="."):
    """Flatten each Linear as [weights (out x in, row-major), bias] into weights.bin."""
    ckpt = torch.load(checkpoint)
    state = ckpt["state"]
    chunks = []
    for name in state:  # TODO: ensure order/layout matches the shader
        chunks.append(state[name].flatten().float())
    flat = torch.cat(chunks).numpy()
    out = Path(out_dir)
    flat.tofile(out / "weights.bin")
    (out / "meta.json").write_text(json.dumps({"sizes": list(ckpt["sizes"])}))
