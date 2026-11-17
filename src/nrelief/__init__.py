from pathlib import Path

import typer

app = typer.Typer(
    help="NRelief: train, export and view a neural relief model.",
    no_args_is_help=True,
)


@app.command()
def train(
    out: Path = typer.Option("checkpoint.pt", help="Where to save the checkpoint."),
    epochs: int = typer.Option(1, min=1, help="Number of training epochs."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
):
    """Train the model and save a checkpoint."""
    from .network.train import train as run

    run(out=out, epochs=epochs, lr=lr)


@app.command()
def export(
    checkpoint: Path = typer.Option("checkpoint.pt", help="Checkpoint to export."),
    out_dir: Path = typer.Option(".", help="Directory for weights.bin and meta.json."),
):
    """Export checkpoint weights to weights.bin and meta.json."""
    from .network.export import export as run

    run(checkpoint=checkpoint, out_dir=out_dir)


@app.command()
def view(obj_path: Path = typer.Argument("assets/cube.obj", help="OBJ mesh to render.")):
    """Open the interactive viewer."""
    from . import gfx

    gfx.run(obj_path=obj_path)


@app.command()
def bake(
    mesh_path: Path = typer.Argument(..., help="Mesh to bake (OBJ etc.)."),
    out_dir: Path = typer.Option("bake_out", help="Directory for the PNG files."),
    res: int = typer.Option(64, min=1, help="Texels per side."),
    max_hits: int = typer.Option(4, min=1, max=4),
):
    """Bake depth layers per axis and save one 16-bit PNG per axis and layer."""
    import numpy as np
    import open3d as o3d

    from .gfx.bake import bake
    from .gfx.mesh import Mesh

    layers = bake(Mesh(mesh_path), resolution=res, max_hits=max_hits)

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for axis, maps in enumerate(layers):
        depth = maps["depth"]
        for k in range(depth.shape[2]):
            # z em [0, 1] -> uint16; texel sem hit (inf) vira 65535.
            img = np.where(np.isinf(depth[..., k]), 1.0, depth[..., k])
            img = np.rint(np.clip(img, 0.0, 1.0) * 65535).astype(np.uint16)
            path = out_dir / f"depth_axis{axis}_hit{k}.png"
            o3d.io.write_image(path, o3d.geometry.Image(np.ascontiguousarray(img)))
            saved += 1
            # Albedo/normal amostrados por camada, RGBA 8-bit (alpha 0 = sem hit).
            for name in ("albedo", "normal"):
                if maps[name] is not None:
                    path = out_dir / f"{name}_axis{axis}_hit{k}.png"
                    img = np.ascontiguousarray(maps[name][:, :, k])
                    o3d.io.write_image(path, o3d.geometry.Image(img))
                    saved += 1
    typer.echo(f"saved {saved} images to {out_dir}")


def main():
    app()
