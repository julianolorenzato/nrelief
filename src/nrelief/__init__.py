from pathlib import Path

import typer

app = typer.Typer(help="NRelief: train, export and view a neural relief model.", no_args_is_help=True)


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
def view(
    obj_path: Path = typer.Argument("assets/cube.obj", help="OBJ mesh to render."),
    weights_path: Path = typer.Argument("weights.bin", help="Exported weights file."),
):
    """Open the interactive viewer."""
    from .mesh.view import view as run

    run(obj_path=obj_path, weights_path=weights_path)


def main():
    app()
