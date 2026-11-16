from pathlib import Path

import slangpy as spy

from .simple_view import SimpleView
from .view import Viewer


def run(
    obj_path: Path = Path("assets/cube.obj"),
    weights_path: Path = Path("weights.bin"),
) -> None:
    """Open the interactive viewer (imports slangpy lazily)."""

    app = spy.App()
    _viewer = Viewer(app, obj_path, weights_path)  # keep a reference or the window is destroyed
    app.run()


def simple_run(obj_path: Path):
    app = spy.App()
    _ = SimpleView(app, obj_path=obj_path)
    app.run()
