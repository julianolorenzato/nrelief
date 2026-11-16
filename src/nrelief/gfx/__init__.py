from pathlib import Path

import slangpy as spy

from .viewer import Viewer


def run(obj_path: Path):
    app = spy.App()
    _ = Viewer(app, obj_path=obj_path)
    app.run()
