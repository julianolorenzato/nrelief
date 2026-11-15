import pathlib

import numpy as np
import slangpy as spy

from .renderer import DEPTH_FORMAT, MeshRenderer


def look_at(eye, target, up=(0, 1, 0)):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def perspective(fov_y, aspect, near=0.05, far=100.0):
    """Right-handed, clip-space z in [0, 1]."""
    f = 1.0 / np.tan(fov_y / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = far / (near - far), near * far / (near - far)
    m[3, 2] = -1.0
    return m


def camera_mvp(yaw, pitch, dist, aspect):
    eye = dist * np.array(
        [
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw),
        ],
        dtype=np.float32,
    )
    return perspective(np.radians(50), aspect) @ look_at(
        eye, np.zeros(3, np.float32)
    )


SURFACE_FORMAT = spy.Format.bgra8_unorm_srgb


class Viewer(spy.AppWindow):
    def __init__(
        self, app: spy.App, obj_path: pathlib.Path, weights_path: pathlib.Path
    ):
        super().__init__(
            app,
            width=800,
            height=600,
            title="NRelief",
            resizable=False,
            surface_format=SURFACE_FORMAT,
        )

        self.renderer = MeshRenderer(
            self.device, obj_path, weights_path, SURFACE_FORMAT
        )
        self.depth: spy.Texture | None = None
        self.yaw, self.pitch, self.dist = 0.6, 0.4, 2.5
        self.dragging: bool = False
        self.last: tuple[float, float] | None = None

    def on_mouse_event(self, event: spy.MouseEvent) -> None:
        if event.is_button_down():
            self.dragging, self.last = True, (event.pos.x, event.pos.y)
        elif event.is_button_up():
            self.dragging = False
        elif event.is_move() and self.dragging and self.last is not None:
            dx, dy = event.pos.x - self.last[0], event.pos.y - self.last[1]
            self.last = (event.pos.x, event.pos.y)
            self.yaw -= dx * 0.01
            self.pitch = float(np.clip(self.pitch + dy * 0.01, -1.5, 1.5))
        elif event.is_scroll():
            self.dist = float(
                np.clip(self.dist * (1 - event.scroll.y * 0.1), 0.5, 20)
            )

    def render(self, render_context: spy.AppWindow.RenderContext) -> None:
        surf = render_context.surface_texture
        if (
            self.depth is None
            or self.depth.width != surf.width
            or self.depth.height != surf.height
        ):
            self.depth = self.device.create_texture(
                format=DEPTH_FORMAT,
                width=surf.width,
                height=surf.height,
                usage=spy.TextureUsage.depth_stencil,
            )
        mvp = camera_mvp(
            self.yaw, self.pitch, self.dist, surf.width / surf.height
        )
        self.renderer.draw(
            render_context.command_encoder, surf, self.depth, mvp
        )


def view(
    obj_path=pathlib.Path("assets/cube.obj"),
    weights_path=pathlib.Path("weights.bin"),
):
    app = spy.App()
    _viewer = Viewer(
        app, obj_path, weights_path
    )  # keep a reference or the window is destroyed
    app.run()
