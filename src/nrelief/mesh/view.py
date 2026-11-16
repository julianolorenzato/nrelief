import pathlib

import slangpy as spy

from .renderer import DEPTH_FORMAT, MeshRenderer


def look_at(
    eye: spy.float3, target: spy.float3, up: spy.float3 = spy.float3(0, 1, 0)
) -> spy.float4x4:
    f = spy.math.normalize(target - eye)
    s = spy.math.normalize(spy.math.cross(f, up))
    u = spy.math.cross(s, f)
    return spy.float4x4(
        [
            s.x, s.y, s.z, -spy.math.dot(s, eye),
            u.x, u.y, u.z, -spy.math.dot(u, eye),
            -f.x, -f.y, -f.z, spy.math.dot(f, eye),
            0.0, 0.0, 0.0, 1.0,
        ]
    )  # fmt: skip


def perspective(
    fov_y: float, aspect: float, near: float = 0.05, far: float = 100.0
) -> spy.float4x4:
    """Right-handed, clip-space z in [0, 1]."""
    f = 1.0 / spy.math.tan(fov_y / 2)
    return spy.float4x4(
        [
            f / aspect, 0.0, 0.0, 0.0,
            0.0, f, 0.0, 0.0,
            0.0, 0.0, far / (near - far), near * far / (near - far),
            0.0, 0.0, -1.0, 0.0,
        ]
    )  # fmt: skip


def camera_mvp(
    yaw: float, pitch: float, dist: float, aspect: float
) -> spy.float4x4:
    eye = spy.float3(
        dist * spy.math.cos(pitch) * spy.math.sin(yaw),
        dist * spy.math.sin(pitch),
        dist * spy.math.cos(pitch) * spy.math.cos(yaw),
    )
    return spy.math.mul(
        perspective(spy.math.radians(50), aspect),
        look_at(eye, spy.float3(0, 0, 0)),
    )


def clamp(x: float, lo: float, hi: float) -> float:
    # spy.math.clamp only accepts vector types, so wrap in float1
    return spy.math.clamp(
        spy.math.float1(x), spy.math.float1(lo), spy.math.float1(hi)
    ).x


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
            self.pitch = clamp(self.pitch + dy * 0.01, -1.5, 1.5)
        elif event.is_scroll():
            self.dist = clamp(self.dist * (1 - event.scroll.y * 0.1), 0.5, 20)

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
