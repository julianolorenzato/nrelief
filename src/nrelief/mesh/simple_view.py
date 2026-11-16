import time
from pathlib import Path

import numpy as np
import open3d as o3d
import slangpy as spy

TITLE = "Neural Relief Mapping"
SHADERS_DIR = Path(__file__).parents[3] / "shaders"
SHADER_PROGRAM_ENTRYPOINT = "simple.slang"
SURFACE_FORMAT = spy.Format.bgra8_unorm_srgb
DEPTH_STENCIL_FORMAT = spy.Format.d32_float_s8_uint
INITIAL_WIDTH = 800
INITIAL_HEIGHT = 600


def rotation_y(angle: float) -> np.ndarray:
    """Model matrix: rotation around the Y axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """View matrix: right-handed, camera looks down -Z."""
    eye, target, up = (np.asarray(v, dtype=np.float32) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    view = np.identity(4, dtype=np.float32)
    view[0, :3], view[1, :3], view[2, :3] = s, u, -f
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Projection matrix: right-handed, clip-space z in [0, 1]."""
    t = 1.0 / np.tan(fov_y / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = t / aspect
    proj[1, 1] = t
    proj[2, 2] = far / (near - far)
    proj[2, 3] = near * far / (near - far)
    proj[3, 2] = -1.0
    return proj


class SimpleView(spy.AppWindow):
    def __init__(self, app, obj_path: Path):
        super().__init__(
            app,
            width=INITIAL_WIDTH,
            height=INITIAL_HEIGHT,
            title=TITLE,
            surface_format=SURFACE_FORMAT,
        )

        program = self.device.load_program(
            str(SHADERS_DIR / SHADER_PROGRAM_ENTRYPOINT), ["vertexMain", "fragmentMain"]
        )

        mesh = o3d.io.read_triangle_mesh(obj_path)

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        self.vbuf = self.device.create_buffer(usage=spy.BufferUsage.vertex_buffer, data=verts)

        self.pipeline = self.device.create_render_pipeline(
            program=program,
            input_layout=None,
            targets=[{"format": SURFACE_FORMAT}],
            rasterizer={"cull_mode": spy.CullMode.back, "fill_mode": spy.FillMode.solid},
            depth_stencil={
                "format": DEPTH_STENCIL_FORMAT,
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": spy.ComparisonFunc.less,
            },
        )

        self.depth_stencil_tex = self.device.create_texture(
            format=DEPTH_STENCIL_FORMAT,
            width=INITIAL_WIDTH,
            height=INITIAL_HEIGHT,
            usage=spy.TextureUsage.depth_stencil,
        )

    def render(self, render_context: spy.AppWindow.RenderContext):
        surf_tex = render_context.surface_texture
        cmd_enc = render_context.command_encoder

        w, h = surf_tex.width, surf_tex.height

        rp_desc: spy.RenderPassDescDict = {
            "color_attachments": [
                {
                    "load_op": spy.LoadOp.clear,
                    "store_op": spy.StoreOp.store,
                    "clear_value": [0.5, 0.5, 1.0, 1.0],
                    "view": surf_tex.create_view({}),
                }
            ],
            "depth_stencil_attachment": {
                "depth_clear_value": 1.0,
                "depth_load_op": spy.LoadOp.clear,
                "view": self.depth_stencil_tex.create_view({}),
            },
        }

        with cmd_enc.begin_render_pass(rp_desc) as rp:
            so = rp.bind_pipeline(self.pipeline)

            cursor = spy.ShaderCursor(so)
            cursor.rmode = 1
            model = rotation_y(time.perf_counter() * 0.5)
            view = look_at(eye=(0.0, 0.5, 3.0), target=(0.0, 0.0, 0.0))
            proj = perspective(np.radians(50.0), w / h, near=0.05, far=100.0)
            mvp = proj @ view @ model
            cursor.mvp = spy.float4x4(mvp.flatten().tolist())

            rp.set_render_state(
                {
                    "vertex_buffers": [{"buffer": self.vbuf}],
                    "viewports": [spy.Viewport.from_size(w, h)],
                    "scissor_rects": [spy.ScissorRect.from_size(w, h)],
                }
            )

            rp.draw({"vertex_count": 3})

    def on_resize(self, width, height):
        self.depth_stencil_tex = self.device.create_texture(
            format=DEPTH_STENCIL_FORMAT,
            width=width,
            height=height,
            usage=spy.TextureUsage.depth_stencil,
        )
