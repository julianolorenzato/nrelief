from dataclasses import dataclass
from pathlib import Path

import numpy as np
import slangpy as spy

from .camera import OrbitCamera
from .mesh import Mesh

TITLE = "Neural Relief Mapping"
SHADERS_DIR = Path(__file__).parents[3] / "shaders"
SHADER_PROGRAM_ENTRYPOINT = "simple.slang"
SURFACE_FORMAT = spy.Format.bgra8_unorm_srgb
DEPTH_STENCIL_FORMAT = spy.Format.d32_float_s8_uint
INITIAL_WIDTH = 800
INITIAL_HEIGHT = 600


@dataclass
class DragState:
    """Mouse-drag state used to rotate the camera."""

    active: bool = False
    last_pos: tuple[float, float] = (0.0, 0.0)


class Viewer(spy.AppWindow):
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

        mesh = Mesh(obj_path)

        self.vertex_count = len(mesh.verts)
        self.vbuf = self.device.create_buffer(usage=spy.BufferUsage.vertex_buffer, data=mesh.verts)

        self.color_tex = self.device.create_texture(
            format=spy.Format.rgba8_unorm_srgb,  # color data -> sRGB
            width=mesh.albedo.shape[1],
            height=mesh.albedo.shape[0],
            usage=spy.TextureUsage.shader_resource,
            data=mesh.albedo,
        )
        self.sampler = self.device.create_sampler()  # linear filtering, wrap addressing

        input_layout = self.device.create_input_layout(
            input_elements=[
                {
                    "semantic_name": "POSITION",
                    "semantic_index": 0,
                    "format": spy.Format.rgb32_float,
                    "offset": 0,
                    "buffer_slot_index": 0,
                },
                {
                    "semantic_name": "NORMAL",
                    "semantic_index": 0,
                    "format": spy.Format.rgb32_float,
                    "offset": 12,
                    "buffer_slot_index": 0,
                },
                {
                    "semantic_name": "TEXCOORD",
                    "semantic_index": 0,
                    "format": spy.Format.rg32_float,
                    "offset": 24,
                    "buffer_slot_index": 0,
                },
            ],
            vertex_streams=[{"stride": 32, "slot_class": spy.InputSlotClass.per_vertex}],
        )

        # Orbit around the center of the mesh bounding box.
        lo, hi = mesh.pos.min(axis=0), mesh.pos.max(axis=0)
        self.camera = OrbitCamera(
            target=(lo + hi) / 2.0, radius=float(np.linalg.norm(hi - lo)) / 2.0
        )
        self.drag = DragState()

        self.pipeline = self.device.create_render_pipeline(
            program=program,
            input_layout=input_layout,
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
            cursor.colorTex = self.color_tex
            cursor.samp = self.sampler
            mvp = self.camera.view_proj(w / h)
            cursor.mvp = spy.float4x4(mvp.flatten().tolist())

            rp.set_render_state(
                {
                    "vertex_buffers": [{"buffer": self.vbuf}],
                    "viewports": [spy.Viewport.from_size(w, h)],
                    "scissor_rects": [spy.ScissorRect.from_size(w, h)],
                }
            )

            rp.draw({"vertex_count": self.vertex_count})

    def on_mouse_event(self, event: spy.MouseEvent):
        if event.type == spy.MouseEventType.button_down and event.button == spy.MouseButton.left:
            self.drag.active = True
            self.drag.last_pos = (event.pos.x, event.pos.y)
        elif event.type == spy.MouseEventType.button_up and event.button == spy.MouseButton.left:
            self.drag.active = False
        elif event.type == spy.MouseEventType.move and self.drag.active:
            x, y = event.pos.x, event.pos.y
            self.camera.rotate(x - self.drag.last_pos[0], y - self.drag.last_pos[1])
            self.drag.last_pos = (x, y)
        elif event.type == spy.MouseEventType.scroll:
            self.camera.zoom(event.scroll.y)

    def on_resize(self, width, height):
        self.depth_stencil_tex = self.device.create_texture(
            format=DEPTH_STENCIL_FORMAT,
            width=width,
            height=height,
            usage=spy.TextureUsage.depth_stencil,
        )
