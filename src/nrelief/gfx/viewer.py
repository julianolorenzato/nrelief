from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import slangpy as spy
from slangpy import math as sm

from .bake import bake
from .camera import OrbitCamera
from .mesh import Mesh

TITLE = "Neural Relief Mapping"
SHADERS_DIR = Path(__file__).parents[3] / "shaders"
SHADER_PROGRAM_ENTRYPOINT = "gfx.slang"
RELIEF_LINEAR_STEPS = 256
SURFACE_FORMAT = spy.Format.bgra8_unorm_srgb
DEPTH_STENCIL_FORMAT = spy.Format.d32_float_s8_uint
INITIAL_WIDTH = 800
INITIAL_HEIGHT = 600


class RenderMode(Enum):
    # Values must match `enum RenderMode` in shaders/definitions.slang.
    RELIEF = 0
    STANDARD = 1

    def next(self):
        members = list(type(self))
        return members[(members.index(self) + 1) % len(members)]


def box_vertices(lo, hi) -> np.ndarray:
    """36 vertices (12 triangles, counter-clockwise seen from outside) of the box [lo, hi],
    in the mesh vertex layout pos(3) nrm(3) uv(2); normal and uv are unused (zero)."""
    c = np.array(
        [[hi[a] if (i >> a) & 1 else lo[a] for a in range(3)] for i in range(8)], dtype=np.float32
    )
    quads = [
        (0, 4, 6, 2),
        (1, 3, 7, 5),  # -X, +X
        (0, 1, 5, 4),
        (2, 6, 7, 3),  # -Y, +Y
        (0, 2, 3, 1),
        (4, 5, 7, 6),
    ]  # -Z, +Z
    idx = [i for a, b, cc, d in quads for i in (a, b, cc, a, cc, d)]
    return np.ascontiguousarray(np.hstack([c[idx], np.zeros((36, 5), np.float32)]))


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

        self.rmode: RenderMode = RenderMode.STANDARD
        program = self.device.load_program(
            str(SHADERS_DIR / SHADER_PROGRAM_ENTRYPOINT), ["vertexMain", "fragmentMain"]
        )

        mesh = Mesh(obj_path)

        relief_textures_data = bake(mesh=mesh, max_hits=4, resolution=1024)

        self.depth_maps: spy.Texture = self.device.create_texture(
            type=spy.TextureType.texture_2d_array,
            format=spy.Format.rgba32_float,
            height=1024,
            width=1024,
            array_length=3,
            usage=spy.TextureUsage.shader_resource,
        )

        self.color_maps: spy.Texture = self.device.create_texture(
            type=spy.TextureType.texture_2d_array,
            format=spy.Format.rgba8_unorm_srgb,
            height=1024,
            width=1024,
            array_length=12,
            usage=spy.TextureUsage.shader_resource,
        )

        for i in range(3):
            self.depth_maps.copy_from_numpy(relief_textures_data[i]["depth"], layer=i)

            # Albedo (res, res, max_hits, 4): uma camada do array por (eixo, hit).
            albedo = relief_textures_data[i]["albedo"]
            if albedo is not None:
                for k in range(albedo.shape[2]):
                    self.color_maps.copy_from_numpy(
                        np.ascontiguousarray(albedo[:, :, k]), layer=i * 4 + k
                    )

        # Vertices stay in object space (the mesh's own coordinates); model places the object in
        # the world. Identity for now: object space == world space.
        self.model_mat = spy.float4x4.identity()
        self.model_inv_mat = sm.inverse(self.model_mat)
        self.normal_mat = sm.transpose(self.model_inv_mat)

        self.vertex_count = len(mesh.verts)
        self.vbuf = self.device.create_buffer(usage=spy.BufferUsage.vertex_buffer, data=mesh.verts)

        # Mesh bounding box in object space: the same box the bake normalizes into [0, 1]^3.
        lo, hi = mesh.pos.min(axis=0), mesh.pos.max(axis=0)
        self.box_min, self.box_max = spy.float3(*lo.tolist()), spy.float3(*hi.tolist())

        if mesh.albedo is None:
            raise ValueError(f"{obj_path}: mesh has no albedo (color) texture")
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

        # Orbit around the center of the mesh bounding box (world space).
        self.camera = OrbitCamera(
            target=(lo + hi) / 2.0, radius=float(np.linalg.norm(hi - lo)) / 2.0
        )
        self.drag = DragState()

        # Relief: a bounding box da malha é a geometria proxy; o shader marcha o raio dentro dela.
        # Mesmo programa e layout de vértice da malha; muda só o culling.
        box = box_vertices(lo, hi)
        self.box_vertex_count = len(box)
        self.box_vbuf = self.device.create_buffer(usage=spy.BufferUsage.vertex_buffer, data=box)
        self.relief_pipeline = self.device.create_render_pipeline(
            program=program,
            input_layout=input_layout,
            targets=[{"format": SURFACE_FORMAT}],
            # Faces de trás: funciona também com a câmera dentro da caixa.
            rasterizer={"cull_mode": spy.CullMode.front, "fill_mode": spy.FillMode.solid},
            depth_stencil={
                "format": DEPTH_STENCIL_FORMAT,
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": spy.ComparisonFunc.less,
            },
        )

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

        match self.rmode:
            case RenderMode.STANDARD:
                pipeline, vbuf, count = self.pipeline, self.vbuf, self.vertex_count
            case RenderMode.RELIEF:
                pipeline, vbuf, count = self.relief_pipeline, self.box_vbuf, self.box_vertex_count

        with cmd_enc.begin_render_pass(rp_desc) as rp:
            cursor = spy.ShaderCursor(rp.bind_pipeline(pipeline))
            cursor.rmode = self.rmode.value
            cursor.model = self.model_mat
            cursor.view = self.camera.view()
            cursor.proj = self.camera.proj(w / h)
            cursor.normal = self.normal_mat
            cursor.colorTex = self.color_tex
            cursor.samp = self.sampler
            cursor.camPos = sm.mul(self.model_inv_mat, spy.float4(self.camera.eye, 1.0)).xyz
            cursor.boxMin = self.box_min
            cursor.boxMax = self.box_max
            cursor.linearSteps = RELIEF_LINEAR_STEPS
            cursor.depthMaps = self.depth_maps
            cursor.colorMaps = self.color_maps

            rp.set_render_state(
                {
                    "vertex_buffers": [{"buffer": vbuf}],
                    "viewports": [spy.Viewport.from_size(w, h)],
                    "scissor_rects": [spy.ScissorRect.from_size(w, h)],
                }
            )

            rp.draw({"vertex_count": count})

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if event.type == spy.KeyboardEventType.key_press and event.key == spy.KeyCode.r:
            self.rmode = self.rmode.next()

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
