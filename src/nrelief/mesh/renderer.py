from pathlib import Path

import numpy as np
import slangpy as spy

from .obj import load_obj

SHADERS = Path(__file__).parents[3] / "shaders"
DEPTH_FORMAT: spy.Format = spy.Format.d32_float

class MeshRenderer:
    """Device-side state: mesh buffers, MLP weights, pipeline. Draws into any color/depth view."""

    device: spy.Device
    vertex_count: int
    vertex_buffer: spy.Buffer
    weights: spy.Buffer
    pipeline: spy.RenderPipeline    

    def __init__(
        self,
        device: spy.Device,
        obj_path: str | Path,
        weights_path: str | Path,
        color_format: spy.Format,
    ) -> None:
        self.device = device
        pos, nrm, uv = load_obj(obj_path)
        self.vertex_count = len(pos)
        verts = np.concatenate([pos, nrm, uv], axis=1).astype(np.float32)  # stride 32
        self.vertex_buffer = device.create_buffer(
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.vertex_buffer,
            data=verts,
        )
        weights = np.fromfile(weights_path, dtype=np.float32)
        self.weights = device.create_buffer(
            element_count=weights.size, struct_size=4,
            usage=spy.BufferUsage.shader_resource, data=weights,
        )
        program = device.load_program(str(SHADERS / "mesh.slang"), ["vertexMain", "fragmentMain"])
        layout = device.create_input_layout(
            input_elements=[
                {"semantic_name": "POSITION", "semantic_index": 0, "format": spy.Format.rgb32_float, "offset": 0},
                {"semantic_name": "NORMAL", "semantic_index": 0, "format": spy.Format.rgb32_float, "offset": 12},
                {"semantic_name": "TEXCOORD", "semantic_index": 0, "format": spy.Format.rg32_float, "offset": 24},
            ],
            vertex_streams=[{"stride": 32}],
        )
        self.pipeline = device.create_render_pipeline(
            program=program,
            input_layout=layout,
            targets=[{"format": color_format}],
            depth_stencil={"format": DEPTH_FORMAT, "depth_test_enable": True, "depth_write_enable": True,
                           "depth_func": spy.ComparisonFunc.less},
            rasterizer={"cull_mode": spy.CullMode.none},  # TODO: enable culling once winding is verified
        )

    def draw(
        self,
        encoder: spy.CommandEncoder,
        color_tex: spy.Texture,
        depth_tex: spy.Texture,
        mvp: spy.float4x4,
    ) -> None:
        w, h = color_tex.width, color_tex.height
        rp: spy.RenderPassDescDict = {
            "color_attachments": [{
                "view": color_tex.create_view({}),
                "load_op": spy.LoadOp.clear, "store_op": spy.StoreOp.store,
                "clear_value": [0.08, 0.08, 0.1, 1.0],
            }],
            "depth_stencil_attachment": {
                "view": depth_tex.create_view({}),
                "depth_load_op": spy.LoadOp.clear, "depth_store_op": spy.StoreOp.store,
                "depth_clear_value": 1.0,
            },
        }
        with encoder.begin_render_pass(rp) as p:
            p.set_render_state({
                "vertex_buffers": [self.vertex_buffer],
                "viewports": [spy.Viewport.from_size(w, h)],
                "scissor_rects": [spy.ScissorRect.from_size(w, h)],
            })
            cursor = spy.ShaderCursor(p.bind_pipeline(self.pipeline))
            cursor.mvp = mvp
            cursor.weights = self.weights
            p.draw({"vertex_count": self.vertex_count})
