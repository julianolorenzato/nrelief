from pathlib import Path

import numpy as np
import open3d as o3d


class Mesh:
    def __init__(self, path: Path):
        model = o3d.io.read_triangle_model(path)  # ty: ignore[invalid-argument-type]
        if not model.meshes:
            raise ValueError(f"failed to load {path}: no triangle meshes found")

        # One vertex per triangle corner (non-indexed): every 3 consecutive
        # vertices form one triangle. Interleaved as pos(3) nrm(3) uv(2).
        pos_parts, nrm_parts, uv_parts = [], [], []
        for info in model.meshes:
            mesh = info.mesh
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            tris = np.asarray(mesh.triangles).reshape(-1)
            pos = np.asarray(mesh.vertices, dtype=np.float32)[tris]
            pos_parts.append(pos)
            nrm_parts.append(np.asarray(mesh.vertex_normals, dtype=np.float32)[tris])
            if mesh.has_triangle_uvs():
                uv_parts.append(np.asarray(mesh.triangle_uvs, dtype=np.float32))  # one per corner
            else:
                uv_parts.append(np.zeros((len(tris), 2), dtype=np.float32))

        self.pos = np.concatenate(pos_parts, dtype=np.float32)
        self.nrm = np.concatenate(nrm_parts, dtype=np.float32)
        self.uv = np.concatenate(uv_parts, dtype=np.float32)
        self.verts = np.ascontiguousarray(
            np.hstack([self.pos, self.nrm, self.uv]), dtype=np.float32
        )

        # Textures of the first mesh's material, already decoded by Open3D.
        material = model.materials[model.meshes[0].material_idx]
        # Each is None when the material has no such texture (e.g. cube.obj).
        self.albedo: np.ndarray | None = self._rgba(material.albedo_img)
        self.normal: np.ndarray | None = self._rgba(material.normal_img)

    @staticmethod
    def _rgba(img) -> np.ndarray | None:
        """Return img as a contiguous HxWx4 uint8 array, or None if absent."""
        img = np.asarray(img)
        if img.ndim != 3 or img.dtype != np.uint8:
            return None
        if img.shape[2] == 3:
            img = np.dstack([img, np.full(img.shape[:2], 255, dtype=np.uint8)])
        return np.ascontiguousarray(img)
