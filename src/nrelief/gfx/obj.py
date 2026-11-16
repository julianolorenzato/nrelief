import numpy as np
import open3d as o3d


def load_obj(path):
    """Return flat, triangulated float32 arrays: positions (N,3), normals (N,3), uvs (N,2).

    Every 3 consecutive vertices form one triangle (no index dedup, so the
    index buffer is just arange(N)). Missing normals/uvs are zero-filled.
    """
    # read_triangle_model triangulates quads/polygons; read_triangle_mesh silently drops them.
    model = o3d.io.read_triangle_model(str(path))
    if not model.meshes:
        raise ValueError(f"failed to load {path}: no triangle meshes found")

    pos, nrm, uv = [], [], []
    for info in model.meshes:
        mesh = info.mesh
        tris = np.asarray(mesh.triangles).reshape(-1)  # (3T,) vertex indices
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        p = verts[tris]
        pos.append(p)
        nrm.append(
            np.asarray(mesh.vertex_normals, dtype=np.float32)[tris]
            if mesh.has_vertex_normals()
            else np.zeros_like(p)
        )
        uv.append(
            np.asarray(mesh.triangle_uvs, dtype=np.float32)  # already one per triangle corner
            if mesh.has_triangle_uvs()
            else np.zeros((len(p), 2), np.float32)
        )
    return np.concatenate(pos), np.concatenate(nrm), np.concatenate(uv)
