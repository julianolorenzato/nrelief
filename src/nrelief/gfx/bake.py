import numba as nb
import numpy as np

from .mesh import Mesh


@nb.njit(cache=True)
def _raster_axis(tris, resolution, max_hits):
    """Rasteriza os triângulos (T, 3, [x, y, z]) e guarda, por texel, os max_hits menores z."""
    layers = np.full((resolution, resolution, max_hits), np.inf, np.float32)
    for t in range(tris.shape[0]):
        ax, ay, az = tris[t, 0]
        bx, by, bz = tris[t, 1]
        cx, cy, cz = tris[t, 2]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-12:
            continue  # triângulo de lado (paralelo ao raio)

        min_x = max(0, int(np.floor(min(ax, bx, cx))))
        max_x = min(resolution - 1, int(np.ceil(max(ax, bx, cx))))
        min_y = max(0, int(np.floor(min(ay, by, cy))))
        max_y = min(resolution - 1, int(np.ceil(max(ay, by, cy))))

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                px, py = x + 0.5, y + 0.5  # centro do texel
                l0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
                l1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
                l2 = ((ax - px) * (by - py) - (ay - py) * (bx - px)) / area
                if min(l0, l1, l2) < -1e-4:
                    continue  # fora do triângulo
                z = l0 * az + l1 * bz + l2 * cz

                # Inserção ordenada nos max_hits slots.
                k = max_hits - 1
                if z >= layers[y, x, k]:
                    continue
                while k > 0 and layers[y, x, k - 1] > z:
                    layers[y, x, k] = layers[y, x, k - 1]
                    k -= 1
                layers[y, x, k] = z
    return layers


def bake(mesh: Mesh, resolution, max_hits):
    # Bounding Box Calculation
    lo = mesh.pos.min(axis=0)
    hi = mesh.pos.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)  # a value for half of the box in each axis

    # Moves all mesh vertices in order to put the
    # bounding box center at the space origin.
    # They now live in the [-half, +half] range, with
    # half[axis] possibly being a different scalar value for each axis.
    p = mesh.pos - center

    result = []  # uma entrada (res, res, MAX_HITS) por eixo

    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3

        # Guard against degenerated meshes (plane mesh in this axis)
        if half[u] <= 0 or half[v] <= 0:
            result.append(np.full((resolution, resolution, max_hits), np.inf, np.float32))
            continue

        # Coordinate transformations for x, y and z:
        # [-half, +half] -> [0, 2*half]     (add half)
        # [0, 2*half]    -> [0, 2]          (divide by half)
        # [0, 2]         -> [0, 1]          (divide by 2)
        # [0, 1]         -> [0, resolution] (multiply by resolution)
        tx = (p[:, u] + half[u]) / (2 * half[u]) * resolution
        ty = (p[:, v] + half[v]) / (2 * half[v]) * resolution
        z = (p[:, axis] + half[axis]) / (2 * half[axis])

        tris = np.stack([tx, ty, z], axis=1).reshape(
            -1, 3, 3
        )  # (T, 3 vértices, [x, y, z]); muda por eixo
        tri_uv = mesh.uv.reshape(-1, 3, 2)  # textura, igual para os 3 eixos

        # As MAX_HITS camadas mais próximas por texel, ordenadas. Texel vazio = inf.
        layers = _raster_axis(np.ascontiguousarray(tris, dtype=np.float64), resolution, max_hits)
        result.append(layers)

    return result
