import numpy as np

from .mesh import Mesh

MAX_HITS = 4


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def bake(mesh: Mesh, resolution):
    # Bounding Box Calculation
    lo = mesh.pos.min(axis=0)
    hi = mesh.pos.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) # a value for half of the box in each axis

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
            result.append(np.full((resolution, resolution, MAX_HITS), np.inf, np.float32))
            continue

        # Coordinate transformations for x, y and z:
        # [-half, +half] -> [0, 2*half]     (add half)
        # [0, 2*half]    -> [0, 2]          (divide by half)
        # [0, 2]         -> [0, 1]          (divide by 2)
        # [0, 1]         -> [0, resolution] (multiply by resolution)
        tx = (p[:, u] + half[u]) / (2 * half[u]) * resolution
        ty = (p[:, v] + half[v]) / (2 * half[v]) * resolution
        z = (p[:, axis] + half[axis]) / (2 * half[axis])

        tris = np.stack([tx, ty, z], axis=1).reshape(-1, 3, 3)  # (T, 3 vértices, [x, y, z]); muda por eixo
        tri_uv = mesh.uv.reshape(-1, 3, 2)  # textura, igual para os 3 eixos

        # Todos os z que atingem cada texel (y, x).
        hits = [[[] for _ in range(resolution)] for _ in range(resolution)]

        for tri in tris:
            tuv, tz = tri[:, :2], tri[:, 2]  # texels e profundidades dos 3 vértices
            area = cross2(tuv[1] - tuv[0], tuv[2] - tuv[0])
            if abs(area) < 1e-12:
                continue  # triângulo de lado (paralelo ao raio)

            min_x = max(0, int(np.floor(tuv[:, 0].min())))
            max_x = min(resolution - 1, int(np.ceil(tuv[:, 0].max())))
            min_y = max(0, int(np.floor(tuv[:, 1].min())))
            max_y = min(resolution - 1, int(np.ceil(tuv[:, 1].max())))

            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    c = np.array([x + 0.5, y + 0.5])  # centro do texel
                    l0 = cross2(tuv[1] - c, tuv[2] - c) / area
                    l1 = cross2(tuv[2] - c, tuv[0] - c) / area
                    l2 = cross2(tuv[0] - c, tuv[1] - c) / area
                    if min(l0, l1, l2) < -1e-4:
                        continue  # fora do triângulo
                    hits[y][x].append(l0 * tz[0] + l1 * tz[1] + l2 * tz[2])

        # As MAX_HITS camadas mais próximas, ordenadas. Texel vazio = inf.
        layers = np.full((resolution, resolution, MAX_HITS), np.inf, np.float32)
        for y in range(resolution):
            for x in range(resolution):
                zs = sorted(hits[y][x])[:MAX_HITS]
                layers[y, x, : len(zs)] = zs
        result.append(layers)

    return result
