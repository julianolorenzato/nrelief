import numba as nb
import numpy as np

from .mesh import Mesh


# Hits mais próximos que isso, em fração da espessura da caixa, são a mesma superfície
# (triângulos que compartilham uma aresta) ou uma dobra de espessura zero.
HIT_MERGE_EPS = 2e-4


@nb.njit(cache=True)
def _raster_axis(tris, tri_uv, resolution, max_hits):
    """Rasteriza os triângulos (T, 3, [x, y, z]) e guarda, por texel, os max_hits menores z
    e o UV de textura interpolado de cada um desses hits.

    Hits na mesma profundidade (até HIT_MERGE_EPS) viram um só: se as faces apontam para o
    mesmo lado, é a mesma superfície vista pelos dois triângulos de uma aresta; se apontam
    para lados opostos, o raio só raspa o sólido (espessura zero) e o hit é descartado."""
    # Candidatos por texel antes de descartar os hits rasantes, para que eles não tomem o
    # lugar de superfícies reais entre os max_hits mais próximos.
    cand = 2 * max_hits
    zs = np.full((resolution, resolution, cand), np.inf, np.float32)
    cuvs = np.zeros((resolution, resolution, cand, 2), np.float32)
    facing = np.zeros((resolution, resolution, cand), np.int8)
    mixed = np.zeros((resolution, resolution, cand), np.bool_)
    count = np.zeros((resolution, resolution), np.int32)
    for t in range(tris.shape[0]):
        ax, ay, az = tris[t, 0]
        bx, by, bz = tris[t, 1]
        cx, cy, cz = tris[t, 2]
        au, av = tri_uv[t, 0]
        bu, bv = tri_uv[t, 1]
        cu, cv = tri_uv[t, 2]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-12:
            continue  # triângulo de lado (paralelo ao raio)
        # (u, v, axis) é uma permutação cíclica de (x, y, z), então o sinal da área projetada
        # é o sinal da normal da face ao longo do eixo: +1 se ela aponta para o lado positivo.
        f = 1 if area > 0 else -1

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

                # Mesma profundidade de um hit já guardado: funde com ele.
                n = count[y, x]
                merged = False
                for i in range(n):
                    if abs(zs[y, x, i] - z) < HIT_MERGE_EPS:
                        if facing[y, x, i] != f:
                            mixed[y, x, i] = True
                        merged = True
                        break
                if merged:
                    continue

                # Inserção ordenada nos cand slots; cheio, o mais distante sai.
                if n == cand and z >= zs[y, x, cand - 1]:
                    continue
                k = min(n, cand - 1)
                while k > 0 and zs[y, x, k - 1] > z:
                    zs[y, x, k] = zs[y, x, k - 1]
                    cuvs[y, x, k] = cuvs[y, x, k - 1]
                    facing[y, x, k] = facing[y, x, k - 1]
                    mixed[y, x, k] = mixed[y, x, k - 1]
                    k -= 1
                zs[y, x, k] = z
                # Projeção ortográfica: interpolação afim do UV é exata.
                cuvs[y, x, k, 0] = l0 * au + l1 * bu + l2 * cu
                cuvs[y, x, k, 1] = l0 * av + l1 * bv + l2 * cv
                facing[y, x, k] = f
                mixed[y, x, k] = False
                count[y, x] = min(n + 1, cand)

    # Descarta os hits rasantes e fica com os max_hits mais próximos.
    layers = np.full((resolution, resolution, max_hits), np.inf, np.float32)
    uvs = np.zeros((resolution, resolution, max_hits, 2), np.float32)
    for y in range(resolution):
        for x in range(resolution):
            k = 0
            for i in range(count[y, x]):
                if mixed[y, x, i]:
                    continue
                if k == max_hits:
                    break
                layers[y, x, k] = zs[y, x, i]
                uvs[y, x, k] = cuvs[y, x, i]
                k += 1
    return layers, uvs


def _sample(tex, uvs, valid):
    """Amostra tex (H, W, 4) uint8 em uvs (..., 2), bilinear com wrap.

    Mesma convenção do viewer: V do OBJ aponta para cima, linha 0 da imagem é o topo.
    Onde valid é False o resultado é (0, 0, 0, 0).
    """
    h, w = tex.shape[:2]
    x = uvs[..., 0] * w - 0.5
    y = (1.0 - uvs[..., 1]) * h - 0.5
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x0 = x0.astype(np.int64) % w
    y0 = y0.astype(np.int64) % h
    x1, y1 = (x0 + 1) % w, (y0 + 1) % h
    t = tex.astype(np.float32)
    top = t[y0, x0] * (1 - fx) + t[y0, x1] * fx
    bot = t[y1, x0] * (1 - fx) + t[y1, x1] * fx
    out = np.rint(top * (1 - fy) + bot * fy).astype(np.uint8)
    out[~valid] = 0
    return out


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

    # Uma entrada por eixo: {"depth": (res, res, max_hits) float32,
    #                       "albedo"/"normal": (res, res, max_hits, 4) uint8 ou None}
    result = []

    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3

        # Guard against degenerated meshes (plane mesh in this axis)
        if half[u] <= 0 or half[v] <= 0:
            layers = np.full((resolution, resolution, max_hits), np.inf, np.float32)
            uvs = np.zeros((resolution, resolution, max_hits, 2), np.float32)
            result.append(_axis_result(mesh, layers, uvs))
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
        layers, uvs = _raster_axis(
            np.ascontiguousarray(tris, dtype=np.float64),
            np.ascontiguousarray(tri_uv, dtype=np.float64),
            resolution,
            max_hits,
        )
        result.append(_axis_result(mesh, layers, uvs))

    return result


def _axis_result(mesh: Mesh, layers, uvs):
    valid = np.isfinite(layers)
    return {
        "depth": layers,
        "albedo": None if mesh.albedo is None else _sample(mesh.albedo, uvs, valid),
        "normal": None if mesh.normal is None else _sample(mesh.normal, uvs, valid),
    }
