import numba as nb
import numpy as np

from .mesh import Mesh


# Hits mais próximos que isso, em fração da espessura da caixa, são a mesma superfície
# (triângulos que compartilham uma aresta) ou uma dobra de espessura zero.
HIT_MERGE_EPS = 2e-4


@nb.njit(cache=True)
def _remove(zs, face, bary, facing, mixed, n, i, m):
    """Remove os hits i..i+m-1 da lista (de tamanho n) de um raio; devolve o novo tamanho."""
    for j in range(i, n - m):
        zs[j] = zs[j + m]
        face[j] = face[j + m]
        bary[j, 0] = bary[j + m, 0]
        bary[j, 1] = bary[j + m, 1]
        facing[j] = facing[j + m]
        mixed[j] = mixed[j + m]
    for j in range(n - m, n):
        zs[j] = np.inf
        face[j] = -1
    return n - m


@nb.njit(cache=True)
def _fill_thinnest_gap(zs, face, bary, facing, mixed, n):
    """Remove o vão de ar mais fino, o par (saída, próxima entrada) mais próximo, juntando as
    duas fatias de sólido. Os demais hits mantêm a paridade dentro/fora: só se preenche ar,
    nunca se apaga sólido (os outros dois eixos ainda podem rejeitar o ar preenchido, mas nada
    traz de volta uma fatia apagada). Devolve o novo tamanho (n se não há vão, n < 3)."""
    best = -1
    for i in range(1, n - 1, 2):
        if best < 0 or zs[i + 1] - zs[i] < zs[best + 1] - zs[best]:
            best = i
    if best < 0:
        return n
    return _remove(zs, face, bary, facing, mixed, n, best, 2)


@nb.njit(cache=True)
def _raster_axis(tris, resolution, max_hits):
    """Rasteriza os triângulos (T, 3, [x, y, z]) e guarda, por texel, até max_hits hits
    ordenados por z e, de cada um, o triângulo atingido, sua orientação ao longo do eixo
    (+1/-1) e as coordenadas baricêntricas (l0, l1) do centro do texel nele (texel vazio:
    z = inf, face -1).

    Hits na mesma profundidade (até HIT_MERGE_EPS) viram um só: se as faces apontam para o
    mesmo lado, é a mesma superfície vista pelos dois triângulos de uma aresta; se apontam
    para lados opostos, o raio só raspa o sólido (espessura zero) e o hit é descartado.

    Um raio com hits demais é reduzido preenchendo seus vãos de ar mais finos
    (_fill_thinnest_gap), e não cortando os mais distantes: isso apagaria as fatias de sólido
    do fim do raio."""
    # Candidatos por texel antes da redução final, com um slot de folga: um hit é inserido
    # antes de a lista voltar a cand.
    cand = 2 * max_hits
    zs = np.full((resolution, resolution, cand + 1), np.inf, np.float32)
    cface = np.full((resolution, resolution, cand + 1), -1, np.int32)
    cbary = np.zeros((resolution, resolution, cand + 1, 2), np.float32)
    facing = np.zeros((resolution, resolution, cand + 1), np.int8)
    mixed = np.zeros((resolution, resolution, cand + 1), np.bool_)
    count = np.zeros((resolution, resolution), np.int32)
    for t in range(tris.shape[0]):
        ax, ay, az = tris[t, 0]
        bx, by, bz = tris[t, 1]
        cx, cy, cz = tris[t, 2]
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

                # Inserção ordenada (há sempre um slot livre: n <= cand).
                k = n
                while k > 0 and zs[y, x, k - 1] > z:
                    zs[y, x, k] = zs[y, x, k - 1]
                    cface[y, x, k] = cface[y, x, k - 1]
                    cbary[y, x, k] = cbary[y, x, k - 1]
                    facing[y, x, k] = facing[y, x, k - 1]
                    mixed[y, x, k] = mixed[y, x, k - 1]
                    k -= 1
                zs[y, x, k] = z
                cface[y, x, k] = t
                cbary[y, x, k, 0] = l0
                cbary[y, x, k, 1] = l1
                facing[y, x, k] = f
                mixed[y, x, k] = False
                n += 1

                if n > cand:
                    # Um hit rasante vale um par de paridade nula: tirá-lo mantém a paridade.
                    dropped = False
                    for i in range(n):
                        if mixed[y, x, i]:
                            n = _remove(zs[y, x], cface[y, x], cbary[y, x], facing[y, x],
                                        mixed[y, x], n, i, 1)
                            dropped = True
                            break
                    if not dropped:
                        n = _fill_thinnest_gap(zs[y, x], cface[y, x], cbary[y, x],
                                               facing[y, x], mixed[y, x], n)
                count[y, x] = n

    # Descarta os hits rasantes e reduz cada raio a max_hits preenchendo os vãos mais finos.
    layers = np.full((resolution, resolution, max_hits), np.inf, np.float32)
    faces = np.full((resolution, resolution, max_hits), -1, np.int32)
    facings = np.zeros((resolution, resolution, max_hits), np.int8)
    bary = np.zeros((resolution, resolution, max_hits, 2), np.float32)
    for y in range(resolution):
        for x in range(resolution):
            n = 0
            for i in range(count[y, x]):
                if mixed[y, x, i]:
                    continue
                zs[y, x, n] = zs[y, x, i]
                cface[y, x, n] = cface[y, x, i]
                cbary[y, x, n] = cbary[y, x, i]
                facing[y, x, n] = facing[y, x, i]
                n += 1
            while n > max_hits:
                reduced = _fill_thinnest_gap(zs[y, x], cface[y, x], cbary[y, x],
                                             facing[y, x], mixed[y, x], n)
                if reduced == n:  # sem vão para preencher (max_hits < 2): corta
                    n = max_hits
                    break
                n = reduced
            for k in range(n):
                layers[y, x, k] = zs[y, x, k]
                faces[y, x, k] = cface[y, x, k]
                facings[y, x, k] = facing[y, x, k]
                bary[y, x, k] = cbary[y, x, k]
    return layers, faces, facings, bary


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
    #                       "albedo": (res, res, max_hits, 4) uint8 ou None,
    #                       "normal": (res, res, max_hits, 4) uint8}
    result = []

    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3

        # Guard against degenerated meshes (plane mesh in this axis)
        if half[u] <= 0 or half[v] <= 0:
            layers = np.full((resolution, resolution, max_hits), np.inf, np.float32)
            faces = np.full((resolution, resolution, max_hits), -1, np.int32)
            facings = np.zeros((resolution, resolution, max_hits), np.int8)
            bary = np.zeros((resolution, resolution, max_hits, 2), np.float32)
            result.append(_axis_result(mesh, axis, layers, faces, facings, bary))
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

        # As MAX_HITS camadas mais próximas por texel, ordenadas. Texel vazio = inf.
        layers, faces, facings, bary = _raster_axis(
            np.ascontiguousarray(tris, dtype=np.float64), resolution, max_hits
        )
        result.append(_axis_result(mesh, axis, layers, faces, facings, bary))

    return result


def _axis_result(mesh: Mesh, axis, layers, faces, facings, bary):
    """Interpola, em cada hit, os atributos dos cantos do triângulo atingido: o UV (para
    amostrar o albedo) e a normal suave da malha, codificada em RGBA 8-bit como n * 0.5 + 0.5
    no espaço do objeto (alpha 0 = sem hit)."""
    valid = faces >= 0
    f = np.where(valid, faces, 0)
    b0, b1 = bary[..., 0], bary[..., 1]
    w = np.stack([b0, b1, 1.0 - b0 - b1], axis=-1)[..., None]  # (res, res, hits, 3, 1)

    # Projeção ortográfica: interpolação afim dos atributos é exata.
    uvs = (mesh.uv.reshape(-1, 3, 2)[f] * w).sum(axis=-2)
    nrm = (mesh.nrm.reshape(-1, 3, 3)[f] * w).sum(axis=-2)

    # Normal degenerada (cantos com normais opostas): usa a direção do eixo para o lado
    # para onde a face aponta.
    length = np.linalg.norm(nrm, axis=-1, keepdims=True)
    fallback = np.zeros_like(nrm)
    fallback[..., axis] = facings
    nrm = np.where(length > 1e-12, nrm / np.maximum(length, 1e-12), fallback)

    normal = np.zeros((*faces.shape, 4), np.uint8)
    normal[..., :3] = np.rint((nrm * 0.5 + 0.5) * 255.0).astype(np.uint8)
    normal[..., 3] = 255
    normal[~valid] = 0

    return {
        "depth": layers,
        "albedo": None if mesh.albedo is None else _sample(mesh.albedo, uvs, valid),
        "normal": normal,
    }
