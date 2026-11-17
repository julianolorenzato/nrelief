import definitions;

// Both modes draw geometry in object space (the mesh's own coordinates); `model` maps it to world.
//   Standard: the mesh itself.
//   Relief:   the mesh bounding box [boxMin, boxMax]; per fragment the view ray is marched through
//             it looking for the solid described by the baked depth layers (src/nrelief/gfx/bake.py).
float4x4 model;
float4x4 view;
float4x4 proj;
float4x4 normal; // inverse transpose of model (only the upper 3x3 is used)
RenderMode rmode;

// Standard
Texture2D colorTex;
SamplerState samp;

// Relief. The bake works in box-normalized space q = (p - boxMin) / (boxMax - boxMin), in [0, 1]^3.
// For each axis a the bake looked along a; texel (s, t) covers (q[(a + 1) % 3], q[(a + 2) % 3]).
//   depthMaps (layer = axis): RGBA = depths q[a] of hits 0..3, sorted ascending; inf = no hit.
//   colorMaps (layer = axis * HITS + hit): albedo of that hit.
//   normalMaps (same layers): smooth object-space normal of that hit, RGB = n * 0.5 + 0.5.
float3 camPos;   // camera position in object space
float3 boxMin;   // mesh bounding box, object space
float3 boxMax;
int linearSteps; // number of uniform steps of the ray search
Texture2DArray<float4> depthMaps;
Texture2DArray<float4> colorMaps;
Texture2DArray<float4> normalMaps;

static const int HITS = 4;
static const int REFINE_STEPS = 10;               // bisections after the uniform search
static const float DEPTH_TOLERANCE_TEXELS = 4.0;  // depth gap, in texel widths, still taken as the same surface
static const float3 LIGHT_DIR = normalize(float3(0.5, 1.0, 0.3)); // direction *towards* the light
static const float AMBIENT = 0.15;

[shader("vertex")]
VSOut vertexMain(VSIn in)
{
    VSOut out;
    out.clip = mul(proj, mul(view, mul(model, float4(in.pos, 1.0))));
    out.nrm = mul((float3x3)normal, in.nrm); // world-space normal
    out.uv = in.uv;
    out.objPos = in.pos;
    return out;
}

/// Integer texel of the axis' depth map that holds `q`.
int2 texelOf(int axis, float3 q)
{
    uint w, h, layers;
    depthMaps.GetDimensions(w, h, layers);
    float2 st = float2(q[(axis + 1) % 3], q[(axis + 2) % 3]);
    return clamp(int2(st * float2(w, h)), int2(0), int2(w, h) - 1);
}

/// Depths (along `axis`) of the surfaces the axis-aligned ray through `q` hits.
float4 sampleHits(int axis, float3 q)
{
    return depthMaps.Load(int4(texelOf(axis, q), axis, 0));
}

/// `q` is inside the solid when, along every axis, an odd number of hits lie before it.
bool insideSolid(float3 q)
{
    for (int axis = 0; axis < 3; axis++)
    {
        float4 hits = sampleHits(axis, q);
        int before = 0;
        for (int k = 0; k < HITS; k++)
            before += hits[k] < q[axis] ? 1 : 0; // inf (no hit) never counts
        if ((before & 1) == 0)
            return false;
    }
    return true;
}

/// Ray / bounding box slab test in object space. Returns (tNear, tFar); tFar < tNear means a miss.
float2 intersectBox(float3 origin, float3 dir)
{
    float3 inv = 1.0 / dir;
    float3 t0 = (boxMin - origin) * inv;
    float3 t1 = (boxMax - origin) * inv;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    return float2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

/// Uniform-step search for the first ray parameter in [tStart, tEnd] whose point is inside the
/// solid, then a binary refinement between it and the previous (outside) step, so the hit lands
/// on the surface instead of up to one step inside it.
bool marchLinear(float3 origin, float3 dir, float tStart, float tEnd, out float hitT)
{
    int steps = max(linearSteps, 1);
    float dt = (tEnd - tStart) / float(steps);
    float prevT = tStart;
    for (int i = 0; i <= steps; i++)
    {
        float t = tStart + dt * float(i);
        if (insideSolid(origin + dir * t))
        {
            // At i == 0 the ray starts inside (camera inside the solid): nothing to refine.
            float lo = (i == 0) ? t : prevT;
            float hi = t;
            for (int k = 0; k < REFINE_STEPS; k++)
            {
                float mid = 0.5 * (lo + hi);
                if (insideSolid(origin + dir * mid))
                    hi = mid;
                else
                    lo = mid;
            }
            hitT = hi;
            return true;
        }
        prevT = t;
    }
    hitT = tEnd;
    return false;
}

/// Layer (axis * HITS + hit) whose stored depth is closest to `q` along its axis; -1 if none.
int bestLayer(float3 q)
{
    int best = -1;
    float bestErr = 1e9;
    for (int axis = 0; axis < 3; axis++)
    {
        float4 hits = sampleHits(axis, q);
        for (int k = 0; k < HITS; k++)
        {
            float err = abs(q[axis] - hits[k]); // inf for a missing hit
            if (err < bestErr)
            {
                bestErr = err;
                best = axis * HITS + k;
            }
        }
    }
    return best;
}

/// Albedo and object-space normal of `layer` (axis * HITS + hit) at `q`.
///
/// The four nearest texels are blended bilinearly, but only those whose stored hit lies at the
/// depth of `q`: next to a depth discontinuity (a fold, a silhouette) the neighbouring texel
/// belongs to a different surface, and blending it in leaves dark and bright lines along the
/// contour. If no texel qualifies, the nearest one is used as is.
void sampleHit(int layer, float3 q, out float3 albedo, out float3 nrm)
{
    int axis = layer / HITS;
    int hit = layer % HITS;
    uint w, h, layers;
    depthMaps.GetDimensions(w, h, layers);
    int2 size = int2(w, h);
    float2 st = float2(q[(axis + 1) % 3], q[(axis + 2) % 3]) * float2(size) - 0.5;
    int2 base = int2(floor(st));
    float2 f = st - float2(base);
    float tol = DEPTH_TOLERANCE_TEXELS / float(w);

    float3 c = float3(0.0);
    float3 n = float3(0.0);
    float total = 0.0;
    for (int j = 0; j < 4; j++)
    {
        int2 o = int2(j & 1, j >> 1);
        int2 p = clamp(base + o, int2(0), size - 1);
        float wt = (o.x == 1 ? f.x : 1.0 - f.x) * (o.y == 1 ? f.y : 1.0 - f.y);
        float d = depthMaps.Load(int4(p, axis, 0))[hit];
        if (wt <= 0.0 || !(abs(d - q[axis]) <= tol)) // also rejects inf (no hit)
            continue;
        c += wt * colorMaps.Load(int4(p, layer, 0)).rgb;
        n += wt * (normalMaps.Load(int4(p, layer, 0)).rgb * 2.0 - 1.0);
        total += wt;
    }
    if (total > 0.0)
    {
        albedo = c / total;
        nrm = normalize(n);
    }
    else
    {
        int2 p = texelOf(axis, q);
        albedo = colorMaps.Load(int4(p, layer, 0)).rgb;
        nrm = normalize(normalMaps.Load(int4(p, layer, 0)).rgb * 2.0 - 1.0);
    }
}

[shader("fragment")]
PSOut fragmentMain(VSOut data)
{
    PSOut o;
    switch (rmode)
    {
    case RenderMode::Standard: {
        // Lambert diffuse + constant ambient.
        float3 n = normalize(data.nrm); // interpolation denormalizes it
        float diffuse = max(dot(n, LIGHT_DIR), 0.0);
        // OBJ uvs have V pointing up, but image row 0 is the top: flip V.
        float2 uv = float2(data.uv.x, 1.0 - data.uv.y);
        float3 albedo = colorTex.Sample(samp, uv).rgb;
        o.color = float4(albedo * (AMBIENT + diffuse), 1.0);
        o.depth = data.clip.z; // rasterized depth, unchanged
        return o;
    }

    case RenderMode::Relief: {
        // The box is drawn with its back faces, so this also works with the camera inside it.
        float3 dir = normalize(data.objPos - camPos);
        float2 tt = intersectBox(camPos, dir);
        float tStart = max(tt.x, 0.0);
        float tEnd = tt.y;
        if (tEnd < tStart)
            discard;

        // Same ray in box-normalized space: q(t) = q0 + dq * t, with the same t as in object space.
        float3 extent = max(boxMax - boxMin, float3(1e-9));
        float3 q0 = (camPos - boxMin) / extent;
        float3 dq = dir / extent;

        float hitT;
        if (!marchLinear(q0, dq, tStart, tEnd, hitT))
            discard;

        float3 q = saturate(q0 + dq * hitT);
        int layer = bestLayer(q);
        if (layer < 0)
            discard;

        float3 albedo, nObj;
        sampleHit(layer, q, albedo, nObj);

        // Same Lambert diffuse + ambient as Standard.
        float3 n = normalize(mul((float3x3)normal, nObj));
        float diffuse = max(dot(n, LIGHT_DIR), 0.0);

        float3 hitObj = camPos + dir * hitT;
        float4 clip = mul(proj, mul(view, mul(model, float4(hitObj, 1.0))));
        o.color = float4(albedo * (AMBIENT + diffuse), 1.0);
        o.depth = clip.z / clip.w; // clip-space z is already in [0, 1] (camera.perspective)
        return o;
    }

    default:
        o.color = float4(1.0, 1.0, 1.0, 1.0);
        o.depth = data.clip.z;
        return o;
    }
}
