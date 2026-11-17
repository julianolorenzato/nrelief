import definitions;

// Both modes draw geometry in the local space [0, 1]^3 (the mesh bounding box, normalized per
// axis); `model` maps it back to world space with the mesh's real proportions.
//   Standard: the mesh itself, normalized into [0, 1]^3.
//   Relief:   the unit cube; per fragment the view ray is marched through it looking for the solid
//             described by the baked depth layers (see src/nrelief/gfx/bake.py).
float4x4 model;
float4x4 view;
float4x4 proj;
float4x4 normalMatrix; // inverse transpose of model (only the upper 3x3 is used)
RenderMode rmode;

// Standard
Texture2D colorTex;
SamplerState samp;

// Relief. The bake's box-normalized space is exactly the local space, so q = local position.
// For each axis a the bake looked along a; texel (s, t) covers (q[(a + 1) % 3], q[(a + 2) % 3]).
//   depthMaps (layer = axis): RGBA = depths q[a] of hits 0..3, sorted ascending; inf = no hit.
//   colorMaps (layer = axis * HITS + hit): albedo of that hit.
float3 camPos;   // camera position in local [0, 1]^3 space
int linearSteps; // number of uniform steps of the ray search
Texture2DArray<float4> depthMaps;
Texture2DArray<float4> colorMaps;

static const int HITS = 4;
static const float3 LIGHT_DIR = normalize(float3(0.5, 1.0, 0.3)); // direction *towards* the light
static const float AMBIENT = 0.15;

[shader("vertex")]
VSOut vertexMain(VSIn in)
{
    VSOut out;
    out.clip = mul(proj, mul(view, mul(model, float4(in.pos, 1.0))));
    out.nrm = mul((float3x3)normalMatrix, in.nrm); // world-space normal
    out.uv = in.uv;
    out.localPos = in.pos;
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

/// Ray / unit box [0, 1]^3 slab test. Returns (tNear, tFar); tFar < tNear means a miss.
float2 intersectUnitBox(float3 origin, float3 dir)
{
    float3 inv = 1.0 / dir;
    float3 t0 = -origin * inv;
    float3 t1 = (1.0 - origin) * inv;
    float3 tmin = min(t0, t1);
    float3 tmax = max(t0, t1);
    return float2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

/// Uniform-step search: first ray parameter in [tStart, tEnd] whose point is inside the solid.
bool marchLinear(float3 origin, float3 dir, float tStart, float tEnd, out float hitT)
{
    int steps = max(linearSteps, 1);
    float dt = (tEnd - tStart) / float(steps);
    for (int i = 0; i <= steps; i++)
    {
        float t = tStart + dt * float(i);
        if (insideSolid(origin + dir * t))
        {
            hitT = t;
            return true;
        }
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
        // The cube is drawn with its back faces, so this also works with the camera inside it.
        float3 dir = normalize(data.localPos - camPos);
        float2 tt = intersectUnitBox(camPos, dir);
        float tStart = max(tt.x, 0.0);
        float tEnd = tt.y;
        if (tEnd < tStart)
            discard;

        float hitT;
        if (!marchLinear(camPos, dir, tStart, tEnd, hitT))
            discard;

        float3 q = saturate(camPos + dir * hitT);
        int layer = bestLayer(q);
        if (layer < 0)
            discard;

        float3 albedo = colorMaps.Load(int4(texelOf(layer / HITS, q), layer, 0)).rgb;
        float4 clip = mul(proj, mul(view, mul(model, float4(q, 1.0))));
        o.color = float4(albedo, 1.0);
        o.depth = clip.z / clip.w; // clip-space z is already in [0, 1] (camera.perspective)
        return o;
    }

    default:
        o.color = float4(1.0, 1.0, 1.0, 1.0);
        o.depth = data.clip.z;
        return o;
    }
}
