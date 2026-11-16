import definitions;
import relief_mapping;

float4x4 mvp;
RenderMode rmode;
Texture2D colorTex;
SamplerState samp;

[shader("vertex")]
VSOut vertexMain(VSIn i)
{
    VSOut o;
    o.clip = mul(mvp, float4(i.pos, 1.0));
    o.nrm = i.nrm; // model matrix is identity, so this is already the world-space normal
    o.uv = i.uv;
    return o;
}

static const float3 LIGHT_DIR = normalize(float3(0.5, 1.0, 0.3)); // direction *towards* the light
static const float AMBIENT = 0.15;

[shader("fragment")]
float4 fragmentMain(VSOut data) : SV_Target
{
    switch (rmode)
    {
    case RenderMode::Standard: {
        // Lambert diffuse + constant ambient.
        float3 n = normalize(data.nrm); // interpolation denormalizes it
        float diffuse = max(dot(n, LIGHT_DIR), 0.0);
        // OBJ uvs have V pointing up, but image row 0 is the top: flip V.
        float2 uv = float2(data.uv.x, 1.0 - data.uv.y);
        float3 albedo = colorTex.Sample(samp, uv).rgb;
        return float4(albedo * (AMBIENT + diffuse), 1.0);
    }

    case RenderMode::Relief: {
        // Temporary debug view: UVs as color.
        return float4(data.uv, 0.0, 1.0);
    }
    default:
        return float4(1.0, 1.0, 1.0, 1.0);
    }
}
