import definitions;
import relief_mapping;

float4x4 mvp;
RenderMode rmode;

[shader("vertex")]
VSOut vertexMain(VSIn i)
{
    VSOut o;
    o.clip = mul(mvp, float4(i.pos, 1.0));
    // o.nrm = i.nrm;
    // o.uv = i.uv;
    return o;
}

[shader("fragment")]
float4 fragmentMain(VSOut data) : SV_Target
{
    switch (rmode)
    {
    case RenderMode::Standard: {
        return float4(1.0, 0.5, 1.0, 1.0);
    }

    case RenderMode::Relief: {
        return float4(0.0, 0.5, 1.0, 1.0);
    }
    default:
        return float4(1.0, 1.0, 1.0, 1.0);
    }
}
