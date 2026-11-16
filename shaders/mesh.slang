// Rasterizes the OBJ mesh; the fragment stage evaluates the MLP.
import mlp; // forward(float2 uv) + weights

float4x4 mvp;

struct VSIn
{
    float3 pos : POSITION;
    float3 nrm : NORMAL;
    float2 uv : TEXCOORD0;
};

struct VSOut
{
    float4 clip : SV_Position;
    float3 nrm : NORMAL;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VSOut vertexMain(VSIn i)
{
    VSOut o;
    o.clip = mul(mvp, float4(i.pos, 1.0));
    o.nrm = i.nrm;
    o.uv = i.uv;
    return o;
}

[shader("fragment")]
float2 fragmentMain(VSOut i) : SV_Target
{
    // TODO: use the MLP output however you like (colour, normal, displacement, ...)
    return i.uv;
}
