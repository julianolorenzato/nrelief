// Evaluates the exported MLP per pixel. Layout must match export.py.
StructuredBuffer<float> weights;   // weights.bin
RWTexture2D<float4> output;

float3 forward(float2 uv)
{
    // TODO: layer-by-layer matmul + activation reading from `weights`
    return float3(uv, 0.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    output.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;
    float2 uv = (float2(tid.xy) + 0.5) / float2(w, h);
    output[tid.xy] = float4(forward(uv), 1.0);
}
