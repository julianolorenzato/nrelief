enum RenderMode
{
    Relief = 0,
    Standard = 1
}

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
    float3 objPos : TEXCOORD1; // position in object space
};

struct PSOut
{
    float4 color : SV_Target;
    float depth : SV_Depth;
};
