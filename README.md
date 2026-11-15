# nrelief

    uv run nrelief train     # -> checkpoint.pt
    uv run nrelief export    # -> weights.bin, meta.json
    slangc shaders/mlp.slang -target spirv -entry computeMain -stage compute -o mlp.spv
    uv run nrelief view      # real-time SlangPy window (needs weights.bin)
