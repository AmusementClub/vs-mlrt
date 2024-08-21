# VapourSynth MIGraphX

The vs-migraphx plugin provides optimized HIP runtime for some popular AI filters on AMD GPUs.

## Usage

Prototype: `core.migx.Model(clip[] clips, string program_path[, int[] overlap, int[] tilesize, int device_id=0, int num_streams=1, string flexible_output_prop=""])`

Arguments:
- `clip[] clips`: the input clips, only 16/32-bit floating point RGB or GRAY clips are supported. For model specific input requirements, please consult our [wiki](https://github.com/AmusementClub/vs-mlrt/wiki).
- `string program_path`: the path to the prebuilt program (see below)
- `int[] overlap`: some networks (e.g. [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)) support arbitrary input shape where other networks might only support fixed input shape and the input clip must be processed in tiles. The `overlap` argument specifies the overlapping (horizontal and vertical, or both, in pixels) between adjacent tiles to minimize boundary issues. Please refer to network specific docs on the recommended overlapping size.
- `int[] tilesize`: Even for CNN where arbitrary input sizes could be supported, sometimes the network does not work well for the entire range of input dimensions, and you have to limit the size of each tile. This parameter specify the tile size (horizontal and vertical, or both, including the overlapping). Please refer to network specific docs on the recommended tile size.
- `int device_id`: Specifies the GPU device id to use, default 0. Requires AMD GPUs with gfx1030 target or RDNA3 architecture onwards ([list](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html#windows-supported-gpus)).
- `int num_streams`: number of concurrent HIP streams to use. Default 1. Increase if GPU not saturated.
- `string flexible_output_prop`: used to support onnx models with arbitrary number of output planes.

  ```python3
  from typing import TypedDict

  class Output(TypedDict):
      clip: vs.VideoNode
      num_planes: int

  prop = "planes" # arbitrary non-empty string
  output = core.migx.Model(src, program_path, flexible_output_prop=prop) # type: Output

  clip = output["clip"]
  num_planes = output["num_planes"]

  output_planes = [
      clip.std.PropToClip(prop=f"{prop}{i}")
      for i in range(num_planes)
  ] # type: list[vs.VideoNode]
  ```
  
When `overlap` and `tilesize` are not specified, the filter will internally try to resize the network to fit the input clips. This might not always work (for example, the network might require the width to be divisible by 8), and the filter will error out in this case.

The general rule is to either:
1. left out `overlap`, `tilesize` at all and just process the input frame in one tile, or
2. set all three so that the frame is processed in `tilesize[0]` x `tilesize[1]` tiles, and adjacent tiles will have an overlap of `overlap[0]` x `overlap[1]` pixels on each direction. The overlapped region will be throw out so that only internal output pixels are used.

## Instructions

### Build program
   ```shell
   migraphx-driver compile --onnx drunet_gray.onnx --gpu --input-dim @input 1 2 1080 1920 --output dpir_gray_1080p.mxr
   ```
   
   The program can be applied to `1920x1080` input.
    
   Also check [migraphx-driver useful arguments](#migraphx-driver-useful-arguments)

### Run model
In vpy script:
```python3
# DPIR
src = core.std.BlankClip(src, width=1920, height=1080, format=vs.GRAYS)
sigma = 10.0
flt = core.migx.Model([src, core.std.BlankClip(src, color=sigma/255.0)], engine_path="dpir_gray_1080p.mxr", tilesize=[1920, 1080])
```

## trtexec useful arguments
- `--fp16`: Enable fp16 precision, in addition to fp32 (default = disabled)

- `--output <file>`: Save the serialized program

- `--migraphx <file>`: Load a serialized program

- `--optimize`: Performs common graph optimizations

- `--exhaustive-tune`: Enables exhaustive search to find the fastest kernel

- `--disable-fast-math`: Disable fast math optimization

Also check the [full list of options](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/migraphx-driver.html#options) and [environment variables](https://rocm.docs.amd.com/projects/AMDMIGraphX/en/latest/dev/env_vars.html).

