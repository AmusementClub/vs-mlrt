# VapourSynth TensorRT & TensorRT-RTX

The vs-tensorrt plugin provides optimized CUDA runtime for some popular AI filters.

## Usage

Prototype: `core.{trt, trt_rtx}.Model(clip[] clips, string engine_path[, int[] overlap, int[] tilesize, int device_id=0, bint use_cuda_graph=False, int num_streams=1, int verbosity=2, string flexible_output_prop=""])`

Arguments:
- `clip[] clips`: the input clips, only 32-bit floating point RGB or GRAY clips are supported. For model specific input requirements, please consult our [wiki](https://github.com/AmusementClub/vs-mlrt/wiki).
- `string engine_path`: the path to the prebuilt engine (see below)
- `int[] overlap`: some networks (e.g. [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)) support arbitrary input shape where other networks might only support fixed input shape and the input clip must be processed in tiles. The `overlap` argument specifies the overlapping (horizontal and vertical, or both, in pixels) between adjacent tiles to minimize boundary issues. Please refer to network specific docs on the recommended overlapping size.
- `int[] tilesize`: Even for CNN where arbitrary input sizes could be supported, sometimes the network does not work well for the entire range of input dimensions, and you have to limit the size of each tile. This parameter specify the tile size (horizontal and vertical, or both, including the overlapping). Please refer to network specific docs on the recommended tile size.
- `int device_id`: Specifies the GPU device id to use, default 0. Requires Nvidia GPUs with second-generation Kepler architecture onwards.
- `bint use_cuda_graph`: whether to use CUDA Graphs to improve performance and reduce CPU overhead.
- `int num_streams`: number of concurrent CUDA streams to use. Default 1. Increase if GPU not saturated.
- `verbosity`: The verbosity level of TensorRT runtime. The message writes to `stderr`.
  `0`: Internal error. `1`: Application error. `2`: Warning. `3`: Informational messages with instructional information. `4`: Verbose messages with debugging information.
- `string flexible_output_prop`: used to support onnx models with arbitrary number of output planes.

  ```python3
  from typing import TypedDict

  class Output(TypedDict):
      clip: vs.VideoNode
      num_planes: int

  prop = "planes" # arbitrary non-empty string
  output = core.trt.Model(src, engine_path, flexible_output_prop=prop) # type: Output

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

## Instructions for TensorRT

### Build engine with dynamic shape support
- Requires models with built-in dynamic shape support, e.g. `waifu2x_v3.7z` and `dpir_v3.7z`.

1. Build engine
   ```shell
   trtexec --onnx=drunet_gray.onnx --minShapes=input:1x2x8x8 --optShapes=input:1x2x64x64 --maxShapes=input:1x2x1080x1920 --saveEngine=dpir_gray_1080p_dynamic.engine
   ```
   
   The engine will be optimized for `64x64` input and can be applied to eligible inputs with shape from `8x8` to `1920x1080` by specifying parameter `tilesize` in the `trt` plugin.
    
   Also check [trtexec useful arguments](#trtexec-useful-arguments)

### Run model
In vpy script:
```python3
# DPIR
src = core.std.BlankClip(src, width=640, height=360, format=vs.GRAYS)
sigma = 10.0
flt = core.trt.Model([src, core.std.BlankClip(src, color=sigma/255.0)], engine_path="dpir_gray_1080p_dynamic.engine", tilesize=[640, 360])
```

## trtexec useful arguments
- `--workspace=N`: Set workspace size in megabytes (default = 16)

- `--fp16`: Enable fp16 precision, in addition to fp32 (default = disabled)

- `--noTF32`: Disable tf32 precision (default is to enable tf32, in addition to fp32, Ampere only)

- `--device=N`: Select cuda device N (default = 0)

- `--timingCacheFile=<file>`:  Save/load the serialized global timing cache

- `--verbose`: Use verbose logging (default = false)

- `--profilingVerbosity=mode`: Specify profiling verbosity.

  ```
  mode ::= layer_names_only|detailed|none
  ```

  (default = layer_names_only)

- `--tacticSources=tactics`: Specify the tactics to be used by adding (+) or removing (-) tactics from the default

  tactic sources (default = all available tactics).

  Note: Currently only cuDNN, cuBLAS and cuBLAS-LT are listed as optional tactics.

  Tactic Sources: 
  ```
  tactics ::= [","tactic]
  tactic  ::= (+|-)lib
  lib     ::= "CUBLAS"|"CUBLAS_LT"|"CUDNN"
  ```

  For example, to disable cudnn and enable cublas: --tacticSources=-CUDNN,+CUBLAS

- `--useCudaGraph`: Use CUDA graph to capture engine execution and then launch inference (default = disabled).
  This flag may be ignored if the graph capture fails.

- `--noDataTransfers`: Disable DMA transfers to and from device (default = enabled).

- `--saveEngine=<file>`: Save the serialized engine

- `--loadEngine=<file>`: Load a serialized engine

## Instructions for TensorRT-RTX
Replace the `trtexec` executable by the `tensorrt_rtx` executable. Some options may not be supported, e.g. `--fp16`.

