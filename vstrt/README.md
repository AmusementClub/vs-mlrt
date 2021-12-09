# VapourSynth TensorRT

The vs-tensorrt plugin provides optimized CUDA runtime for some popular AI filters.

## Usage
```python
core.trt.Model(clip[] clips, string engine_path[, int pad, int block_w, int block_h, int device_id=0, bint use_cuda_graph=False, int num_streams=1, int verbosity=2])
```

- `clip[] clips`: the input clips, only 32-bit floating point RGB or GRAY clips are supported. For model specific input requirements, please consult our [wiki](https://github.com/AmusementClub/vs-mlrt/wiki).
- `string engine_path`: the path to the prebuilt engine (see below)
- `int pad`: some networks (e.g. [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)) support arbitrary input shape where other networks might only support fixed input shape and the input clip must be processed in tiles. The `pad` argument specifies the overlapping (both horizontal and vertical, in pixels) between adjacent tiles to minimize boundary issues. Please refer to network specific docs on the recommended padding size.
- `int block_w`: Even for CNN where arbitrary input sizes could be supported, sometimes the network does not work well for the entire range of input dimensions, and you have to limit the size of each tile. This parameter specify the horizontal tile size (including the padding). Please refer to network specific docs on the recommended tile size.
- `int block_h`: Similar to `block_w`, this set the height of the tile. If unspecified, it will default to `block_w`.
- `int device_id`: Specifies the GPU device id to use, default 0.
- `int num_streams`: number of concurrent CUDA streams to use. Default 1. Increase if GPU not saturated.
- `verbosity`: The verbosity level of TensorRT runtime. The message writes to `stderr`.
  `0`: Internal error. `1`: Application error. `2`: Warning. `3`: Informational messages with instructional information. `4`: Verbose messages with debugging information.

## Instructions

### Build engine with dynamic shape support
- Requires models with built-in dynamic shape support, e.g. `waifu2x_v3.7z` and `dpir_v3.7z`.

1. Build engine
   ```shell
   trtexec --onnx=drunet_gray.onnx --minShapes=input:1x1x0x0 --optShapes=input:1x1x64x64 --maxShapes=input:1x1x1080x1920 --saveEngine=dpir_gray_1080p_dynamic.engine --tacticSources=+CUDNN,-CUBLAS,-CUBLAS_LT
   ```
   
   The engine will be optimized for `64x64` input and can be applied to eligible inputs with shape from `0x0` to `1920x1080` by specifying parameters `block_w` and `block_h` in the `trt` plugin.
    
   Also check [trtexec useful arguments](#trtexec-useful-arguments)

### Run model
In vpy script:
```python3
# DPIR
src = core.std.BlankClip(src, width=640, height=360, format=vs.GRAYS)
sigma = 10.0
flt = core.trt.Model([src, core.std.BlankClip(src, color=sigma/255.0)], engine_path="dpir_gray_640_360.engine", block_w=640, block_h=360)
```

## trtexec useful arguments
- `--workspace=N`: Set workspace size in megabytes (default = 16)

- `--fp16`: Enable fp16 precision, in addition to fp32 (default = disabled)

- `--noTF32`: Disable tf32 precision (default is to enable tf32, in addition to fp32, Ampere only)

- `--device=N`: Select cuda device N (default = 0)

- `--timingCacheFile=<file>`:  Save/load the serialized global timing cache

- `--buildOnly` :Skip inference perf measurement (default = disabled)

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

