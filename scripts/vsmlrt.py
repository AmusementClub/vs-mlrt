__version__ = "3.1.0"

__all__ = [
    "Backend",
    "Waifu2x", "Waifu2xModel",
    "DPIR", "DPIRModel",
    "RealESRGANv2", "RealESRGANv2Model"
]

from dataclasses import dataclass, field
import enum
import math
import os
import subprocess
import sys
import typing
import zlib

import vapoursynth as vs
from vapoursynth import core


def get_plugins_path() -> str:
    path = b""

    try:
        path = core.trt.Version()["path"]
    except AttributeError:
        try:
            path = core.ort.Version()["path"]
        except AttributeError:
            path = core.ov.Version()["path"]

    assert path != b""

    return os.path.dirname(path).decode()

plugins_path: str = get_plugins_path()
trtexec_path: str = os.path.join(plugins_path, "vsmlrt-cuda", "trtexec")
models_path: str = os.path.join(plugins_path, "models")


class Backend:
    @dataclass(frozen=True)
    class ORT_CPU:
        num_streams: int = 1
        verbosity: int = 2

    @dataclass(frozen=True)
    class ORT_CUDA:
        device_id: int = 0
        cudnn_benchmark: bool = True
        num_streams: int = 1
        verbosity: int = 2

    @dataclass(frozen=True)
    class OV_CPU:
        pass

    @dataclass
    class TRT:
        max_shapes: typing.Tuple[int, int]

        device_id: int = 0
        opt_shapes: typing.Tuple[int, int] = (64, 64)
        fp16: bool = False
        workspace: int = 128
        verbose: bool = False
        use_cuda_graph: bool = False
        num_streams: int = 1
        use_cublas: bool = False # cuBLAS + cuBLASLt

        _channels: int = field(init=False, repr=False, compare=False)


def calc_size(width: int, tiles: int, overlap: int, multiple: int = 1) -> int:
    return math.ceil((width + 2 * overlap * (tiles - 1)) / (tiles * multiple)) * multiple


def inference(
    clips: typing.List[vs.VideoNode],
    network_path: str,
    overlap: typing.Tuple[int, int],
    tilesize: typing.Tuple[int, int],
    backend: typing.Union[Backend.OV_CPU, Backend.ORT_CPU, Backend.ORT_CUDA]
) -> vs.VideoNode:

    if not os.path.exists(network_path):
        raise RuntimeError(
            f'"{network_path}" not found, '
            f'built-in models can be found at https://github.com/AmusementClub/vs-mlrt/releases'
        )

    if isinstance(backend, Backend.ORT_CPU):
        clip = core.ort.Model(
            clips, network_path,
            overlap=overlap, tilesize=tilesize,
            provider="CPU", builtin=False,
            num_streams=backend.num_streams,
            verbosity=backend.verbosity
        )
    elif isinstance(backend, Backend.ORT_CUDA):
        clip = core.ort.Model(
            clips, network_path,
            overlap=overlap, tilesize=tilesize,
            provider="CUDA", builtin=False,
            device_id=backend.device_id,
            num_streams=backend.num_streams,
            verbosity=backend.verbosity,
            cudnn_benchmark=backend.cudnn_benchmark
        )
    elif isinstance(backend, Backend.OV_CPU):
        clip = core.ov.Model(
            clips, network_path,
            overlap=overlap, tilesize=tilesize,
            device="CPU", builtin=False
        )
    elif isinstance(backend, Backend.TRT):
        engine_path = trtexec(
            network_path,
            channels=backend._channels,
            opt_shapes=backend.opt_shapes,
            max_shapes=backend.max_shapes,
            fp16=backend.fp16,
            device_id=backend.device_id,
            workspace=backend.workspace,
            verbose=backend.verbose,
            use_cuda_graph=backend.use_cuda_graph,
            use_cublas=backend.use_cublas
        )
        clip = core.trt.Model(
            clips, engine_path,
            overlap=overlap, tilesize=tilesize,
            device_id=backend.device_id,
            use_cuda_graph=backend.use_cuda_graph,
            num_streams=backend.num_streams,
            verbosity=4 if backend.verbose else 2
        )
    else:
        raise ValueError(f'unknown backend {backend}')

    return clip


@enum.unique
class Waifu2xModel(enum.IntEnum):
    anime_style_art = 0
    anime_style_art_rgb = 1
    photo = 2
    upconv_7_anime_style_art_rgb = 3
    upconv_7_photo = 4
    upresnet10 = 5
    cunet = 6


def Waifu2x(
    clip: vs.VideoNode,
    noise: typing.Literal[-1, 0, 1, 2, 3] = -1,
    scale: typing.Literal[1, 2] = 2,
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    overlap: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    model: typing.Literal[0, 1, 2, 3, 4, 5, 6] = 6,
    backend: typing.Union[Backend.OV_CPU, Backend.ORT_CPU, Backend.ORT_CUDA] = Backend.OV_CPU()
) -> vs.VideoNode:

    funcName = "vsmlrt.Waifu2x"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample != 32:
        raise ValueError(f"{funcName}: only constant format 32 bit float input supported")

    if not isinstance(noise, int) or noise not in range(-1, 4):
        raise ValueError(f'{funcName}: "noise" must be -1, 0, 1, 2, or 3')

    if not isinstance(scale, int) or scale not in (1, 2):
        raise ValueError(f'{funcName}: "scale" must be 1 or 2')

    if not isinstance(model, int) or model not in Waifu2xModel.__members__.values():
        raise ValueError(f'{funcName}: "model" must be 0, 1, 2, 3, 4, 5, or 6')

    if model == 0 and noise == 0:
        raise ValueError(
            f'{funcName}: "anime_style_art" model'
            ' does not support noise reduction level 0'
        )

    if model == 0:
        if clip.format.id != vs.GRAYS:
            raise ValueError(f'{funcName}: "clip" must be of GRAYS format')
    elif clip.format.id != vs.RGBS:
        raise ValueError(f'{funcName}: "clip" must be of RGBS format')

    if overlap is None:
        overlap_w = overlap_h = [8, 8, 8, 8, 8, 4, 4][model]
    elif isinstance(overlap, int):
        overlap_w = overlap_h = overlap
    else:
        overlap_w, overlap_h = overlap

    if model == 6:
        multiple = 4
    else:
        multiple = 1

    if tilesize is None:
        if tiles is None:
            overlap = 0
            tile_w = clip.width
            tile_h = clip.height
        elif isinstance(tiles, int):
            tile_w = calc_size(clip.width, tiles, overlap_w, multiple)
            tile_h = calc_size(clip.height, tiles, overlap_h, multiple)
        else:
            tile_w = calc_size(clip.width, tiles[0], overlap_w, multiple)
            tile_h = calc_size(clip.height, tiles[1], overlap_h, multiple)
    elif isinstance(tilesize, int):
        tile_w = tilesize
        tile_h = tilesize
    else:
        tile_w, tile_h = tilesize

    if model == 6 and (tile_w % 4 != 0 or tile_h % 4 != 0):
        raise ValueError(f'{funcName}: tile size of cunet model must be divisible by 4 ({tile_w}, {tile_h})')

    if backend is Backend.ORT_CPU: # type: ignore
        backend = Backend.ORT_CPU()
    elif backend is Backend.ORT_CUDA: # type: ignore
        backend = Backend.ORT_CUDA()
    elif backend is Backend.OV_CPU: # type: ignore
        backend = Backend.OV_CPU()
    elif backend is Backend.TRT: # type: ignore
        raise TypeError(f'{funcName}: trt backend must be instantiated')

    if isinstance(backend, Backend.TRT):
        if model == 0:
            backend._channels = 1
        else:
            backend._channels = 3

    folder_path = os.path.join(
        models_path,
        "waifu2x",
        tuple(Waifu2xModel.__members__)[model]
    )

    if model in (0, 1, 2):
        if noise == -1:
            model_name = "scale2.0x_model.onnx"
        else:
            model_name = f"noise{noise}_model.onnx"
    elif model in (3, 4, 5):
        if noise == -1:
            model_name = "scale2.0x_model.onnx"
        else:
            model_name = f"noise{noise}_scale2.0x_model.onnx"
    else:
        if scale == 1:
            scale_name = ""
        else:
            scale_name = "scale2.0x_"

        if noise == -1:
            model_name = "scale2.0x_model.onnx"
        else:
            model_name = f"noise{noise}_{scale_name}model.onnx"

    network_path = os.path.join(folder_path, model_name)

    width, height = clip.width, clip.height
    if model in (0, 1, 2):
        # emulating cv2.resize(interpolation=cv2.INTER_CUBIC)
        clip = core.resize.Bicubic(
            clip,
            width * 2, height * 2,
            filter_param_a=0, filter_param_b=0.75
        )

    clip = inference(
        clips=[clip], network_path=network_path,
        overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
        backend=backend
    )

    if scale == 1 and clip.width // width == 2:
        # emulating cv2.resize(interpolation=cv2.INTER_CUBIC)
        # cr: @AkarinVS
        clip = core.fmtc.resample(
            clip, scale=0.5,
            kernel="impulse", impulse=[-0.1875, 1.375, -0.1875],
            kovrspl=2
        )

    return clip


@enum.unique
class DPIRModel(enum.IntEnum):
    drunet_gray = 0
    drunet_color = 1
    drunet_deblocking_grayscale = 2
    drunet_deblocking_color = 3


def DPIR(
    clip: vs.VideoNode,
    strength: typing.Optional[typing.Union[typing.SupportsFloat, vs.VideoNode]],
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    overlap: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    model: typing.Literal[0, 1, 2, 3] = 0,
    backend: typing.Union[Backend.OV_CPU, Backend.ORT_CPU, Backend.ORT_CUDA] = Backend.OV_CPU()
) -> vs.VideoNode:

    funcName = "vsmlrt.DPIR"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample != 32:
        raise ValueError(f"{funcName}: only constant format 32 bit float input supported")

    if not isinstance(model, int) or model not in DPIRModel.__members__.values():
        raise ValueError(f'{funcName}: "model" must be 0, 1, 2 or 3')

    if model in [0, 2] and clip.format.id != vs.GRAYS:
        raise ValueError(f'{funcName}: "clip" must be of GRAYS format')
    elif model in [1, 3] and clip.format.id != vs.RGBS:
        raise ValueError(f'{funcName}: "clip" must be of RGBS format')

    if strength is None:
        strength = 5.0

    if isinstance(strength, vs.VideoNode):
        if strength.format.id != vs.GRAYS:
            raise ValueError(f'{funcName}: "strength" must be of GRAYS format')
        if strength.width != clip.width or strength.height != clip.height:
            raise ValueError(f'{funcName}: "strength" must be of the same size as "clip"')
        if strength.num_frames != clip.num_frames:
            raise ValueError(f'{funcName}: "strength" must be of the same length as "clip"')

        strength = core.std.Expr(strength, "x 255 /")
    else:
        try:
            strength = float(strength)
        except TypeError as e:
            raise TypeError(f'{funcName}: "strength" must be a float or a clip') from e

        strength = core.std.BlankClip(clip, format=vs.GRAYS, color=strength / 255)

    if overlap is None:
        overlap_w = overlap_h = 0
    elif isinstance(overlap, int):
        overlap_w = overlap_h = overlap
    else:
        overlap_w, overlap_h = overlap

    multiple = 8

    if tilesize is None:
        if tiles is None:
            overlap = 0
            tile_w = clip.width
            tile_h = clip.height
        elif isinstance(tiles, int):
            tile_w = calc_size(clip.width, tiles, overlap_w, multiple)
            tile_h = calc_size(clip.height, tiles, overlap_h, multiple)
        else:
            tile_w = calc_size(clip.width, tiles[0], overlap_w, multiple)
            tile_h = calc_size(clip.height, tiles[1], overlap_h, multiple)
    elif isinstance(tilesize, int):
        tile_w = tilesize
        tile_h = tilesize
    else:
        tile_w, tile_h = tilesize

    if tile_w % 8 != 0 or tile_h % 8 != 0:
        raise ValueError(f'{funcName}: tile size must be divisible by 8 ({tile_w}, {tile_h})')

    if backend is Backend.ORT_CPU: # type: ignore
        backend = Backend.ORT_CPU()
    elif backend is Backend.ORT_CUDA: # type: ignore
        backend = Backend.ORT_CUDA()
    elif backend is Backend.OV_CPU: # type: ignore
        backend = Backend.OV_CPU()
    elif backend is Backend.TRT: # type: ignore
        raise TypeError(f'{funcName}: trt backend must be instantiated')

    if isinstance(backend, Backend.TRT):
        if model in [0, 2]:
            backend._channels = 2
        elif model in [1, 3]:
            backend._channels = 4

    network_path = os.path.join(
        models_path,
        "dpir",
        f"{tuple(DPIRModel.__members__)[model]}.onnx"
    )

    clip = inference(
        clips=[clip, strength], network_path=network_path,
        overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
        backend=backend
    )

    return clip


@enum.unique
class RealESRGANv2Model(enum.IntEnum):
    animevideo_xsx2 = 0
    animevideo_xsx4 = 1


def RealESRGANv2(
    clip: vs.VideoNode,
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    overlap: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    model: typing.Literal[0, 1] = 0,
    backend: typing.Union[Backend.OV_CPU, Backend.ORT_CPU, Backend.ORT_CUDA] = Backend.OV_CPU()
) -> vs.VideoNode:

    funcName = "vsmlrt.RealESRGANv2"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{funcName}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample != 32:
        raise ValueError(f"{funcName}: only constant format 32 bit float input supported")

    if clip.format.id != vs.RGBS:
        raise ValueError(f'{funcName}: "clip" must be of RGBS format')

    if not isinstance(model, int) or model not in RealESRGANv2Model.__members__.values():
        raise ValueError(f'{funcName}: "model" must be 0 or 1')

    if overlap is None:
        overlap_w = overlap_h = 8
    elif isinstance(overlap, int):
        overlap_w = overlap_h = overlap
    else:
        overlap_w, overlap_h = overlap

    if tilesize is None:
        if tiles is None:
            overlap = 0
            tile_w = clip.width
            tile_h = clip.height
        elif isinstance(tiles, int):
            tile_w = calc_size(clip.width, tiles, overlap_w)
            tile_h = calc_size(clip.height, tiles, overlap_h)
        else:
            tile_w = calc_size(clip.width, tiles[0], overlap_w)
            tile_h = calc_size(clip.height, tiles[1], overlap_h)
    elif isinstance(tilesize, int):
        tile_w = tilesize
        tile_h = tilesize
    else:
        tile_w, tile_h = tilesize

    if backend is Backend.ORT_CPU: # type: ignore
        backend = Backend.ORT_CPU()
    elif backend is Backend.ORT_CUDA: # type: ignore
        backend = Backend.ORT_CUDA()
    elif backend is Backend.OV_CPU: # type: ignore
        backend = Backend.OV_CPU()
    elif backend is Backend.TRT: # type: ignore
        raise TypeError(f'{funcName}: trt backend must be instantiated')

    if isinstance(backend, Backend.TRT):
        backend._channels = 3

    network_path = os.path.join(
        models_path,
        "RealESRGANv2",
        f"RealESRGANv2-{tuple(RealESRGANv2Model.__members__)[model]}.onnx".replace('_', '-')
    )

    clip = inference(
        clips=[clip], network_path=network_path,
        overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
        backend=backend
    )

    return clip


def get_engine_name(
    network_path: str,
    opt_shapes: typing.Tuple[int, int],
    max_shapes: typing.Tuple[int, int],
    workspace: int,
    fp16: bool,
    device_id: int,
    use_cublas: bool
) -> str:

    with open(network_path, "rb") as f:
        checksum = zlib.adler32(f.read())

    trt_version = core.trt.Version()["tensorrt_version"].decode()

    return (
        network_path +
        f".{checksum}" +
        f"_trt-{trt_version}"
        f"_device{device_id}" +
        f"_opt{opt_shapes[1]}x{opt_shapes[0]}" +
        f"_max{max_shapes[1]}x{max_shapes[0]}" +
        f"_workspace{workspace}" +
        ("_fp16" if fp16 else "") +
        ("_cublas" if use_cublas else "") +
        ".engine"
    )


def trtexec(
    network_path: str,
    channels: int,
    opt_shapes: typing.Tuple[int, int],
    max_shapes: typing.Tuple[int, int],
    fp16: bool,
    device_id: int,
    workspace: int = 128,
    verbose: bool = False,
    use_cuda_graph: bool = False,
    use_cublas: bool = False
) -> str:

    if isinstance(opt_shapes, int):
        opt_shapes = (opt_shapes, opt_shapes)

    if isinstance(max_shapes, int):
        max_shapes = (max_shapes, max_shapes)

    engine_path = get_engine_name(
        network_path=network_path,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
        workspace=workspace,
        fp16=fp16,
        device_id=device_id,
        use_cublas=use_cublas
    )

    if os.path.exists(engine_path):
        return engine_path

    args = [
        trtexec_path,
        f"--onnx={network_path}",
        f"--minShapes=input:1x{channels}x0x0",
        f"--optShapes=input:1x{channels}x{opt_shapes[1]}x{opt_shapes[0]}",
        f"--maxShapes=input:1x{channels}x{max_shapes[1]}x{max_shapes[0]}",
        f"--workspace={workspace}",
        f"--timingCacheFile={engine_path + '.cache'}",
        f"--device={device_id}",
        f"--saveEngine={engine_path}"
    ]

    if fp16:
        args.append("--fp16")

    if verbose:
        args.append("--verbose")

    if not use_cublas:
        args.append("--tacticSources=-CUBLAS,-CUBLAS_LT")

    if use_cuda_graph:
        args.extend((
            "--useCudaGraph",
            "--noDataTransfers"
        ))
    else:
        args.append("--buildOnly")

    subprocess.run(args, check=True, stdout=sys.stderr)

    return engine_path
