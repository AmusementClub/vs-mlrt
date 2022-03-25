__version__ = "3.7.0"

__all__ = [
    "Backend",
    "Waifu2x", "Waifu2xModel",
    "DPIR", "DPIRModel",
    "RealESRGANv2", "RealESRGANv2Model",
    "CUGAN"
]

import copy
from dataclasses import dataclass, field
import enum
import math
import os
import subprocess
import sys
import tempfile
import time
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
    @dataclass(frozen=False)
    class ORT_CPU:
        num_streams: int = 1
        verbosity: int = 2
        fp16: bool = False

    @dataclass(frozen=False)
    class ORT_CUDA:
        device_id: int = 0
        cudnn_benchmark: bool = True
        num_streams: int = 1
        verbosity: int = 2
        fp16: bool = False

    @dataclass(frozen=False)
    class OV_CPU:
        fp16: bool = False

    @dataclass(frozen=False)
    class TRT:
        max_shapes: typing.Optional[typing.Tuple[int, int]] = None
        opt_shapes: typing.Optional[typing.Tuple[int, int]] = None
        fp16: bool = False

        device_id: int = 0
        workspace: int = 128
        verbose: bool = False
        use_cuda_graph: bool = False
        num_streams: int = 1
        use_cublas: bool = False # cuBLAS + cuBLASLt
        static_shape: bool = True
        tf32: bool = True
        log: bool = True

        _channels: int = field(init=False, repr=False, compare=False)

backendT = typing.Union[
    Backend.OV_CPU,
    Backend.ORT_CPU,
    Backend.ORT_CUDA,
    Backend.TRT
]


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
    backend: backendT = Backend.OV_CPU(),
    preprocess: bool = True
) -> vs.VideoNode:

    func_name = "vsmlrt.Waifu2x"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample != 32:
        raise ValueError(f"{func_name}: only constant format 32 bit float input supported")

    if not isinstance(noise, int) or noise not in range(-1, 4):
        raise ValueError(f'{func_name}: "noise" must be -1, 0, 1, 2, or 3')

    if not isinstance(scale, int) or scale not in (1, 2):
        raise ValueError(f'{func_name}: "scale" must be 1 or 2')

    if not isinstance(model, int) or model not in Waifu2xModel.__members__.values():
        raise ValueError(f'{func_name}: "model" must be 0, 1, 2, 3, 4, 5, or 6')

    if model == 0 and noise == 0:
        raise ValueError(
            f'{func_name}: "anime_style_art" model'
            ' does not support noise reduction level 0'
        )

    if model == 0:
        if clip.format.id != vs.GRAYS:
            raise ValueError(f'{func_name}: "clip" must be of GRAYS format')
    elif clip.format.id != vs.RGBS:
        raise ValueError(f'{func_name}: "clip" must be of RGBS format')

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

    width, height = clip.width, clip.height
    if preprocess and model in (0, 1, 2):
        # emulating cv2.resize(interpolation=cv2.INTER_CUBIC)
        clip = core.resize.Bicubic(
            clip,
            width * 2, height * 2,
            filter_param_a=0, filter_param_b=0.75
        )

    (tile_w, tile_h), (overlap_w, overlap_h) = calc_tilesize(
        tiles=tiles, tilesize=tilesize,
        width=clip.width, height=clip.height,
        multiple=multiple,
        overlap_w=overlap_w, overlap_h=overlap_h
    )

    if tile_w % multiple != 0 or tile_h % multiple != 0:
        raise ValueError(
            f'{func_name}: tile size must be divisible by {multiple} ({tile_w}, {tile_h})'
        )

    if model == 0:
        channels = 1
    else:
        channels = 3

    backend = init_backend(
        backend=backend,
        channels=channels,
        trt_max_shapes=(tile_w, tile_h)
    )

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
    backend: backendT = Backend.OV_CPU()
) -> vs.VideoNode:

    func_name = "vsmlrt.DPIR"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample != 32:
        raise ValueError(f"{func_name}: only constant format 32 bit float input supported")

    if not isinstance(model, int) or model not in DPIRModel.__members__.values():
        raise ValueError(f'{func_name}: "model" must be 0, 1, 2 or 3')

    if model in [0, 2] and clip.format.id != vs.GRAYS:
        raise ValueError(f'{func_name}: "clip" must be of GRAYS format')
    elif model in [1, 3] and clip.format.id != vs.RGBS:
        raise ValueError(f'{func_name}: "clip" must be of RGBS format')

    if strength is None:
        strength = 5.0

    if isinstance(strength, vs.VideoNode):
        if strength.format.id != vs.GRAYS:
            raise ValueError(f'{func_name}: "strength" must be of GRAYS format')
        if strength.width != clip.width or strength.height != clip.height:
            raise ValueError(f'{func_name}: "strength" must be of the same size as "clip"')
        if strength.num_frames != clip.num_frames:
            raise ValueError(f'{func_name}: "strength" must be of the same length as "clip"')

        strength = core.std.Expr(strength, "x 255 /")
    else:
        try:
            strength = float(strength)
        except TypeError as e:
            raise TypeError(f'{func_name}: "strength" must be a float or a clip') from e

        strength = core.std.BlankClip(clip, format=vs.GRAYS, color=strength / 255)

    if overlap is None:
        overlap_w = overlap_h = 0
    elif isinstance(overlap, int):
        overlap_w = overlap_h = overlap
    else:
        overlap_w, overlap_h = overlap

    multiple = 8

    (tile_w, tile_h), (overlap_w, overlap_h) = calc_tilesize(
        tiles=tiles, tilesize=tilesize,
        width=clip.width, height=clip.height,
        multiple=multiple,
        overlap_w=overlap_w, overlap_h=overlap_h
    )

    if tile_w % multiple != 0 or tile_h % multiple != 0:
        raise ValueError(
            f'{func_name}: tile size must be divisible by {multiple} ({tile_w}, {tile_h})'
        )

    if model in [0, 2]:
        channels = 2
    elif model in [1, 3]:
        channels = 4

    backend = init_backend(
        backend=backend,
        channels=channels,
        trt_max_shapes=(tile_w, tile_h)
    )

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
    backend: backendT = Backend.OV_CPU()
) -> vs.VideoNode:

    func_name = "vsmlrt.RealESRGANv2"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample != 32:
        raise ValueError(f"{func_name}: only constant format 32 bit float input supported")

    if clip.format.id != vs.RGBS:
        raise ValueError(f'{func_name}: "clip" must be of RGBS format')

    if not isinstance(model, int) or model not in RealESRGANv2Model.__members__.values():
        raise ValueError(f'{func_name}: "model" must be 0 or 1')

    if overlap is None:
        overlap_w = overlap_h = 8
    elif isinstance(overlap, int):
        overlap_w = overlap_h = overlap
    else:
        overlap_w, overlap_h = overlap

    multiple = 1

    (tile_w, tile_h), (overlap_w, overlap_h) = calc_tilesize(
        tiles=tiles, tilesize=tilesize,
        width=clip.width, height=clip.height,
        multiple=multiple,
        overlap_w=overlap_w, overlap_h=overlap_h
    )

    channels = 3

    backend = init_backend(
        backend=backend,
        channels=channels,
        trt_max_shapes=(tile_w, tile_h)
    )

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


def CUGAN(
    clip: vs.VideoNode,
    noise: typing.Literal[-1, 0, 1, 2, 3] = -1,
    scale: typing.Literal[2, 3, 4] = 2,
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    overlap: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    backend: backendT = Backend.OV_CPU(),
    preprocess: bool = True,
    alpha: float = 1.0
) -> vs.VideoNode:

    func_name = "vsmlrt.CUGAN"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample != 32:
        raise ValueError(f"{func_name}: only constant format 32 bit float input supported")

    if not isinstance(noise, int) or noise not in range(-1, 4):
        raise ValueError(f'{func_name}: "noise" must be -1, 0, 1, 2, or 3')

    if not isinstance(scale, int) or scale not in (2, 3, 4):
        raise ValueError(f'{func_name}: "scale" must be 2, 3 or 4')

    if scale != 2 and noise in [1, 2]:
        raise ValueError(
            f'{func_name}: "scale={scale}" model'
            f' does not support noise reduction level {noise}'
        )

    if clip.format.id != vs.RGBS:
        raise ValueError(f'{func_name}: "clip" must be of RGBS format')

    if overlap is None:
        overlap_w = overlap_h = 4
    elif isinstance(overlap, int):
        overlap_w = overlap_h = overlap
    else:
        overlap_w, overlap_h = overlap

    multiple = 2

    width, height = clip.width, clip.height

    (tile_w, tile_h), (overlap_w, overlap_h) = calc_tilesize(
        tiles=tiles, tilesize=tilesize,
        width=clip.width, height=clip.height,
        multiple=multiple,
        overlap_w=overlap_w, overlap_h=overlap_h
    )

    if tile_w % multiple != 0 or tile_h % multiple != 0:
        raise ValueError(
            f'{func_name}: tile size must be divisible by {multiple} ({tile_w}, {tile_h})'
        )

    channels = 3

    backend = init_backend(
        backend=backend,
        channels=channels,
        trt_max_shapes=(tile_w, tile_h)
    )

    folder_path = os.path.join(models_path, "cugan")

    if noise == -1:
        model_name = f"up{scale}x-latest-no-denoise.onnx"
    elif noise == 0:
        model_name = f"up{scale}x-latest-conservative.onnx"
    else:
        model_name = f"up{scale}x-latest-denoise{noise}x.onnx"

    network_path = os.path.join(folder_path, model_name)

    # https://github.com/bilibili/ailab/blob/978f3be762183d7fa79525f29a43e65afb995f6b/Real-CUGAN/upcunet_v3.py#L207
    # mutates network_path
    if alpha != 1.0:
        alpha = float(alpha)

        import numpy as np
        import onnx
        from onnx import numpy_helper

        model = onnx.load(network_path)

        for idx, node in reversed(list(enumerate(model.graph.node))):
            if node.op_type == "ConvTranspose":
                break

        upstream_name = node.input[0]
        downstream_name = node.input[0] + "_mul"
        node.input[0] = downstream_name

        alpha_array = np.array(alpha, dtype=np.float32)
        alpha_tensor = numpy_helper.from_array(alpha_array)
        alpha_constant = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["alpha"],
            value=alpha_tensor
        )
        model.graph.node.insert(idx, alpha_constant)

        mul_node = onnx.helper.make_node(
            "Mul",
            inputs=[upstream_name, "alpha"],
            outputs=[downstream_name]
        )
        model.graph.node.insert(idx+1, mul_node)

        network_path = f"{network_path}_alpha{alpha!r}.onnx"
        onnx.save(model, network_path)

    clip = inference(
        clips=[clip], network_path=network_path,
        overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
        backend=backend
    )

    return clip


def get_engine_path(
    network_path: str,
    opt_shapes: typing.Tuple[int, int],
    max_shapes: typing.Tuple[int, int],
    workspace: int,
    fp16: bool,
    device_id: int,
    use_cublas: bool,
    static_shape: bool,
    tf32: bool
) -> str:

    with open(network_path, "rb") as file:
        checksum = zlib.adler32(file.read())

    trt_version = core.trt.Version()["tensorrt_version"].decode()

    try:
        device_name = core.trt.DeviceProperties(device_id)["name"].decode()
        device_name = device_name.replace(' ', '-')
    except AttributeError:
        device_name = f"device{device_id}"

    if static_shape:
        shape_str = f".{max_shapes[0]}x{max_shapes[1]}"
    else:
        shape_str = f".opt{opt_shapes[0]}x{opt_shapes[1]}" + f"_max{max_shapes[0]}x{max_shapes[1]}"

    return (
        network_path +
        shape_str +
        ("_fp16" if fp16 else "") +
        ("_no-tf32" if not tf32 else "") +
        f"_workspace{workspace}" +
        f"_trt-{trt_version}" +
        ("_cublas" if use_cublas else "") +
        f"_{device_name}" +
        f"_{checksum:x}" +
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
    use_cublas: bool = False,
    static_shape: bool = True,
    tf32: bool = True,
    log: bool = False
) -> str:

    if isinstance(opt_shapes, int):
        opt_shapes = (opt_shapes, opt_shapes)

    if isinstance(max_shapes, int):
        max_shapes = (max_shapes, max_shapes)

    engine_path = get_engine_path(
        network_path=network_path,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
        workspace=workspace,
        fp16=fp16,
        device_id=device_id,
        use_cublas=use_cublas,
        static_shape=static_shape,
        tf32=tf32
    )

    if os.access(engine_path, mode=os.R_OK):
        return engine_path

    alter_engine_path = os.path.join(
        tempfile.gettempdir(),
        os.path.splitdrive(engine_path)[1][1:]
    )

    if os.access(alter_engine_path, mode=os.R_OK):
        return alter_engine_path

    try:
        # test writability
        with open(engine_path, "w") as f:
            pass
        os.remove(engine_path)
    except PermissionError:
        print(f"{engine_path} not writable", file=sys.stderr)
        engine_path = alter_engine_path
        dirname = os.path.dirname(engine_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print(f"change engine path to {engine_path}", file=sys.stderr)

    args = [
        trtexec_path,
        f"--onnx={network_path}",
        f"--workspace={workspace}",
        f"--timingCacheFile={engine_path}.cache",
        f"--device={device_id}",
        f"--saveEngine={engine_path}"
    ]

    if static_shape:
        args.append(f"--shapes=input:1x{channels}x{max_shapes[1]}x{max_shapes[0]}")
    else:
        args.extend([
            f"--minShapes=input:1x{channels}x0x0",
            f"--optShapes=input:1x{channels}x{opt_shapes[1]}x{opt_shapes[0]}",
            f"--maxShapes=input:1x{channels}x{max_shapes[1]}x{max_shapes[0]}"
        ])

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

    if not tf32:
        args.append("--noTF32")

    if log:
        env_key = "TRTEXEC_LOG_FILE"
        prev_env_value = os.environ.get(env_key)

        if prev_env_value is not None and len(prev_env_value) > 0:
            # env_key has been set, no extra action
            env = {env_key: prev_env_value}
            subprocess.run(args, env=env, check=True, stdout=sys.stderr)
        else:
            time_str = time.strftime('%y%m%d_%H%M%S', time.localtime())

            temp_filename = os.path.join(
                tempfile.gettempdir(),
                f"trtexec_{time_str}.log"
            )

            env = {env_key: temp_filename}

            completed_process = subprocess.run(args, env=env, check=False, stdout=sys.stderr)

            if completed_process.returncode == 0:
                os.remove(temp_filename)
            else:
                raise RuntimeError(f"trtexec execution fails, log has been written to {temp_filename}")
    else:
        subprocess.run(args, check=True, stdout=sys.stderr)

    return engine_path


def calc_size(width: int, tiles: int, overlap: int, multiple: int = 1) -> int:
    return math.ceil((width + 2 * overlap * (tiles - 1)) / (tiles * multiple)) * multiple


def calc_tilesize(
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]],
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]],
    width: int,
    height: int,
    multiple: int,
    overlap_w: int,
    overlap_h: int
) -> typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]]:

    if tilesize is None:
        if tiles is None:
            overlap_w = 0
            overlap_h = 0
            tile_w = width
            tile_h = height
        elif isinstance(tiles, int):
            tile_w = calc_size(width, tiles, overlap_w, multiple)
            tile_h = calc_size(height, tiles, overlap_h, multiple)
        else:
            tile_w = calc_size(width, tiles[0], overlap_w, multiple)
            tile_h = calc_size(height, tiles[1], overlap_h, multiple)
    elif isinstance(tilesize, int):
        tile_w = tilesize
        tile_h = tilesize
    else:
        tile_w, tile_h = tilesize

    return (tile_w, tile_h), (overlap_w, overlap_h)


def init_backend(
    backend: backendT,
    channels: int,
    trt_max_shapes: typing.Tuple[int, int]
) -> backendT:

    if backend is Backend.ORT_CPU: # type: ignore
        backend = Backend.ORT_CPU()
    elif backend is Backend.ORT_CUDA: # type: ignore
        backend = Backend.ORT_CUDA()
    elif backend is Backend.OV_CPU: # type: ignore
        backend = Backend.OV_CPU()
    elif backend is Backend.TRT: # type: ignore
        backend = Backend.TRT()

    backend = copy.deepcopy(backend)

    if isinstance(backend, Backend.TRT):
        backend._channels = channels

        if backend.max_shapes is None:
            backend.max_shapes = trt_max_shapes

        if backend.opt_shapes is None:
            backend.opt_shapes = backend.max_shapes

    return backend


def inference(
    clips: typing.List[vs.VideoNode],
    network_path: str,
    overlap: typing.Tuple[int, int],
    tilesize: typing.Tuple[int, int],
    backend: backendT
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
            verbosity=backend.verbosity,
            fp16=backend.fp16
        )
    elif isinstance(backend, Backend.ORT_CUDA):
        clip = core.ort.Model(
            clips, network_path,
            overlap=overlap, tilesize=tilesize,
            provider="CUDA", builtin=False,
            device_id=backend.device_id,
            num_streams=backend.num_streams,
            verbosity=backend.verbosity,
            cudnn_benchmark=backend.cudnn_benchmark,
            fp16=backend.fp16
        )
    elif isinstance(backend, Backend.OV_CPU):
        clip = core.ov.Model(
            clips, network_path,
            overlap=overlap, tilesize=tilesize,
            device="CPU", builtin=False,
            fp16=backend.fp16
        )
    elif isinstance(backend, Backend.TRT):
        engine_path = trtexec(
            network_path,
            channels=backend._channels,
            opt_shapes=backend.opt_shapes, # type: ignore
            max_shapes=backend.max_shapes, # type: ignore
            fp16=backend.fp16,
            device_id=backend.device_id,
            workspace=backend.workspace,
            verbose=backend.verbose,
            use_cuda_graph=backend.use_cuda_graph,
            use_cublas=backend.use_cublas,
            static_shape=backend.static_shape,
            tf32=backend.tf32,
            log=backend.log
        )
        clip = core.trt.Model(
            clips, engine_path,
            overlap=overlap,
            tilesize=tilesize,
            device_id=backend.device_id,
            use_cuda_graph=backend.use_cuda_graph,
            num_streams=backend.num_streams,
            verbosity=4 if backend.verbose else 2
        )
    else:
        raise TypeError(f'unknown backend {backend}')

    return clip
