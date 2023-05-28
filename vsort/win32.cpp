#ifdef _MSC_VER
#include <windows.h>
#include <delayimp.h>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>

#define DLL_DIR L"vsort"
#define COMMON_CUDA_DIR L"vsmlrt-cuda"

namespace {
std::vector<std::wstring> dlls = {
    // This list must be sorted by dependency.
    L"DirectML.dll",
    L"onnxruntime.dll", // must be the last
};

static std::vector<std::wstring> cudaDlls {
    L"cudart64",
    L"cublasLt64", L"cublas64",
    L"cufft64",
    L"zlibwapi", // cuDNN version 8.3.0+ depends on zlib as a shared library dependency
    L"cudnn_ops_infer64", L"cudnn_cnn_infer64", L"cudnn_adv_infer64", L"cudnn64",
    L"cupti64",
};

bool verbose() { return getenv("VSORT_VERBOSE") != nullptr; }

namespace fs = std::filesystem;
static fs::path dllDir() {
    static const std::wstring res = []() -> std::wstring {
        HMODULE mod = 0;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (char *)dllDir, &mod)) {
            std::vector<wchar_t> buf;
            size_t n = 0;
            do {
                buf.resize(buf.size() + MAX_PATH);
                n = GetModuleFileNameW(mod, buf.data(), buf.size());
            } while (n >= buf.size());
            buf.resize(n);
            std::wstring path(buf.begin(), buf.end());
            return path;
        }
        throw std::runtime_error("unable to locate myself");
    }();
    return fs::path(res).parent_path();
}

FARPROC loadDLLs() {
    fs::path dir = dllDir() / DLL_DIR;
    HMODULE h = nullptr;
    for (const auto dll: dlls) {
        fs::path p = dir / dll;
        std::wstring s = p;
        h = LoadLibraryW(s.c_str());
        if (verbose())
            std::wcerr << DLL_DIR << L": preloading " << p << L": " << h << std::endl;
        if (!h)
            std::wcerr << DLL_DIR << L": failed to preload " << s << std::endl;
    }
    return (FARPROC)h;
}

static void *dummy() { // mimic OrtGetApiBase
    return nullptr;
}

extern "C" FARPROC WINAPI delayload_hook(unsigned reason, DelayLoadInfo* info) {
    switch (reason) {
    case dliNoteStartProcessing:
    case dliNoteEndProcessing:
        // Nothing to do here.
        break;
    case dliNotePreLoadLibrary:
        //std::cerr << "loading " << info->szDll << std::endl;
        if (std::string(info->szDll).find("onnxruntime.dll") != std::string::npos)
            return loadDLLs();
        break;
    case dliNotePreGetProcAddress:
        // Nothing to do here.
        break;
    case dliFailLoadLib:
    case dliFailGetProc:
        // Returning NULL from error notifications will cause the delay load
        // runtime to raise a VcppException structured exception, that some code
        // might want to handle.
        // The SE will crash the process, so instead we return a dummy function.
        return (FARPROC)dummy;
        break;
    default:
        abort(); // unreachable.
        break;
    }
    // Returning NULL causes the delay load machinery to perform default
    // processing for this notification.
    return NULL;
}
} // namespace

extern "C" {
    const PfnDliHook __pfnDliNotifyHook2 = delayload_hook;
    const PfnDliHook __pfnDliFailureHook2 = delayload_hook;
};

bool preloadCudaDlls() {
    std::map<std::wstring, std::filesystem::path> dllmap;

    auto findDllIn = [&](const std::filesystem::path &dir) {
        if (!std::filesystem::is_directory(dir))
            return;
        for (const auto &ent: std::filesystem::directory_iterator{dir}) {
            if (!ent.is_regular_file())
                continue;
            const auto path = ent.path();
            if (path.extension() != ".dll")
                continue;
            const std::wstring filename = path.filename().wstring();
            for (const auto &dll: cudaDlls) {
                if (dllmap.count(dll) > 0)
                    continue;
                if (filename.find(dll) == 0) {
                    if (verbose())
                        std::wcerr << DLL_DIR << L": found " << path << L" for " << dll << std::endl;
                    dllmap.insert({ dll, path });
                    break;
                }
            }
        }
    };
    const fs::path dir = dllDir();
    findDllIn(dir / DLL_DIR);
    findDllIn(dir / COMMON_CUDA_DIR);

    if (verbose()) {
        for (const auto pair: dllmap)
            std::wcerr << DLL_DIR << L": will load " << pair.first << L" from " << pair.second << std::endl;
    }
    for (const auto &dll: cudaDlls) {
        if (dllmap.count(dll) == 0) {
            if (verbose()) {
                std::wcerr << DLL_DIR << L": unable to preload " << dll << L": not found" << std::endl;
                return false;
            }
        }
        std::wstring p = dllmap[dll];
        HMODULE h = LoadLibraryW(p.c_str());
        if (verbose())
            std::wcerr << DLL_DIR << L": preloading " << p << L": " << h << std::endl;
        if (!h) return false;
    }
    return true;
}
#endif
