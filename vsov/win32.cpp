#ifdef _MSC_VER
#include <windows.h>
#include <delayimp.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem>

#define DLL_DIR L"vsov"

#include <iostream>

namespace {
std::vector<std::wstring> dlls = {
    // This list must be sorted by dependency.
#ifdef WIN32_SHARED_OPENVINO
    L"tbb12.dll",
    L"openvino.dll", // must be the last
#else // WIN32_SHARED_OPENVINO
    L"tbb12.dll", // must be the last
#endif // WIN32_SHARED_OPENVINO
};

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
        if (getenv("VSOV_VERBOSE"))
            std::wcerr << L"vsov: preloading " << p << L": " << h << std::endl;
        if (!h)
            std::wcerr << L"vsov: failed to preload " << s << std::endl;
    }
    return (FARPROC)h;
}

extern "C" FARPROC WINAPI delayload_hook(unsigned reason, DelayLoadInfo* info) {
    switch (reason) {
    case dliNoteStartProcessing:
    case dliNoteEndProcessing:
        // Nothing to do here.
        break;
    case dliNotePreLoadLibrary:
        //std::cerr << "loading " << info->szDll << std::endl;
#ifdef WIN32_SHARED_OPENVINO
        if (std::string(info->szDll).find("openvino.dll") != std::string::npos)
            return loadDLLs();
#else // WIN32_SHARED_OPENVINO
        if (std::string(info->szDll).find("tbb.dll") != std::string::npos)
            return loadDLLs();
#endif // WIN32_SHARED_OPENVINO
        break;
    case dliNotePreGetProcAddress:
        // Nothing to do here.
        break;
    case dliFailLoadLib:
    case dliFailGetProc:
        // Returning NULL from error notifications will cause the delay load
        // runtime to raise a VcppException structured exception, that some code
        // might want to handle.
        return NULL;
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
#endif
