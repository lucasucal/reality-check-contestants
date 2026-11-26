"""Native-binary detection by extension and magic bytes (never executes anything)."""
from __future__ import annotations

import os
import re
import struct
from typing import Optional

NATIVE_EXTS = frozenset({
    ".so", ".pyd", ".dylib", ".dll", ".exe", ".o", ".a", ".lib", ".node", ".whl", ".egg",
    ".ptx", ".cubin", ".fatbin",
})
# CUDA source: harmless text, but nothing compiles it and custom CUDA extensions are unsupported.
CUDA_SOURCE_EXTS = frozenset({".cu", ".cuh"})
CUDA_SOURCE_REASON = "CUDA source (not compiled or used; custom CUDA extensions are not supported)"
# TensorRT engines are built for one GPU and one TensorRT version (snapshots only: in a repository
# `.plan` and `.engine` are ordinary names for text files).
ENGINE_EXTS = frozenset({".engine", ".plan", ".trt"})
ENGINE_REASON = "TensorRT engines are built for one GPU and TensorRT version; ship the original weights"
# Zip containers that can hold compiled code (AOTInductor packages) and are therefore scanned inside.
SCANNED_ZIP_EXTS = frozenset({".pt2"})
MAX_ZIP_MEMBERS = 5000
ARCHIVE_EXTS = frozenset({".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".7z", ".rar"})
ARCHIVE_EXEMPT = frozenset({".npz", ".pt", ".pth", ".safetensors"})
CHECKPOINT_EXTS = frozenset({".safetensors", ".pt", ".pth", ".bin", ".ckpt", ".npz", ".npy", ".gguf"})

_VERSIONED_SO = re.compile(r"\.so(\.\d+)+$")
_MACHO = (b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe")


def extension(name: str) -> str:
    return os.path.splitext(name)[1].lower()


def native_extension(name: str) -> Optional[str]:
    """Return the matched extension (e.g. '.so') or None. Versioned '.so.1.2' counts as '.so'."""
    lowered = name.lower()
    if _VERSIONED_SO.search(lowered):
        return ".so"
    ext = extension(lowered)
    return ext if ext in NATIVE_EXTS else None


def cuda_source(name: str) -> bool:
    return extension(name) in CUDA_SOURCE_EXTS


MAX_ZIP_DIRECTORY_BYTES = 4 * 1024 * 1024   # central directory size we are willing to load
MAX_NESTED_ZIP_DEPTH = 1                    # a zip inside the .pt2 is scanned; deeper is rejected
MAX_NESTED_COMPRESSED_BYTES = 64 * 1024 * 1024  # nested non-stored zips larger than this are not decompressed
MAX_PE_HEADER_OFFSET = 1024 * 1024  # bound decompression when checking a ZIP member's PE signature
_EOCD_SIG = b"PK\x05\x06"
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06")


def _directory_size_reason(path) -> Optional[str]:
    """Look only at the end-of-central-directory record: reject before zipfile loads a huge directory."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            tail_len = min(size, 65535 + 22)
            handle.seek(size - tail_len)
            tail = handle.read(tail_len)
    except OSError:
        return None
    at = tail.rfind(_EOCD_SIG)
    if at < 0 or len(tail) - at < 22:
        return None  # not a zip (zipfile will say so)
    total, cd_size = struct.unpack("<H", tail[at + 10:at + 12])[0], struct.unpack("<I", tail[at + 12:at + 16])[0]
    if total == 0xFFFF or cd_size == 0xFFFFFFFF:
        return "has a ZIP64 directory (more than 65,534 entries or a huge directory) and cannot be fully scanned"
    if total > MAX_ZIP_MEMBERS or cd_size > MAX_ZIP_DIRECTORY_BYTES:
        return (f"has {total} members / a {cd_size}-byte directory; archives over {MAX_ZIP_MEMBERS} members "
                f"or {MAX_ZIP_DIRECTORY_BYTES} bytes of directory cannot be fully scanned")
    return None


def _scan_zip(fileobj, depth: int, budget: list) -> Optional[str]:
    """Scan one zip. budget[0] counts members across nested archives. Fails closed: any member that
    cannot be inspected is reported, never treated as clean."""
    import zipfile
    with zipfile.ZipFile(fileobj) as archive:
        infos = archive.infolist()
        budget[0] += len(infos)
        if len(infos) > MAX_ZIP_MEMBERS or budget[0] > MAX_ZIP_MEMBERS:
            return f"has more than {MAX_ZIP_MEMBERS} members and cannot be fully scanned"
        for info in infos:
            if info.is_dir():
                continue
            name = info.filename
            ext = native_extension(name)
            if ext or cuda_source(name):
                return f"contains {name} (extension {ext or extension(name)}; AOTInductor packages carry compiled code)"
            with archive.open(info) as member:
                head = member.read(16)
                if len(head) < min(16, info.file_size):
                    return f"member {name} is shorter than its declared size and cannot be scanned"
                magic = magic_reason(None, info.file_size, head=head)
                if magic:
                    return f"contains {name} ({magic}; AOTInductor packages carry compiled code)"
                if head[:2] == b"MZ" and info.file_size >= 64:
                    member.seek(0x3C)
                    raw = member.read(4)
                    if len(raw) != 4:
                        return f"member {name} has an incomplete PE header and cannot be scanned"
                    offset = struct.unpack("<I", raw)[0]
                    if offset + 4 <= info.file_size:
                        if offset > MAX_PE_HEADER_OFFSET:
                            return f"member {name} has a distant PE header and cannot be fully scanned"
                        member.seek(offset)
                        signature = member.read(4)
                        if len(signature) != 4:
                            return f"member {name} has an incomplete PE signature and cannot be scanned"
                        if signature == b"PE\0\0":
                            return f"contains {name} (PE magic; AOTInductor packages carry compiled code)"
                if head[:4] in _ZIP_MAGIC:
                    if depth >= MAX_NESTED_ZIP_DEPTH:
                        return f"contains nested archive {name} (nesting deeper than {MAX_NESTED_ZIP_DEPTH} level cannot be scanned)"
                    if info.file_size > MAX_NESTED_COMPRESSED_BYTES and info.compress_type != zipfile.ZIP_STORED:
                        return f"contains compressed nested archive {name} that is too large to scan"
                    if info.file_size > MAX_NESTED_COMPRESSED_BYTES:
                        member.seek(0)
                        inner = _scan_zip(member, depth + 1, budget)  # stored: seekable in place
                    else:
                        import io
                        inner = _scan_zip(io.BytesIO(head + member.read()), depth + 1, budget)
                    if inner:
                        return f"nested archive {name} {inner}"
    return None


def zip_native_reason(path) -> Optional[str]:
    """For a zip container (.pt2): name the first member that is native code, else None.

    Looks at member names and magic bytes, including a bounded PE header lookup.
    Never extracts or executes anything. Not a zip: None. Fails closed:
    an archive whose directory exceeds the member/size limits, or one that errors while being
    scanned, is reported rather than treated as clean (the end-of-central-directory record is
    checked first, so a huge directory is never loaded). One nested zip level is scanned.
    """
    import zipfile
    too_big = _directory_size_reason(path)
    if too_big:
        return too_big
    try:
        return _scan_zip(path, 0, [0])
    except zipfile.BadZipFile as exc:
        if "File is not a zip file" in str(exc):
            return None
        return f"is a malformed zip and cannot be fully scanned ({exc})"
    except (OSError, RuntimeError, ValueError, NotImplementedError, EOFError, zipfile.LargeZipFile) as exc:
        return f"could not be fully scanned ({type(exc).__name__}: {exc})"


def magic_reason(path, size: int, head: Optional[bytes] = None) -> Optional[str]:
    """Return 'ELF magic' / 'Mach-O magic' / 'PE magic' / 'ar archive' or None.

    Reads the first 16 bytes; a file starting with 'MZ' additionally has 4 bytes read at e_lfanew.
    """
    try:
        if head is None:
            with open(path, "rb") as handle:
                head = handle.read(16)
        if head[:4] == b"\x7fELF":
            return "ELF magic"
        if head[:4] in _MACHO:
            return "Mach-O magic"
        if head[:4] == b"\xca\xfe\xba\xbe" and size >= 4096 and len(head) >= 8:
            if 1 <= struct.unpack(">I", head[4:8])[0] <= 30:
                return "Mach-O magic"
        if head[:8] == b"!<arch>\n":
            return "ar archive"
        if head[:2] == b"MZ" and size >= 64 and path is not None:
            with open(path, "rb") as handle:
                handle.seek(0x3C)
                raw = handle.read(4)
                if len(raw) == 4:
                    offset = struct.unpack("<I", raw)[0]
                    if offset + 4 <= size:
                        handle.seek(offset)
                        if handle.read(4) == b"PE\0\0":
                            return "PE magic"
    except OSError:
        return None
    return None
