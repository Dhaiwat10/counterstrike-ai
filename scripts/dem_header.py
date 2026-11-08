#!/usr/bin/env python3
"""Print the header of GoldSrc (Half-Life / CS 1.6) .dem files.

Layout: char[8] magic "HLDEMO\\0", int32 demo protocol, int32 network protocol,
char[260] map name, char[260] game dir. Little-endian.

Usage: python3 -I scripts/dem_header.py FILE.dem [FILE.dem ...]
Reads files as data only; nothing from them is executed.
"""
import os
import struct
import sys

HEADER_SIZE = 8 + 4 + 4 + 260 + 260


def cstr(b: bytes) -> str:
    return b.split(b"\0", 1)[0].decode("latin-1")


def read_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(HEADER_SIZE)
    if len(raw) < HEADER_SIZE:
        raise ValueError(f"file too short ({len(raw)} bytes)")
    magic = raw[:8]
    demo_proto, net_proto = struct.unpack_from("<ii", raw, 8)
    # Optional trailer fields after the spec'd header: uint32 map CRC, int32 directory offset.
    map_crc = dir_offset = dir_entries = None
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE)
        extra = f.read(8)
        if len(extra) == 8:
            map_crc, dir_offset = struct.unpack("<Ii", extra)
            if 0 < dir_offset < os.path.getsize(path) - 4:
                f.seek(dir_offset)
                dir_entries = struct.unpack("<i", f.read(4))[0]
    return {
        "map_crc": map_crc,
        "dir_offset": dir_offset,
        "dir_entries": dir_entries,
        "magic": magic,
        "magic_ok": magic == b"HLDEMO\0\0" or magic[:7] == b"HLDEMO\0",
        "demo_protocol": demo_proto,
        "network_protocol": net_proto,
        "map": cstr(raw[16:276]),
        "gamedir": cstr(raw[276:536]),
    }


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    rc = 0
    for p in argv:
        try:
            h = read_header(p)
        except Exception as e:  # noqa: BLE001
            print(f"{p}\tERROR: {e}")
            rc = 1
            continue
        print(
            f"{p}\tsize={os.path.getsize(p)}\tmagic={h['magic']!r} ok={h['magic_ok']}"
            f"\tdemo_proto={h['demo_protocol']}\tnet_proto={h['network_protocol']}"
            f"\tmap={h['map']}\tgamedir={h['gamedir']}"
            f"\tmap_crc={h['map_crc']:#010x}\tdir_entries={h['dir_entries']}"
        )
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
