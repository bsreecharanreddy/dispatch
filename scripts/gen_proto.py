"""Regenerates the Python gRPC stubs from proto/dispatch.proto. Two known
grpc_tools limitations are worked around here: it emits a bare
`import dispatch_pb2` in the generated _grpc.py file, which breaks once
the file lives inside the dispatch.proto package (fixed by rewriting that
one import to a relative one after generation); and --python_out alone
generates message classes via runtime reflection, invisible to mypy --
--pyi_out generates the accompanying .pyi so GenerateRequest/
GenerateResponse are statically typed (dispatch_pb2_grpc's
service/stub/servicer classes still aren't -- no official pyi generator
covers those, which is why pyproject.toml's mypy override for
dispatch.proto.* uses follow_imports="skip" rather than relying on stubs
alone).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = REPO_ROOT / "proto"
OUT_DIR = REPO_ROOT / "src" / "dispatch" / "proto"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "__init__.py").touch(exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={OUT_DIR}",
            f"--pyi_out={OUT_DIR}",
            f"--grpc_python_out={OUT_DIR}",
            str(PROTO_DIR / "dispatch.proto"),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    grpc_file = OUT_DIR / "dispatch_pb2_grpc.py"
    text = grpc_file.read_text()
    patched = re.sub(
        r"^import dispatch_pb2 as dispatch__pb2$",
        "from . import dispatch_pb2 as dispatch__pb2",
        text,
        flags=re.MULTILINE,
    )
    if patched == text:
        raise RuntimeError(
            "expected bare 'import dispatch_pb2' line not found -- "
            "grpc_tools output format changed since this script was written"
        )
    grpc_file.write_text(patched)


if __name__ == "__main__":
    main()
