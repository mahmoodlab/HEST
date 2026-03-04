from __future__ import annotations

import os


def get_path_relative(src_file: str, rel_path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(src_file), rel_path))
