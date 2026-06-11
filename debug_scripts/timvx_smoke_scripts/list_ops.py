#!/usr/bin/env python3
"""List the distinct dialect.<op> identifiers used as ops in a .mlir file.
"""
import re
import sys


OP_LINE = re.compile(
    r"""^\s*
        %\d+(?::\d+)?
        (?:\s*,\s*%\d+(?::\d+)?)*
        \s*=\s*
        "?
        ([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <file.mlir>", file=sys.stderr)
        return 2
    ops: set[str] = set()
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        for line in f:
            m = OP_LINE.match(line)
            if m:
                ops.add(m.group(1))
    for op in sorted(ops):
        print(op)
    return 0


if __name__ == "__main__":
    sys.exit(main())
