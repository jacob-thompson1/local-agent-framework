"""Built-in example tools, each annotated with measured token costs.

Approximate prompt costs (measured with the default counter; your tokenizer
may vary by a few tokens):

=====================  ========  ============  =====================================
Tool                   ~Tokens   Severity      Recommended for
=====================  ========  ============  =====================================
calculator             ~20       read_only     all sizes (3B+)
get_current_time       ~15       read_only     all sizes
read_file              ~25       read_only     all sizes
list_directory         ~22       read_only     all sizes
write_file             ~28       write         5B+ (3B models mangle long content)
run_python             ~30       destructive   7B+ only; always behind confirmation
web_search             ~25       read_only     5B+ (result synthesis needs capacity)
=====================  ========  ============  =====================================

Rule of thumb: a 3B model with a 4096-token window should carry **3-4 tools
max** (~60-100 tokens of definitions); a 7B model can carry 6-8 (~150-220).

``web_search`` uses the optional ``ddgs`` package and only imports it (and
touches the network) when the tool is actually called -- never at import time.
``run_python`` executes in a subprocess with a timeout; it is classified
DESTRUCTIVE and the default policy will demand confirmation.
"""

from __future__ import annotations

import ast
import datetime
import operator
import subprocess
import sys
from pathlib import Path

from ..registry import tool
from ..safety import Severity
from ..editing import propose_edit, apply_edit

# ---------------------------------------------------------------------------
# Safe arithmetic (no eval)
# ---------------------------------------------------------------------------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Only basic arithmetic is supported.")


@tool("Evaluate an arithmetic expression, e.g. '2*(3+4)'.",
      keywords=["math", "calculate", "arithmetic", "compute", "number"])
def calculator(expression: str) -> str:
    """Safely evaluate +, -, *, /, //, %, ** on numbers (AST-based, no eval)."""
    parsed = ast.parse(expression, mode="eval")
    result = _eval_node(parsed)
    return str(int(result) if isinstance(result, float) and result.is_integer()
               else result)


@tool("Get the current date and time.",
      keywords=["time", "date", "today", "now", "clock"])
def get_current_time() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# File I/O (pathlib throughout; newline='' handling is left to Python's
# universal newlines so CRLF/LF both read transparently)
# ---------------------------------------------------------------------------

@tool("Read a text file and return its contents.",
      keywords=["file", "read", "open", "text", "contents"])
def read_file(path: str, max_chars: int = 8000) -> str:
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"
    return text


@tool("List files in a directory.",
      keywords=["directory", "folder", "list", "files", "ls"])
def list_directory(path: str = ".") -> str:
    p = Path(path).expanduser()
    entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    lines = [f"{'[dir] ' if e.is_dir() else ''}{e.name}" for e in entries[:200]]
    return "\n".join(lines) if lines else "(empty directory)"


@tool("Write text to a file (overwrites).",
      severity=Severity.WRITE,
      keywords=["file", "write", "save", "create", "output"])
def write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8", newline="\n")
    return f"Wrote {len(content)} chars to {p}"


# ---------------------------------------------------------------------------
# Code execution -- DESTRUCTIVE by design
# ---------------------------------------------------------------------------

@tool("Run a short Python snippet and return stdout.",
      severity=Severity.DESTRUCTIVE,
      keywords=["python", "code", "execute", "run", "script"],
      timeout_s=20.0)
def run_python(code: str) -> str:
    """Execute *code* in a fresh subprocess (isolated interpreter, 20s cap).

    This is arbitrary code execution: classified DESTRUCTIVE so the default
    policy requires confirmation, and read-only policies block it entirely.
    For hard isolation, run the whole framework inside a container.
    """
    proc = subprocess.run(
        [sys.executable, "-I", "-c", code],  # -I: isolated mode
        capture_output=True, text=True, timeout=20,
    )
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    if proc.returncode != 0:
        return f"Exit {proc.returncode}. stderr: {err[:1000]}"
    return out[:4000] if out else "(no output)"


# ---------------------------------------------------------------------------
# Web search (optional dependency, lazy import, only on call)
# ---------------------------------------------------------------------------

@tool("Search the web and return top result snippets.",
      keywords=["web", "search", "internet", "lookup", "news"])
def web_search(query: str, max_results: int = 3) -> str:
    """Requires the optional ``ddgs`` package. The import and the network
    request both happen only when the agent calls this tool at runtime."""
    try:
        from ddgs import DDGS  # noqa: PLC0415
    except ImportError:
        return (
            "web_search unavailable: install with "
            "'pip install my-agent-framework[web]'."
        )
    results = list(DDGS().text(query, max_results=max_results))
    if not results:
        return "No results."
    return "\n".join(
        f"- {r.get('title', '')}: {r.get('body', '')[:200]}" for r in results
    )


#: Convenience bundles by model size (see docs/MODEL_GUIDE.md).
BASIC_TOOLS = [calculator, get_current_time, read_file]                    # 3B
STANDARD_TOOLS = BASIC_TOOLS + [list_directory, write_file]                # 5B
FULL_TOOLS = STANDARD_TOOLS + [run_python, web_search, propose_edit, apply_edit]                     # 7B+
