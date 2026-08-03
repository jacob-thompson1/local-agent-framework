# Cross-Platform Guide

The framework is pure Python (3.10+) with no OS-specific bindings. Paths use `pathlib` throughout, text I/O is explicit UTF-8 with universal-newline reads (LF and CRLF both work), config/audit locations come from `platformdirs`, and subprocess use (`run_python`) invokes `sys.executable` directly with an argument list — no shell, so no shell-quoting differences between platforms. All dependencies are pure-Python or ship wheels for Windows, macOS, and Linux. CI runs the full test suite on all three OSes across Python 3.10–3.12 (`.github/workflows/ci.yml`).

## One installation method, three platforms

```
pip install my-agent-framework[ollama]
```

Use a virtual environment on every platform. Windows: `py -m venv .venv && .venv\Scripts\activate`. macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`.

## Where things live

| | Config | Audit logs | Secrets |
|---|---|---|---|
| Linux | `~/.config/my-agent-framework/config.json` | `~/.local/share/my-agent-framework/audit/` | Secret Service (GNOME Keyring/KWallet) |
| macOS | `~/Library/Application Support/my-agent-framework/config.json` | `~/Library/Application Support/my-agent-framework/audit/` | Keychain |
| Windows | `%APPDATA%\my-agent-framework\config.json` | `%LOCALAPPDATA%\my-agent-framework\audit\` | Credential Manager |

`my-agent config show` prints the active config path; `AuditExporter(root=...)` and the `audit-root` setting override the audit location (e.g., to a mounted compliance share).

## Per-OS notes

**Windows.** Install Python from python.org or `winget install Python.Python.3.12` and check "Add to PATH". Ollama installs as a native app and serves on `http://localhost:11434` like everywhere else. If console output garbles non-ASCII, set `PYTHONUTF8=1`. Long paths: enable Win32 long paths via Group Policy if your tool arguments produce >260-char paths. Headless servers (no user profile loaded) may lack Credential Manager access — use `MY_AGENT_*_API_KEY` environment variables instead of keyring there.

**macOS.** `brew install ollama && brew services start ollama` (or the app). First keyring use prompts a Keychain access dialog once — approve it for the Python binary in your venv. Apple Silicon runs 7B Q4 models comfortably on 16 GB.

**Linux.** `curl -fsSL https://ollama.com/install.sh | sh` (review it first, per your own policy). Headless servers usually lack a Secret Service daemon; either install `gnome-keyring` + `dbus`, or skip keyring and use environment variables — the framework treats env vars as first-class and never falls back to plain-text secret storage.

## Troubleshooting

**"Connection refused" to Ollama** — the daemon isn't running (`ollama serve`, or check `systemctl status ollama` / the tray app), or it's bound to a non-default host. The framework respects `OLLAMA_HOST` via `langchain-ollama`, or pass `model_kwargs={"base_url": "http://host:11434"}`. In Docker or WSL2, `localhost` is the container/VM, not your machine — use `host.docker.internal` (Docker Desktop) or the Windows host IP (WSL2).

**Mangled characters in files or logs** — some file was written by another program in a legacy encoding. The framework reads with `errors="replace"` so it degrades instead of crashing; convert the source file to UTF-8 for clean results. On Windows consoles, `PYTHONUTF8=1`.

**Path errors in tool arguments** — models sometimes emit backslashes that JSON-escape badly (`"C:\new"` → newline). Built-in file tools accept forward slashes on Windows (`C:/Users/...`), which sidesteps the issue; tell users to phrase paths that way, or normalize in your custom tools with `Path(arg)`.

**Keyring errors** — `RuntimeError: Storing API keys requires 'keyring'` means the extra isn't installed (`pip install my-agent-framework[keyring]`); `NoKeyringError` on Linux means no Secret Service backend — use env vars.

**Line endings in written files** — `write_file` writes `\n` explicitly for byte-identical output across OSes. If a downstream Windows tool insists on CRLF, wrap or replace the tool; reads are unaffected either way.

## Docker

The framework itself needs no container, but containerizing is a clean way to sandbox `run_python`. A minimal image is `python:3.12-slim` + `pip install my-agent-framework[all]`; run Ollama as a sibling container or on the host, and point the agent at it via `base_url`/`OLLAMA_HOST` (remember the `host.docker.internal` note above). Docker Desktop behaves identically for this purpose on all three OSes; mount a host volume over the audit directory if logs must survive the container.
