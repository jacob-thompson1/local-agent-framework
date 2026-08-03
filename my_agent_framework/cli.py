"""``my-agent`` command-line interface.

Commands
--------
my-agent config show                       # merged settings (secrets excluded)
my-agent config set model ollama:mistral:7b
my-agent config set hybrid-mode true
my-agent config set anthropic-api-key sk-...   # -> OS credential manager
my-agent config get model
my-agent config unset max-tools
my-agent config reset

my-agent run "What is 12 * 7?"             # one-shot task with builtin tools
my-agent run "..." --tools calculator,read_file --dry-run

my-agent analyze                           # tool/token weight report
my-agent export-audit --session-id X --format json
my-agent export-audit --date-range 2026-01-01:2026-06-30 --redact -o out.json
my-agent audit-summary --date-range 2026-01-01:2026-06-30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from .audit import DEFAULT_REDACTION_PATTERNS, AuditExporter
from .config import DEFAULTS, Settings, agent_from_settings, config_path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _parse_date_range(text: Optional[str]) -> tuple[Optional[date], Optional[date]]:
    if not text:
        return None, None
    try:
        start, end = text.split(":")
        return date.fromisoformat(start), date.fromisoformat(end)
    except ValueError:
        raise SystemExit(
            f"Invalid --date-range {text!r}; expected YYYY-MM-DD:YYYY-MM-DD"
        )


# -- subcommand handlers -----------------------------------------------------

def cmd_config(args: argparse.Namespace) -> int:
    settings = Settings()
    if args.config_cmd == "show":
        data = settings.as_dict()
        print(json.dumps(data, indent=2))
        print(f"\n# file: {config_path()}", file=sys.stderr)
    elif args.config_cmd == "get":
        print(json.dumps(settings.get(args.key)))
    elif args.config_cmd == "set":
        if args.key not in DEFAULTS and not args.key.endswith("-api-key"):
            print(f"note: {args.key!r} is not a known setting "
                  f"(known: {sorted(DEFAULTS)})", file=sys.stderr)
        settings.set(args.key, args.value)
        if args.key.endswith("-api-key"):
            print("Stored in OS credential manager (not in the config file).")
        else:
            print(f"{args.key} = {settings.get(args.key)!r}")
    elif args.config_cmd == "unset":
        settings.unset(args.key)
        print(f"{args.key} removed (default now applies).")
    elif args.config_cmd == "reset":
        settings.reset()
        print("Settings cleared.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .tools.builtin import FULL_TOOLS  # lazy: only when running

    overrides: dict = {}
    if args.model:
        overrides["model"] = args.model
    agent = agent_from_settings(tools=FULL_TOOLS, **overrides)
    if args.dry_run:
        agent.policy.dry_run = True
    if args.yes:
        agent.policy.confirm_at_or_above = None
    subset = args.tools.split(",") if args.tools else None
    result = agent.run(args.task, tool_subset=subset)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        for step in result.steps:
            if step.tool:
                print(f"[{step.iteration}] {step.tool}({step.args}) "
                      f"-> {str(step.observation)[:120]}")
        print(f"\nStatus: {result.status}")
        print(f"Answer: {result.final_answer}")
        print(f"Session: {result.session_id}  "
              f"(~{result.total_prompt_tokens} prompt tokens, "
              f"{result.iterations} steps)")
        if result.audit_path:
            print(f"Audit log: {result.audit_path}")
    return 0 if result.status in ("success", "dry_run") else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    from .agent import _SYSTEM_TEMPLATE
    from .registry import ToolRegistry
    from .tokens import TokenCounter
    from .tools.builtin import FULL_TOOLS

    settings = Settings()
    counter = TokenCounter()
    registry = ToolRegistry(counter)
    registry.register_all(FULL_TOOLS)
    size = args.size_class or settings.get("size-class") or "7b"
    from .models import PROFILES
    profile = PROFILES.get(size, PROFILES["7b"])
    ctx = settings.get("context-window") or profile.context_window
    system_tokens = counter.count(
        _SYSTEM_TEMPLATE.format(tool_block="", few_shot="")
    )
    report = registry.analyze(size, ctx, system_prompt_tokens=system_tokens)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def cmd_export_audit(args: argparse.Namespace) -> int:
    date_from, date_to = _parse_date_range(args.date_range)
    exporter = AuditExporter(
        root=Path(args.audit_root) if args.audit_root else None,
        redact_patterns=DEFAULT_REDACTION_PATTERNS if args.redact else None,
    )
    text = exporter.export(
        fmt=args.format, session_id=args.session_id,
        date_from=date_from, date_to=date_to,
        out_path=Path(args.output) if args.output else None,
    )
    if args.output:
        print(f"Exported to {args.output}")
    else:
        print(text)
    return 0


def cmd_audit_summary(args: argparse.Namespace) -> int:
    date_from, date_to = _parse_date_range(args.date_range)
    exporter = AuditExporter(
        root=Path(args.audit_root) if args.audit_root else None
    )
    print(json.dumps(exporter.summary(date_from, date_to), indent=2))
    return 0


# -- parser ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="my-agent",
        description="Small-model-first agent framework CLI.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_config = sub.add_parser("config", help="view/edit persistent settings")
    csub = p_config.add_subparsers(dest="config_cmd", required=True)
    csub.add_parser("show")
    p_get = csub.add_parser("get")
    p_get.add_argument("key")
    p_set = csub.add_parser("set")
    p_set.add_argument("key")
    p_set.add_argument("value")
    p_unset = csub.add_parser("unset")
    p_unset.add_argument("key")
    csub.add_parser("reset")
    p_config.set_defaults(func=cmd_config)

    p_run = sub.add_parser("run", help="run a one-shot task")
    p_run.add_argument("task")
    p_run.add_argument("--model", help="override configured model spec")
    p_run.add_argument("--tools", help="comma-separated tool subset")
    p_run.add_argument("--dry-run", action="store_true",
                       help="show planned actions without executing")
    p_run.add_argument("--yes", action="store_true",
                       help="skip confirmations (use with care)")
    p_run.add_argument("--json", action="store_true",
                       help="print the full structured result as JSON")
    p_run.set_defaults(func=cmd_run)

    p_analyze = sub.add_parser(
        "analyze", help="report tool token costs vs model capacity")
    p_analyze.add_argument("--size-class", choices=["3b", "5b", "7b", "large"])
    p_analyze.set_defaults(func=cmd_analyze)

    p_export = sub.add_parser("export-audit", help="export audit logs")
    p_export.add_argument("--session-id")
    p_export.add_argument("--date-range", help="YYYY-MM-DD:YYYY-MM-DD")
    p_export.add_argument("--format", choices=["json", "csv"], default="json")
    p_export.add_argument("--redact", action="store_true",
                          help="apply built-in PII redaction patterns")
    p_export.add_argument("-o", "--output", help="write to file")
    p_export.add_argument("--audit-root", help="override audit directory")
    p_export.set_defaults(func=cmd_export_audit)

    p_summary = sub.add_parser("audit-summary", help="aggregate audit statistics")
    p_summary.add_argument("--date-range", help="YYYY-MM-DD:YYYY-MM-DD")
    p_summary.add_argument("--audit-root")
    p_summary.set_defaults(func=cmd_audit_summary)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
