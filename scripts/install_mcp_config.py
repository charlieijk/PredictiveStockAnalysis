#!/usr/bin/env python3
"""Helper to install the Codex MCP entry for this repository."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


MCP_HEADER = "[mcp_servers.predictive_stock_server]"


def _load_template(repo_root: Path) -> str:
    """Read the MCP config template and substitute the repo path placeholder."""
    template_path = repo_root / ".codex" / "config.toml"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    raw_template = template_path.read_text()
    return raw_template.replace("__PROJECT_ROOT__", str(repo_root))


def _remove_existing_block(content: str) -> str:
    """Drop the predictive_stock_server block if it already exists."""
    pattern = re.compile(
        r"(?ms)^\[mcp_servers\.predictive_stock_server\]\n.*?(?=^\[|\Z)"
    )
    return pattern.sub("", content).rstrip()


def install(force: bool) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    rendered_template = _load_template(repo_root).strip() + "\n"

    target_dir = Path.home() / ".config" / "codex"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "config.toml"

    if target_path.exists():
        current = target_path.read_text()
        if MCP_HEADER in current and not force:
            print("predictive_stock_server already configured; use --force to replace it.")
            return target_path

        trimmed = _remove_existing_block(current) if MCP_HEADER in current else current.rstrip()
        if trimmed:
            trimmed += "\n\n"
        target_path.write_text(trimmed + rendered_template)
    else:
        target_path.write_text(rendered_template)

    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the predictive_stock_server MCP entry for Codex CLI."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing predictive_stock_server entry if present.",
    )
    args = parser.parse_args()

    target_path = install(force=args.force)
    print(f"MCP configuration written to: {target_path}")
    print("Restart Codex CLI (or reload MCPs) to pick up the new tool entry.")


if __name__ == "__main__":
    main()
