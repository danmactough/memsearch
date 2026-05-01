#!/usr/bin/env python3
"""Manual session summarization helpers for memsearch agent plugins."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


MAX_RESULT_CHARS = int(os.environ.get("MEMSEARCH_MAX_RESULT_CHARS", "1000"))
MAX_TRANSCRIPT_CHARS = int(os.environ.get("MEMSEARCH_MAX_TRANSCRIPT_CHARS", "12000"))


@dataclass
class SessionDoc:
    path: Path
    session_id: str
    created: datetime
    date: str
    label: str
    text: str


def format_claude_transcript(path: str | Path, *, plugin_root: Path | None = None) -> str:
    formatter = _load_claude_summary_formatter(plugin_root)
    if formatter is not None:
        return formatter(path)
    return _format_claude_transcript_fallback(path)


def _format_claude_transcript_fallback(path: str | Path) -> str:
    rows = _read_jsonl(Path(path))
    output = ["=== Transcript of a conversation between a human and Claude Code ==="]
    content_count = 0

    for entry in rows:
        entry_type = entry.get("type", "")
        message = entry.get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""

        if entry_type == "user":
            if isinstance(content, str):
                text = _strip_hook_tags(content).strip()
                if text and not _is_transcript_scaffolding_line(text):
                    output.append(f"[Human]: {text}")
                    content_count += 1
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text = _strip_hook_tags(str(block.get("text", ""))).strip()
                        if text and not _is_transcript_scaffolding_line(text):
                            output.append(f"[Human]: {text}")
                            content_count += 1
                    elif block.get("type") == "tool_result":
                        result = _extract_text(block.get("content", ""))
                        result = _truncate(result, MAX_RESULT_CHARS)
                        label = "Tool error" if block.get("is_error") else "Tool output"
                        if result:
                            output.append(f"[{label}]: {result}")
                            content_count += 1

        elif entry_type == "assistant":
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")
                if block_type == "text":
                    text = str(block.get("text", "")).strip()
                    if text:
                        output.append(f"[Claude Code]: {text}")
                        content_count += 1
                elif block_type == "tool_use":
                    name = str(block.get("name", "unknown"))
                    summary = _format_mapping(block.get("input", {}))
                    output.append(f"[Claude Code calls tool]: {name}({summary})")
                    content_count += 1

    if content_count == 0:
        return ""
    return "\n".join(output).strip()


def format_codex_rollout(path: str | Path, *, last_turn_only: bool = False) -> str:
    rows = _read_jsonl(Path(path))
    if last_turn_only:
        rows = _last_codex_turn(rows)

    output = ["=== Transcript of a conversation between a human and Codex CLI ==="]
    content_count = 0

    for entry in rows:
        entry_type = entry.get("type", "")
        payload = entry.get("payload", {}) if isinstance(entry.get("payload", {}), dict) else {}

        if entry_type == "event_msg":
            message_type = payload.get("type", "")
            if message_type == "user_message":
                text = str(payload.get("message", "")).strip()
                if text:
                    output.append(f"[Human]: {text}")
                    content_count += 1
            elif message_type == "agent_message":
                text = str(payload.get("message", "")).strip()
                if text:
                    output.append(f"[Codex]: {text}")
                    content_count += 1

        elif entry_type == "response_item":
            item_type = payload.get("type", "")
            if item_type == "function_call":
                name = str(payload.get("name", "unknown"))
                args = _parse_jsonish(payload.get("arguments", ""))
                output.append(f"[Codex calls tool]: {name}({_format_mapping(args)})")
                content_count += 1
            elif item_type == "function_call_output":
                result = _truncate(str(payload.get("output", "")), MAX_RESULT_CHARS)
                if result:
                    output.append(f"[Tool output]: {result}")
                    content_count += 1

    if content_count == 0:
        return ""
    return "\n".join(output).strip()


def load_sessions(
    *,
    agent: str,
    mode: str,
    project_dir: Path,
    plugin_root: Path | None = None,
    session_id: str = "",
) -> list[SessionDoc]:
    if agent == "claude":
        paths = _claude_session_paths(project_dir)
        selected = _select_paths(paths, mode=mode, session_id=session_id, agent=agent)
        return [_claude_doc(path, plugin_root=plugin_root) for path in selected]
    if agent == "codex":
        paths = _codex_session_paths(project_dir)
        selected = _select_paths(paths, mode=mode, session_id=session_id, agent=agent)
        return [_codex_doc(path, current=(mode == "current")) for path in selected]
    raise ValueError(f"unknown agent: {agent}")


def summarize_sessions(
    *,
    agent: str,
    mode: str,
    project_dir: Path,
    memory_dir: Path,
    collection: str,
    plugin_root: Path,
    dry_run: bool = False,
    session_id: str = "",
) -> int:
    sessions = [
        doc
        for doc in load_sessions(
            agent=agent,
            mode=mode,
            project_dir=project_dir,
            plugin_root=plugin_root,
            session_id=session_id,
        )
        if doc.text.strip()
    ]
    if not sessions:
        print("No sessions found to summarize.")
        return 1

    prompt = _load_prompt(agent=agent, plugin_root=plugin_root)

    if dry_run:
        for index, doc in enumerate(sessions, start=1):
            transcript = _truncate(doc.text, MAX_TRANSCRIPT_CHARS)
            print(f"=== Dry run {index}/{len(sessions)}: {doc.label} ===")
            print(_build_llm_prompt(system_prompt=prompt, transcript=transcript))
            print("=== End dry run ===")
        return 0

    memory_dir.mkdir(parents=True, exist_ok=True)
    touched: set[Path] = set()
    summarized = 0

    for index, doc in enumerate(sessions, start=1):
        if len(sessions) > 1:
            print(f"Summarizing {index}/{len(sessions)}: {doc.label}", file=sys.stderr)
        transcript = _truncate(doc.text, MAX_TRANSCRIPT_CHARS)
        summary = _summarize(agent=agent, system_prompt=prompt, transcript=transcript)
        if not summary.strip():
            continue

        memory_file = memory_dir / f"{doc.date}.md"
        anchor = _anchor(agent=agent, doc=doc)
        with memory_file.open("a", encoding="utf-8") as handle:
            handle.write(f"### {doc.label}\n")
            if anchor:
                handle.write(f"{anchor}\n")
            handle.write(summary.strip())
            handle.write("\n\n")
        touched.add(memory_file)
        summarized += 1

    if summarized == 0:
        print("Summarization produced empty output.")
        return 1

    indexed = _index_memory(memory_dir=memory_dir, collection=collection)
    files = ", ".join(path.name for path in sorted(touched))
    if indexed:
        print(f"Summarized {summarized}/{len(sessions)} sessions. Saved to {files} and re-indexed.")
    else:
        print(f"Summarized {summarized}/{len(sessions)} sessions. Saved to {files}; re-index failed or memsearch unavailable.")
    return 0


def _claude_session_paths(project_dir: Path) -> list[Path]:
    claude_home = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
    projects_dir = claude_home / "projects"
    project_path = projects_dir / _claude_project_slug(project_dir)
    if project_path.exists():
        paths = list(project_path.glob("*.jsonl"))
        if paths:
            return sorted(paths, key=lambda path: path.stat().st_mtime)

    if not projects_dir.exists():
        return []

    matches: list[Path] = []
    for path in projects_dir.glob("*/*.jsonl"):
        cwd = _first_claude_cwd(path)
        if not cwd:
            continue
        try:
            cwd_path = Path(cwd).resolve()
            project = project_dir.resolve()
            if cwd_path == project or project in cwd_path.parents:
                matches.append(path)
        except OSError:
            continue
    return sorted(matches, key=lambda path: path.stat().st_mtime)


def _codex_session_paths(project_dir: Path) -> list[Path]:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    sessions_dir = codex_home / "sessions"
    if not sessions_dir.exists():
        return []

    matches: list[Path] = []
    for path in sessions_dir.glob("**/*.jsonl"):
        meta = _session_meta(path)
        if _is_internal_codex_session(meta):
            continue
        cwd = meta.get("cwd")
        if not cwd:
            continue
        try:
            cwd_path = Path(str(cwd)).resolve()
            project = project_dir.resolve()
            if cwd_path == project or project in cwd_path.parents:
                matches.append(path)
        except OSError:
            continue
    return sorted(matches, key=lambda path: path.stat().st_mtime)


def _select_paths(paths: Sequence[Path], *, mode: str, session_id: str, agent: str) -> list[Path]:
    if session_id:
        return [path for path in paths if _path_matches_session(path, session_id=session_id, agent=agent)]
    if mode == "all":
        return list(paths)
    if mode == "current":
        return [max(paths, key=lambda path: path.stat().st_mtime)] if paths else []
    raise ValueError(f"unknown mode: {mode}")


def _path_matches_session(path: Path, *, session_id: str, agent: str) -> bool:
    normalized = session_id.removeprefix("rollout-")
    if path.stem == session_id or path.stem.startswith(session_id):
        return True
    if agent == "codex":
        meta = _session_meta(path)
        codex_id = str(meta.get("id") or path.stem.removeprefix("rollout-"))
        return codex_id == normalized or codex_id.startswith(normalized)
    return False


def _claude_doc(path: Path, *, plugin_root: Path | None) -> SessionDoc:
    rows = _read_jsonl(path)
    created = _first_timestamp(rows) or _mtime_datetime(path)
    session_id = path.stem
    text = format_claude_transcript(path, plugin_root=plugin_root)
    return SessionDoc(
        path=path,
        session_id=session_id,
        created=created,
        date=created.date().isoformat(),
        label=_label(created, session_id),
        text=text,
    )


def _codex_doc(path: Path, *, current: bool) -> SessionDoc:
    rows = _read_jsonl(path)
    meta = _first_payload(rows, "session_meta")
    created = _parse_datetime(str(meta.get("timestamp", ""))) or _first_timestamp(rows) or _mtime_datetime(path)
    session_id = str(meta.get("id") or path.stem.removeprefix("rollout-"))
    text = format_codex_rollout(path, last_turn_only=False)
    return SessionDoc(
        path=path,
        session_id=session_id,
        created=created,
        date=created.date().isoformat(),
        label=_label(created, session_id),
        text=text,
    )


def _anchor(*, agent: str, doc: SessionDoc) -> str:
    if agent == "claude":
        return f"<!-- session:{doc.session_id} transcript:{doc.path} -->"
    if agent == "codex":
        return f"<!-- session:{doc.session_id} rollout:{doc.path} -->"
    return ""


def _build_llm_prompt(*, system_prompt: str, transcript: str) -> str:
    return f"{system_prompt}\n\nHere is the transcript:\n\n{transcript}"


def _summarize(*, agent: str, system_prompt: str, transcript: str) -> str:
    llm_prompt = _build_llm_prompt(system_prompt=system_prompt, transcript=transcript)
    if agent == "claude" and shutil.which("claude"):
        command = ["claude", "-p", "--model", "haiku", "--bare", "--no-session-persistence", "--no-chrome", llm_prompt]
        result = _run_llm(command, env={"MEMSEARCH_NO_WATCH": "1", "CLAUDECODE": ""})
        if result:
            return result
    if agent == "codex" and shutil.which("codex"):
        command = [
            "codex",
            "exec",
            "--model",
            "gpt-5.4-mini",
            "--ephemeral",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "-c",
            "features.codex_hooks=false",
            "-c",
            "model_reasoning_effort=\"low\"",
            llm_prompt,
        ]
        result = _run_llm(command, env={"MEMSEARCH_NO_WATCH": "1", "MEMSEARCH_IN_STOP_WORKER": "1"})
        if result:
            return result
    return _format_raw_bullets(transcript)


def _run_llm(command: Sequence[str], *, env: dict[str, str]) -> str:
    merged_env = os.environ.copy()
    merged_env.update(env)
    try:
        completed = subprocess.run(command, text=True, capture_output=True, timeout=60, env=merged_env, check=False)
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _load_prompt(*, agent: str, plugin_root: Path) -> str:
    agent_name = "Claude Code" if agent == "claude" else "Codex"
    prompt_file = _configured_prompt_file()
    candidates = []
    if prompt_file:
        candidates.append(prompt_file)
    candidates.append(plugin_root / "prompts" / "summarize.txt")
    candidates.append(plugin_root.parent / "_shared" / "prompts" / "summarize.txt")
    for path in candidates:
        if path and path.is_file():
            return path.read_text(encoding="utf-8").replace("{{AGENT_NAME}}", agent_name)
    return "You are a third-person note-taker. Summarize the transcript as 2-6 bullet points. Write in third person. Output ONLY bullet points."


def _configured_prompt_file() -> Path | None:
    command = _memsearch_command()
    if not command:
        return None
    try:
        completed = subprocess.run(
            [*command, "config", "get", "prompts.summarize"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    text = completed.stdout.strip()
    return Path(text) if completed.returncode == 0 and text else None


def _index_memory(*, memory_dir: Path, collection: str) -> bool:
    command = _memsearch_command()
    if not command:
        return False
    args = [*command, "index", str(memory_dir)]
    if collection:
        args.extend(["--collection", collection])
    try:
        completed = subprocess.run(args, text=True, capture_output=True, timeout=60, check=False)
    except Exception:
        return False
    return completed.returncode == 0


def _memsearch_command() -> list[str] | None:
    memsearch = shutil.which("memsearch")
    if memsearch:
        return [memsearch]
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "--from", "memsearch[onnx]", "memsearch"]
    return None


def derive_collection(project_dir: Path) -> str:
    name = project_dir.name or "memsearch"
    digest = hashlib.sha256(str(project_dir.resolve()).encode()).hexdigest()[:8]
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    return f"ms_{safe}_{digest}"


def _claude_project_slug(project_dir: Path) -> str:
    return str(project_dir.resolve()).replace("/", "-")


def _default_plugin_root(agent: str) -> Path:
    plugin_dir = "claude-code" if agent == "claude" else "codex"
    script_dir = Path(__file__).resolve().parent
    if script_dir.name == "scripts" and script_dir.parent.name == plugin_dir:
        return script_dir.parent
    return script_dir.parent / plugin_dir


def _load_claude_summary_formatter(plugin_root: Path | None):
    candidates: list[Path] = []
    if plugin_root is not None:
        candidates.append(plugin_root / "transcript.py")
    candidates.extend(
        [
            Path(__file__).resolve().parent.parent / "transcript.py",
            Path(__file__).resolve().parent.parent / "claude-code" / "transcript.py",
        ]
    )

    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("memsearch_claude_transcript", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["memsearch_claude_transcript"] = module
            spec.loader.exec_module(module)
        except Exception:
            continue
        formatter = getattr(module, "format_transcript_for_summary", None)
        if callable(formatter):
            return formatter
    return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError:
        return []
    return rows


def _last_codex_turn(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index in range(len(rows) - 1, -1, -1):
        entry = rows[index]
        payload = entry.get("payload", {}) if isinstance(entry.get("payload", {}), dict) else {}
        if entry.get("type") == "event_msg" and payload.get("type") == "task_started":
            return rows[index:]
    for index in range(len(rows) - 1, -1, -1):
        entry = rows[index]
        payload = entry.get("payload", {}) if isinstance(entry.get("payload", {}), dict) else {}
        if entry.get("type") == "event_msg" and payload.get("type") == "user_message":
            return rows[index:]
    return rows


def _is_internal_codex_session(meta: dict[str, Any]) -> bool:
    originator = str(meta.get("originator", ""))
    source = meta.get("source")
    if originator == "codex_exec" or source == "exec":
        return True
    return isinstance(source, dict) and "subagent" in source


def _first_claude_cwd(path: Path) -> str:
    for entry in _read_jsonl(path):
        cwd = entry.get("cwd")
        if isinstance(cwd, str) and cwd:
            return cwd
    return ""


def _session_meta(path: Path) -> dict[str, Any]:
    for entry in _read_jsonl(path):
        if entry.get("type") == "session_meta":
            payload = entry.get("payload", {})
            return payload if isinstance(payload, dict) else {}
    return {}


def _first_payload(rows: Iterable[dict[str, Any]], entry_type: str) -> dict[str, Any]:
    for entry in rows:
        if entry.get("type") == entry_type:
            payload = entry.get("payload", {})
            return payload if isinstance(payload, dict) else {}
    return {}


def _first_timestamp(rows: Iterable[dict[str, Any]]) -> datetime | None:
    for entry in rows:
        timestamp = entry.get("timestamp")
        if timestamp:
            parsed = _parse_datetime(str(timestamp))
            if parsed:
                return parsed
    return None


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone()
    except ValueError:
        return None


def _mtime_datetime(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone()


def _label(created: datetime, session_id: str) -> str:
    return f"{created.strftime('%H:%M')} {session_id[:8]}"


def _strip_hook_tags(text: str) -> str:
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL)
    text = re.sub(r"<local-command-\w+>.*?</local-command-\w+>", "", text, flags=re.DOTALL)
    text = re.sub(r"<command-\w+>.*?</command-\w+>", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part.strip()).strip()
    return str(content).strip()


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _format_mapping(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key, raw in value.items():
            text = str(raw).replace("\n", " ")
            parts.append(f"{key}={_truncate(text, 120)}")
        return ", ".join(parts)
    return _truncate(str(value).replace("\n", " "), 400)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def _is_transcript_scaffolding_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("=== Transcript of a conversation between"):
        return True
    if stripped.lower() == "[continuing from a previous session]":
        return True
    return re.match(r"^\[\d{2}:\d{2}:\d{2}\]\s+[0-9a-f-]{8,}", stripped, flags=re.IGNORECASE) is not None


def _format_raw_bullets(text: str) -> str:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not _is_transcript_scaffolding_line(line)
    ]
    return "\n".join(f"- {line}" for line in lines[:20])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Claude Code or Codex sessions into memsearch memory.")
    parser.add_argument("mode", nargs="?", choices=("current", "all"), default="current")
    parser.add_argument("--agent", required=True, choices=("claude", "codex"), help="Agent transcript format to read.")
    parser.add_argument("--session", default="", help="Summarize a specific session ID or prefix instead of current/all.")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--memory-dir", type=Path, default=None)
    parser.add_argument("--collection", default="")
    parser.add_argument("--plugin-root", type=Path, default=None, help="Plugin root. Defaults from --agent.")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Print the AI prompt without sending it or writing memory.")
    args = parser.parse_args(argv)

    project_dir = args.project_dir.resolve()
    memory_dir = args.memory_dir or project_dir / ".memsearch" / "memory"
    collection = args.collection or derive_collection(project_dir)
    plugin_root = (args.plugin_root or _default_plugin_root(args.agent)).resolve()
    return summarize_sessions(
        agent=args.agent,
        mode=args.mode,
        project_dir=project_dir,
        memory_dir=memory_dir,
        collection=collection,
        plugin_root=plugin_root,
        dry_run=args.dry_run,
        session_id=args.session,
    )


if __name__ == "__main__":
    raise SystemExit(main())
