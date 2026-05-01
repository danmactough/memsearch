from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "plugins" / "_shared" / "session_summarizer.py"
spec = importlib.util.spec_from_file_location("session_summarizer", MODULE_PATH)
assert spec and spec.loader
session_summarizer = importlib.util.module_from_spec(spec)
sys.modules["session_summarizer"] = session_summarizer
spec.loader.exec_module(session_summarizer)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_format_claude_transcript_includes_user_assistant_tools(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    write_jsonl(
        transcript,
        [
            {
                "type": "user",
                "uuid": "u1",
                "timestamp": "2026-04-30T10:00:00Z",
                "message": {"content": "Fix the failing tests"},
            },
            {
                "type": "assistant",
                "uuid": "a1",
                "timestamp": "2026-04-30T10:01:00Z",
                "message": {
                    "content": [
                        {"type": "text", "text": "I will inspect them."},
                        {"type": "tool_use", "name": "Bash", "input": {"command": "pytest -q"}},
                    ]
                },
            },
            {
                "type": "user",
                "uuid": "tool-result",
                "timestamp": "2026-04-30T10:02:00Z",
                "message": {"content": [{"type": "tool_result", "content": "1 failed", "is_error": True}]},
            },
        ],
    )

    formatted = session_summarizer.format_claude_transcript(transcript)

    assert "Fix the failing tests" in formatted
    assert "**Assistant**: I will inspect them." in formatted
    assert "Tools: Bash(pytest -q)" in formatted
    assert "1 failed" not in formatted


def test_empty_transcripts_are_ignored(tmp_path: Path) -> None:
    transcript = tmp_path / "empty.jsonl"
    write_jsonl(transcript, [{"type": "session_meta", "payload": {"cwd": str(tmp_path)}}])

    assert session_summarizer.format_claude_transcript(transcript) == ""
    assert session_summarizer.format_codex_rollout(transcript) == ""


def test_format_codex_rollout_can_render_all_or_last_turn(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.jsonl"
    write_jsonl(
        rollout,
        [
            {"type": "session_meta", "payload": {"id": "s1", "timestamp": "2026-04-30T10:00:00Z", "cwd": str(tmp_path)}},
            {"type": "event_msg", "payload": {"type": "task_started"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "first question"}},
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "first answer"}},
            {"type": "event_msg", "payload": {"type": "task_started"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "second question"}},
            {"type": "response_item", "payload": {"type": "function_call", "name": "shell", "arguments": '{"cmd": "ls"}'}},
            {"type": "response_item", "payload": {"type": "function_call_output", "output": "file.txt"}},
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "second answer"}},
        ],
    )

    full = session_summarizer.format_codex_rollout(rollout)
    last = session_summarizer.format_codex_rollout(rollout, last_turn_only=True)

    assert "[Human]: first question" in full
    assert "[Codex]: second answer" in full
    assert "[Human]: first question" not in last
    assert "[Human]: second question" in last
    assert "[Codex calls tool]: shell(cmd=ls)" in last
    assert "[Tool output]: file.txt" in last


def test_load_sessions_selects_specific_codex_session_by_id_prefix(tmp_path: Path, monkeypatch) -> None:
    codex_home = tmp_path / ".codex"
    session_dir = codex_home / "sessions" / "2026" / "04" / "30"
    session_dir.mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()

    first_rollout = session_dir / "rollout-first.jsonl"
    second_rollout = session_dir / "rollout-second.jsonl"
    write_jsonl(
        first_rollout,
        [
            {"type": "session_meta", "payload": {"id": "019-first", "timestamp": "2026-04-30T10:00:00Z", "cwd": str(project), "originator": "codex_cli_rs", "source": "cli"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "first"}},
        ],
    )
    write_jsonl(
        second_rollout,
        [
            {"type": "session_meta", "payload": {"id": "019-second", "timestamp": "2026-04-30T10:01:00Z", "cwd": str(project), "originator": "codex_cli_rs", "source": "cli"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "second"}},
        ],
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    sessions = session_summarizer.load_sessions(
        agent="codex",
        mode="all",
        project_dir=project,
        session_id="019-sec",
    )

    assert [session.session_id for session in sessions] == ["019-second"]


def test_default_plugin_root_uses_agent_from_shared_and_packaged_paths() -> None:
    assert session_summarizer._default_plugin_root("claude").name == "claude-code"
    assert session_summarizer._default_plugin_root("codex").name == "codex"


def test_dry_run_dumps_prompt_without_writing_or_summarizing(tmp_path: Path, monkeypatch, capsys) -> None:
    codex_home = tmp_path / ".codex"
    session_dir = codex_home / "sessions" / "2026" / "04" / "30"
    session_dir.mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()
    rollout = session_dir / "rollout-user.jsonl"
    write_jsonl(
        rollout,
        [
            {"type": "session_meta", "payload": {"id": "user", "timestamp": "2026-04-30T10:00:00Z", "cwd": str(project), "originator": "codex_cli_rs", "source": "cli"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "keep me"}},
        ],
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(session_summarizer, "_load_prompt", lambda **_kwargs: "PROMPT")

    def fail_summarize(**_kwargs):
        raise AssertionError("dry-run must not call the AI")

    monkeypatch.setattr(session_summarizer, "_summarize", fail_summarize)

    code = session_summarizer.summarize_sessions(
        agent="codex",
        mode="all",
        project_dir=project,
        memory_dir=tmp_path / "memory",
        collection="test",
        plugin_root=tmp_path,
        dry_run=True,
    )

    output = capsys.readouterr().out
    assert code == 0
    assert "PROMPT" in output
    assert "Here is the transcript:" in output
    assert "[Human]: keep me" in output
    assert not (tmp_path / "memory").exists()


def test_codex_session_discovery_skips_internal_exec_and_subagent_rollouts(tmp_path: Path, monkeypatch) -> None:
    codex_home = tmp_path / ".codex"
    session_dir = codex_home / "sessions" / "2026" / "04" / "30"
    session_dir.mkdir(parents=True)
    project = tmp_path / "project"
    project.mkdir()

    user_rollout = session_dir / "rollout-user.jsonl"
    exec_rollout = session_dir / "rollout-exec.jsonl"
    subagent_rollout = session_dir / "rollout-subagent.jsonl"

    write_jsonl(
        user_rollout,
        [
            {"type": "session_meta", "payload": {"id": "user", "timestamp": "2026-04-30T10:00:00Z", "cwd": str(project), "originator": "codex_cli_rs", "source": "cli"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "keep me"}},
        ],
    )
    write_jsonl(
        exec_rollout,
        [
            {"type": "session_meta", "payload": {"id": "exec", "timestamp": "2026-04-30T10:01:00Z", "cwd": str(project), "originator": "codex_exec", "source": "exec"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "skip me"}},
        ],
    )
    write_jsonl(
        subagent_rollout,
        [
            {"type": "session_meta", "payload": {"id": "subagent", "timestamp": "2026-04-30T10:02:00Z", "cwd": str(project), "originator": "codex_cli_rs", "source": {"subagent": "review"}}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "skip me too"}},
        ],
    )

    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    sessions = session_summarizer.load_sessions(agent="codex", mode="all", project_dir=project)

    assert [session.session_id for session in sessions] == ["user"]
