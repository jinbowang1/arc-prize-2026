#!/usr/bin/env python3
"""One asynchronous NVFP4 curator for ARC3 cross-game guidance.

The process owns one persistent chat stream against the gameplay vLLM server.
It never blocks gameplay: new evidence coalesces by game while the sole request
is running, and the complete replacement ledger is published atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


THEME_SYSTEM = """You are the single persistent curator for 28 parallel ARC3 game agents.
Maintain a compact ledger of genuinely cross-game themes that may transfer to future games.
You receive at most ten newly observed frames per turn plus the current ledger. Compare games;
do not summarize individual games. Admit a claim only when multiple independent games support
it, make it falsifiable, say how a gameplay agent should test it, and include a caution against
overgeneralization. Return JSON only with this schema:
{"entries":[{"id":"T1","claim":"...","category":"game_mechanics|hud_progress|interaction|animation_state|failure_pattern|transfer_strategy|other","evidence_games":["game-id", "..."],"confidence":"low|medium|high","why_helpful":"...","test":"...","caution":"..."}]}
Return the complete replacement ledger, reusing stable IDs. Prefer zero strong entries over weak
or game-specific advice. Keep at most six concise entries. Keep each claim, why_helpful, test,
and caution field under 45 words."""


WORLD_MODEL_SYSTEM = """You are the single persistent curator for 28 parallel ARC3 game agents.
Each gameplay agent maintains a working world model. Continuously synthesize those observed world
models into compact, transferable priors for future games. Focus on recurring human-designed
mechanics, action-effect patterns, goal/HUD semantics, failure modes, and discriminating tests.
Do not merge incompatible mechanics into a universal rule and do not retell individual games.
Return JSON only with this schema:
{"entries":[{"id":"W1","claim":"...","category":"world_dynamics|action_model|goal_model|hud_progress|failure_pattern|transfer_test|other","evidence_games":["game-id", "..."],"confidence":"low|medium|high","why_helpful":"...","test":"...","caution":"..."}]}
Return the complete replacement ledger, reusing stable IDs. Prefer zero strong entries over weak
advice. Keep at most six concise entries. Keep each claim, why_helpful, test, and caution field
under 45 words."""


WORLD_MODEL_RE = re.compile(
    r"Working world model carried from earlier turns:\s*(.*?)\s*End of carried world model\.",
    flags=re.DOTALL,
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (compact_json(row) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    decoder = json.JSONDecoder()
    while start >= 0:
        try:
            value, _ = decoder.raw_decode(stripped[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        start = stripped.find("{", start + 1)
    raise ValueError("curator response did not contain a JSON object")


def game_id_from_path(path: Path) -> str:
    name = path.name
    for suffix in ("_events.jsonl", "_requests.jsonl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


class JsonlTail:
    def __init__(self) -> None:
        self.offsets: dict[str, int] = {}

    def read(self, path: Path) -> list[dict[str, Any]]:
        key = str(path)
        size = path.stat().st_size
        offset = min(self.offsets.get(key, 0), size)
        rows: list[dict[str, Any]] = []
        with path.open("rb") as handle:
            handle.seek(offset)
            while True:
                line_start = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    handle.seek(line_start)
                    break
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(value, dict):
                    rows.append(value)
            self.offsets[key] = handle.tell()
        return rows


class Curator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.output_dir / "ledger.json"
        self.health_path = self.output_dir / "health.json"
        self.request_log = self.output_dir / "curator-requests.jsonl"
        self.revision_log = self.output_dir / "ledger-revisions.jsonl"
        self.tail = JsonlTail()
        self.pending: dict[str, dict[str, Any]] = {}
        self.last_evidence_hash: dict[str, str] = {}
        self.messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": THEME_SYSTEM if args.mode == "themes" else WORLD_MODEL_SYSTEM,
            }
        ]
        self.revision = 0
        self.entries: list[dict[str, Any]] = []
        self.requests_started = 0
        self.requests_completed = 0
        self.failures = 0
        self.evidence_seen = 0
        self.games_seen: set[str] = set()
        self.started_at = utc_now()
        self._write_ledger([])

    @property
    def influence_mode(self) -> str:
        if self.args.mode == "themes":
            return "nvfp4_persistent_theme_curator_to_gameplay"
        return "nvfp4_persistent_world_models_to_gameplay"

    def _write_ledger(self, entries: list[dict[str, Any]]) -> None:
        payload = {
            "revision": self.revision,
            "updated_at": utc_now(),
            "influence_mode": self.influence_mode,
            "curator_model": self.args.model,
            "curator_topology": "one persistent asynchronous NVFP4 request stream",
            "observer_workers": 0,
            "reviewer_workers": 1,
            "games_observed_total": len(self.games_seen),
            "frames_observed_total": self.evidence_seen if self.args.mode == "themes" else 0,
            "world_models_observed_total": self.evidence_seen if self.args.mode == "world_models" else 0,
            "themes": entries,
        }
        atomic_write_json(self.ledger_path, payload)

    def write_health(self, status: str, error: str | None = None) -> None:
        payload = {
            "status": status,
            "mode": self.args.mode,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": utc_now(),
            "revision": self.revision,
            "pending_games": sorted(self.pending),
            "games_seen": len(self.games_seen),
            "evidence_seen": self.evidence_seen,
            "requests_started": self.requests_started,
            "requests_completed": self.requests_completed,
            "failures": self.failures,
            "persistent_message_count": len(self.messages),
            "error": error,
        }
        atomic_write_json(self.health_path, payload)

    def _queue(self, game_id: str, evidence: dict[str, Any]) -> None:
        digest = hashlib.sha256(compact_json(evidence).encode("utf-8")).hexdigest()
        if self.last_evidence_hash.get(game_id) == digest:
            return
        self.last_evidence_hash[game_id] = digest
        self.pending[game_id] = evidence
        self.games_seen.add(game_id)
        self.evidence_seen += 1

    def collect_themes(self) -> None:
        for path in sorted(self.args.events_dir.glob("*_events.jsonl")):
            game_id = game_id_from_path(path)
            latest: dict[str, Any] | None = None
            for row in self.tail.read(path):
                if not row.get("board_ascii"):
                    continue
                useful = (
                    row.get("type") == "initial"
                    or bool(row.get("board_changed"))
                    or bool(row.get("level_completed"))
                    or bool(row.get("run_complete"))
                    or float(row.get("reward") or 0) > 0
                )
                if not useful:
                    continue
                latest = {
                    "game_id": game_id,
                    "frame_label": f"{game_id}:a{row.get('action_num', 0)}:l{row.get('level', 1)}",
                    "action": row.get("action_display") or row.get("action_name") or "RESET",
                    "level": row.get("level"),
                    "reward": row.get("reward"),
                    "level_completed": bool(row.get("level_completed")),
                    "board_ascii": str(row.get("board_ascii"))[:5000],
                }
            if latest is not None:
                self._queue(game_id, latest)

    def collect_world_models(self) -> None:
        for path in sorted(self.args.events_dir.glob("*_requests.jsonl")):
            game_id = game_id_from_path(path)
            latest: dict[str, Any] | None = None
            for row in self.tail.read(path):
                if row.get("event") not in {"request", "response"}:
                    continue
                messages = row.get("messages")
                if not isinstance(messages, list):
                    continue
                carried = ""
                state_line = ""
                for message in reversed(messages):
                    if not isinstance(message, dict) or message.get("role") != "user":
                        continue
                    text = message_text(message)
                    match = WORLD_MODEL_RE.search(text)
                    if match:
                        carried = match.group(1).strip()
                        break
                for message in reversed(messages):
                    if not isinstance(message, dict) or message.get("role") != "user":
                        continue
                    state_match = re.search(r"Current state:[^\n]+", message_text(message))
                    if state_match:
                        state_line = state_match.group(0)
                        break
                # The compact-English gameplay model often expresses its current
                # world model as ordinary assistant analysis instead of emitting
                # the optional structured scientist-note headings. In that case,
                # summarize the latest narrative analyses rather than falling back
                # to raw frames. This keeps the 29th thread grounded in what each
                # gameplay agent actually believes.
                if not carried or "No carried world model" in carried:
                    analyses: list[str] = []
                    for message in reversed(messages):
                        if not isinstance(message, dict) or message.get("role") != "assistant":
                            continue
                        text = message_text(message).strip()
                        if text:
                            analyses.append(text)
                        if len(analyses) >= 3:
                            break
                    carried = "\n\n".join(reversed(analyses))[-6000:]
                if not carried:
                    continue
                latest = {
                    "game_id": game_id,
                    "action": row.get("action"),
                    "analysis_step": row.get("analysis_step"),
                    "state": state_line,
                    "working_world_model": carried[:6000],
                }
            if latest is not None:
                self._queue(game_id, latest)

    def collect(self) -> None:
        if self.args.mode == "themes":
            self.collect_themes()
        else:
            self.collect_world_models()

    def _user_prompt(self, evidence: list[dict[str, Any]]) -> str:
        kind = "new frame evidence" if self.args.mode == "themes" else "new per-game working world models"
        # The on-disk/injection schema uses theme_id/theme/support_games, while the
        # model-facing schema uses id/claim/evidence_games. Feed the model only its
        # declared schema so it does not learn to echo the storage representation.
        model_entries = [
            {
                "id": entry.get("theme_id"),
                "claim": entry.get("theme"),
                "category": entry.get("category"),
                "evidence_games": entry.get("support_games", []),
                "confidence": entry.get("confidence"),
                "why_helpful": entry.get("why_helpful"),
                "test": "",
                "caution": entry.get("caution"),
            }
            for entry in self.entries
        ]
        return (
            f"Ledger revision {self.revision}. Current complete ledger:\n"
            f"{json.dumps({'entries': model_entries}, ensure_ascii=False, indent=2)}\n\n"
            f"Here are {len(evidence)} items of {kind}, at most one latest item per game:\n"
            f"{json.dumps(evidence, ensure_ascii=False, indent=2)}\n\n"
            "Return the complete replacement ledger as JSON only."
        )

    def _call_model(self, prompt: str) -> tuple[str, dict[str, Any]]:
        request_messages = [*self.messages, {"role": "user", "content": prompt}]
        payload = {
            "model": self.args.model,
            "messages": request_messages,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "max_tokens": self.args.max_tokens,
            "response_format": {"type": "json_object"},
            # Curator output must be compact JSON. With thinking enabled Qwen can
            # consume the entire output budget in reasoning_content and return an
            # empty final content field, which produces no usable ledger.
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
        }
        encoded = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.args.base_url.rstrip("/") + "/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
        )
        started = time.time()
        with urllib.request.urlopen(request, timeout=self.args.request_timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        choice = result["choices"][0]
        finish_reason = str(choice.get("finish_reason") or "")
        if finish_reason == "length":
            raise ValueError("curator response reached max_tokens and was not publishable")
        message = choice["message"]
        content = str(message.get("content") or "")
        if not content:
            raise ValueError("curator returned empty assistant content")
        return content, {
            "wall_seconds": time.time() - started,
            "usage": result.get("usage"),
            "finish_reason": finish_reason,
            "reasoning_chars": len(str(message.get("reasoning_content") or message.get("reasoning") or "")),
        }

    def _sanitize(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        source = payload.get("entries", [])
        if not isinstance(source, list):
            raise ValueError("curator JSON entries must be a list")
        prefix = "T" if self.args.mode == "themes" else "W"
        entries: list[dict[str, Any]] = []
        used: set[str] = set()
        for raw in source:
            if not isinstance(raw, dict):
                continue
            # Be tolerant of the storage schema as a defensive fallback. Earlier
            # curators occasionally echoed theme/theme_id/support_games after the
            # storage-form ledger appeared in context, which silently sanitized a
            # useful response to an empty ledger.
            claim = str(raw.get("claim") or raw.get("theme") or "").strip()[:420]
            if not claim:
                continue
            proposed = str(raw.get("id") or raw.get("theme_id") or "").strip().upper()
            if not re.fullmatch(rf"{prefix}\d+", proposed) or proposed in used:
                number = 1
                while f"{prefix}{number}" in used:
                    number += 1
                proposed = f"{prefix}{number}"
            used.add(proposed)
            games = raw.get("evidence_games", raw.get("support_games", []))
            if not isinstance(games, list):
                games = []
            games = sorted({str(game).strip() for game in games if str(game).strip()})[:25]
            confidence = str(raw.get("confidence") or "low").lower()
            if confidence not in {"low", "medium", "high"}:
                confidence = "low"
            helpful = str(raw.get("why_helpful") or "").strip()[:300]
            test = str(raw.get("test") or "").strip()[:280]
            if test:
                helpful = f"{helpful} Test: {test}".strip()
            entries.append(
                {
                    "theme_id": proposed,
                    "theme": claim,
                    "category": str(raw.get("category") or "other").strip()[:80],
                    "support_games": games,
                    "confidence": confidence,
                    "why_helpful": helpful,
                    "caution": str(raw.get("caution") or "").strip()[:260],
                }
            )
            if len(entries) >= self.args.max_entries:
                break
        return entries

    def synthesize(self) -> bool:
        if len(self.pending) < self.args.min_games:
            return False
        selected_games = sorted(self.pending)[: self.args.max_evidence]
        evidence = [self.pending.pop(game) for game in selected_games]
        prompt = self._user_prompt(evidence)
        self.requests_started += 1
        request_row = {
            "event": "request",
            "started_at": utc_now(),
            "mode": self.args.mode,
            "revision_before": self.revision,
            "evidence_games": selected_games,
            "evidence_count": len(evidence),
            "persistent_message_count": len(self.messages),
            "prompt_chars": len(prompt),
        }
        append_jsonl(self.request_log, request_row)
        try:
            content, metadata = self._call_model(prompt)
            entries = self._sanitize(extract_json_object(content))
        except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
            self.failures += 1
            for item in evidence:
                self.pending[str(item["game_id"])] = item
            append_jsonl(
                self.request_log,
                {**request_row, "event": "failure", "failed_at": utc_now(), "error": repr(exc)},
            )
            self.write_health("degraded", repr(exc))
            return False

        before_hash = hashlib.sha256(compact_json(self.entries).encode("utf-8")).hexdigest()
        after_hash = hashlib.sha256(compact_json(entries).encode("utf-8")).hexdigest()
        self.revision += 1
        self.entries = entries
        self.requests_completed += 1
        self.messages.extend(
            [
                {
                    "role": "user",
                    "content": (
                        f"Curator revision {self.revision} incorporated fresh evidence from "
                        f"{', '.join(selected_games)}. The complete replacement ledger follows."
                    ),
                },
                {"role": "assistant", "content": content},
            ]
        )
        # Preserve one stable system prefix plus at most three *compact* completed
        # turns. Never retain the raw 10-frame prompt: three such prompts exceed
        # the model's 65k context even though each individual synthesis fits.
        self.messages = [self.messages[0], *self.messages[1:][-6:]]
        self._write_ledger(entries)
        revision_row = {
            "event": "completed",
            "completed_at": utc_now(),
            "mode": self.args.mode,
            "revision_after": self.revision,
            "evidence_games": selected_games,
            "entry_count": len(entries),
            "ledger_hash_before": before_hash,
            "ledger_hash_after": after_hash,
            **metadata,
        }
        append_jsonl(self.request_log, {**request_row, **revision_row})
        append_jsonl(self.revision_log, revision_row)
        self.write_health("healthy")
        return True

    def run(self) -> int:
        self.write_health("starting")
        while True:
            try:
                failures_before = self.failures
                self.collect()
                completed = self.synthesize()
                if not completed and self.failures == failures_before:
                    self.write_health("healthy")
            except Exception as exc:  # noqa: BLE001 - daemon must remain observable
                self.failures += 1
                self.write_health("degraded", repr(exc))
                append_jsonl(
                    self.request_log,
                    {"event": "loop_failure", "failed_at": utc_now(), "error": repr(exc)},
                )
            time.sleep(self.args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("themes", "world_models"), required=True)
    parser.add_argument("--events-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--model", default="unsloth/Qwen3.8-27B-NVFP4")
    parser.add_argument("--max-evidence", type=int, default=10)
    parser.add_argument("--min-games", type=int, default=3)
    parser.add_argument("--max-entries", type=int, default=6)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--max-tokens", type=int, default=3600)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    return Curator(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
