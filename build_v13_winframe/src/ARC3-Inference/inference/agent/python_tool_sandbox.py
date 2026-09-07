"""Lightweight isolated runner for analyzer Python tool calls."""
from __future__ import annotations

import inspect
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import textwrap
import time
from typing import Any, Callable

from inference.utils import segmentation as _segmentation
from inference.utils.grid_utils import ARC_COLOR_CHARS


_SANDBOX_BOOTSTRAP = textwrap.dedent(
    r"""
    import builtins
    import contextlib
    import io
    import json
    import os
    import sys
    import traceback

    try:
        import resource
    except ImportError:  # pragma: no cover
        resource = None

    COLOR_CHARS = ""

    __SEGMENTATION_SOURCE__

    HOST_STDOUT = sys.stdout

    SAFE_MODULES = {
        "bisect",
        "collections",
        "copy",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
    }
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "complex",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "oct",
        "ord",
        "pow",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "TypeError",
        "type",
        "ValueError",
        "RuntimeError",
        "zip",
    }


    def _send(payload):
        HOST_STDOUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
        HOST_STDOUT.flush()


    def _recv():
        line = sys.stdin.readline()
        if not line:
            raise EOFError("sandbox input closed")
        return json.loads(line)


    class FrameView:
        def __init__(self, *, ascii, step, level, shape, grid):
            self.ascii = ascii
            self.step = step
            self.level = level
            self.shape = tuple(shape)
            self._grid = grid
            self._segmentation = None

        @property
        def segmentation(self):
            if self._segmentation is None:
                self._segmentation = segment_layer(self._grid, COLOR_CHARS)
            return self._segmentation

        def __str__(self):
            rows, cols = self.shape
            return f"AsciiFrameView(level={self.level}, step={self.step}, shape={rows}x{cols})"

        __repr__ = __str__


    class HistoryEntryView:
        def __init__(self, *, action, frame):
            self.action = action
            self.frame = frame

        def __str__(self):
            return f"AsciiHistoryEntryView(action={self.action!r}, frame={self.frame})"

        __repr__ = __str__


    class TransitionView:
        def __init__(self, *, action, before_frame, after_frame, result):
            self.action = action
            self.before_frame = before_frame
            self.after_frame = after_frame
            self.frame = after_frame
            self.result = dict(result) if isinstance(result, dict) else {}

        def __str__(self):
            return (
                "ActionTransitionView("
                f"action={self.action!r}, "
                f"before_frame={self.before_frame}, "
                f"after_frame={self.after_frame})"
            )

        __repr__ = __str__


    def _sample_animation_records(records, limit):
        maximum = max(2, int(limit))
        if len(records) <= maximum:
            return list(records)
        positions = sorted(
            set(
                int(round(index * (len(records) - 1) / (maximum - 1)))
                for index in range(maximum)
            )
        )
        return [records[index] for index in positions]


    def _reduce_animation_grids(grids, block_size):
        if not grids:
            return []
        rows = len(grids[0])
        cols = len(grids[0][0]) if rows else 0
        block = max(1, min(8, int(block_size)))
        outputs = [[] for _ in grids]
        for row_start in range(0, rows, block):
            row_end = min(rows, row_start + block)
            output_rows = [[] for _ in grids]
            for col_start in range(0, cols, block):
                col_end = min(cols, col_start + block)
                candidates = [
                    (row, col)
                    for row in range(row_start, row_end)
                    for col in range(col_start, col_end)
                ]
                centre_row = (row_start + row_end - 1) / 2.0
                centre_col = (col_start + col_end - 1) / 2.0
                chosen = min(
                    candidates,
                    key=lambda coordinate: (
                        -sum(
                            grids[index][coordinate[0]][coordinate[1]]
                            != grids[index - 1][coordinate[0]][coordinate[1]]
                            for index in range(1, len(grids))
                        ),
                        abs(coordinate[0] - centre_row)
                        + abs(coordinate[1] - centre_col),
                        coordinate,
                    ),
                )
                for index, grid in enumerate(grids):
                    output_rows[index].append(int(grid[chosen[0]][chosen[1]]))
            for index in range(len(grids)):
                outputs[index].append(output_rows[index])
        return outputs


    def _animation_visible_transitions(grids):
        return sum(
            previous != current
            for previous, current in zip(grids, grids[1:])
        )


    def _render_animation_inspection(region, records, bounds, reduced, block):
        top, bottom, left, right = bounds
        source_rows = bottom - top + 1
        source_cols = right - left + 1
        output_rows = len(reduced[0]) if reduced else 0
        output_cols = len(reduced[0][0]) if output_rows else 0
        visible = _animation_visible_transitions(reduced)
        lines = [
            f"LOCAL ANIMATION REGION {region.id} - ON-DEMAND INSPECTION",
            f"Likely behavior: {region.behavior}.",
            (
                f"Requested absolute crop: rows {top}-{bottom}, cols {left}-{right} "
                f"({source_rows}x{source_cols})."
            ),
            (
                f"Adaptive aspect-preserving scale: one uniform {block}x{block} "
                f"source block per output cell, producing {output_rows}x{output_cols}. "
                "The scale never exceeds 8x8 and neither axis is stretched."
            ),
            f"Visible transitions after reduction: {visible}.",
            "Use the settled board for exact final coordinates.",
        ]
        for record, grid in zip(records, reduced):
            index = int(record.get("index") or 0)
            label = "before" if index < 0 else f"f{index}"
            changed_bbox = tuple(record.get("changed_bbox") or [])
            detail = ""
            if len(changed_bbox) == 4:
                height = changed_bbox[1] - changed_bbox[0] + 1
                width = changed_bbox[3] - changed_bbox[2] + 1
                detail = (
                    f" changed-region rows {changed_bbox[0]}-{changed_bbox[1]}, "
                    f"cols {changed_bbox[2]}-{changed_bbox[3]}, size {height}x{width}"
                )
            lines.append(f"[{label}]{detail}")
            lines.extend(
                "".join(COLOR_CHARS[max(0, min(15, int(cell)))] for cell in row)
                for row in grid
            )
        text = "\n".join(lines)
        return text, max(1, (len(json.dumps(text, ensure_ascii=True)) + 2) // 3), visible


    class AnimationRegionView:
        def __init__(self, payload):
            self.id = int(payload.get("region_id") or 0)
            self.summary = str(payload.get("summary") or "")
            self.behavior = str(payload.get("behavior") or "transform")
            self.bbox = tuple(payload.get("bbox") or [])
            self.transition_frames = int(payload.get("transition_frames") or 0)
            self.frame_indices = [
                int(index) for index in payload.get("frame_indices") or []
            ]
            self.unique_cells = int(payload.get("unique_cells") or 0)
            self.change_sum = int(payload.get("change_sum") or 0)
            self.min_changed_region_size = tuple(
                payload.get("min_changed_region_size") or []
            )
            self.max_changed_region_size = tuple(
                payload.get("max_changed_region_size") or []
            )
            self.palette_transitions = [
                tuple(item) for item in payload.get("palette_transitions") or []
            ]
            self.inspection_bbox = tuple(payload.get("inspection_bbox") or [])
            self.inspection_frame_indices = [
                int(index) for index in payload.get("inspection_frame_indices") or []
            ]
            self._inspection_frames = [
                dict(item)
                for item in payload.get("inspection_frames") or []
                if isinstance(item, dict)
            ]

        def inspect(self, rows=None, cols=None, max_tokens=2000, max_frames=12):
            if len(self.inspection_bbox) != 4 or not self._inspection_frames:
                return "No local animation frames are available for this region."
            source_top, source_bottom, source_left, source_right = self.inspection_bbox
            if rows is None:
                top, bottom = self.bbox[0], self.bbox[1]
            else:
                if not isinstance(rows, (list, tuple)) or len(rows) != 2:
                    raise ValueError("rows must be an inclusive (top, bottom) pair")
                top, bottom = int(rows[0]), int(rows[1])
            if cols is None:
                left, right = self.bbox[2], self.bbox[3]
            else:
                if not isinstance(cols, (list, tuple)) or len(cols) != 2:
                    raise ValueError("cols must be an inclusive (left, right) pair")
                left, right = int(cols[0]), int(cols[1])
            if top > bottom or left > right:
                raise ValueError("crop bounds must be ordered and inclusive")
            if (
                top < source_top
                or bottom > source_bottom
                or left < source_left
                or right > source_right
            ):
                raise ValueError(
                    f"requested crop must stay within the available inspection area "
                    f"rows {source_top}-{source_bottom}, cols {source_left}-{source_right}"
                )

            records = _sample_animation_records(self._inspection_frames, max_frames)
            local_top = top - source_top
            local_bottom = bottom - source_top
            local_left = left - source_left
            local_right = right - source_left
            grids = [
                [
                    row[local_left : local_right + 1]
                    for row in (record.get("grid") or [])[local_top : local_bottom + 1]
                ]
                for record in records
            ]
            if not grids or not grids[0] or not grids[0][0]:
                return "The requested local animation crop is empty."

            token_budget = max(128, int(max_tokens))
            visible_candidates = []
            for block in range(1, 9):
                reduced = _reduce_animation_grids(grids, block)
                text, estimate, visible = _render_animation_inspection(
                    self, records, (top, bottom, left, right), reduced, block
                )
                if visible:
                    visible_candidates.append((text, estimate, block))
                    if estimate <= token_budget:
                        return text
            if visible_candidates:
                text, estimate, block = visible_candidates[-1]
                return (
                    text
                    + f"\nWarning: estimated size {estimate} tokens exceeds the requested "
                    f"{token_budget}-token budget even at the hard {block}x{block} scale ceiling; "
                    "request a smaller row/column crop or fewer frames."
                )
            return (
                "No animation transition remains visible inside the requested crop. "
                "Choose a wider portion of the reported region."
            )

        def __str__(self):
            return (
                "AnimationRegionView("
                f"id={self.id}, behavior={self.behavior!r}, bbox={self.bbox}, "
                f"transition_frames={self.transition_frames}, "
                f"inspection_bbox={self.inspection_bbox})"
            )

        __repr__ = __str__


    class AnimationView:
        def __init__(self, payload):
            self.total_frames = int(payload.get("animation_frame_count") or 0)
            self.changed_frames = int(payload.get("animation_changed_frame_count") or 0)
            self.unique_frames = int(payload.get("animation_unique_frame_count") or 0)
            self.outlier = bool(payload.get("animation_outlier"))
            self.temporal_outlier = bool(payload.get("animation_temporal_outlier"))
            self.threshold = payload.get("animation_checkpoint_threshold")
            self.baseline_median = payload.get("animation_baseline_median")
            self.baseline_source = str(payload.get("animation_baseline_source") or "")
            self.action_family = str(payload.get("animation_action_family") or "")
            self.source_level = payload.get("animation_source_level")
            self.hud_border = int(payload.get("animation_hud_border") or 0)
            self.spatial_changed_frames = int(
                payload.get("animation_spatial_changed_frames") or 0
            )
            self.spatial_unique_cells = int(
                payload.get("animation_spatial_unique_cells") or 0
            )
            self.spatial_change_sum = int(
                payload.get("animation_spatial_change_sum") or 0
            )
            self.spatial_peak_changed_cells = int(
                payload.get("animation_spatial_peak_changed_cells") or 0
            )
            self.spatial_gate_passed = bool(
                payload.get("animation_spatial_gate_passed")
            )
            self.large_spatial_gate_passed = bool(
                payload.get("animation_large_spatial_gate_passed")
            )
            self.local_spatial_gate_passed = bool(
                payload.get("animation_local_spatial_gate_passed")
            )
            self.checkpoint_mode = str(
                payload.get("animation_checkpoint_mode") or ""
            )
            self.suppression_reason = str(
                payload.get("animation_checkpoint_suppression_reason") or ""
            )
            self.before_frame = _frame_from_payload(payload.get("before_frame"))
            raw_keyframes = payload.get("keyframes") or []
            self.frames = [
                frame
                for raw in raw_keyframes
                for frame in [_frame_from_payload(raw)]
                if frame is not None
            ]
            self.frame_indices = [
                int(raw.get("index") or 0)
                for raw in raw_keyframes
                if isinstance(raw, dict)
            ]
            self.regions = [
                AnimationRegionView(raw)
                for raw in payload.get("regions") or []
                if isinstance(raw, dict)
            ]

        def region(self, index=0):
            parsed = int(index)
            if parsed < 0 or parsed >= len(self.regions):
                raise ValueError(
                    f"animation region index {parsed} is out of range; "
                    f"available regions: 0-{max(0, len(self.regions) - 1)}"
                )
            return self.regions[parsed]

        def __str__(self):
            return (
                "AnimationView("
                f"total_frames={self.total_frames}, changed_frames={self.changed_frames}, "
                f"sampled_keyframes={len(self.frames)}, regions={len(self.regions)}, "
                f"outlier={self.outlier}, checkpoint_mode={self.checkpoint_mode!r})"
            )

        __repr__ = __str__


    def _frame_from_payload(payload):
        if not isinstance(payload, dict):
            return None
        return FrameView(
            ascii=str(payload.get("ascii", "")),
            step=int(payload.get("step", 0)),
            level=int(payload.get("level", 0)),
            shape=payload.get("shape", [0, 0]),
            grid=payload.get("grid", []),
        )


    def _history_from_payload(payload):
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            items.append(
                HistoryEntryView(
                    action=str(entry.get("action", "")),
                    frame=_frame_from_payload(entry.get("frame")),
                )
            )
        return items


    def _transitions_from_history(history, last_action_result):
        transitions = []
        for index, entry in enumerate(history):
            action = str(getattr(entry, "action", "") or "").strip()
            if not action:
                continue
            before_frame = history[index - 1].frame if index > 0 else None
            transitions.append(
                TransitionView(
                    action=action,
                    before_frame=before_frame,
                    after_frame=entry.frame,
                    result={},
                )
            )
        if transitions and isinstance(last_action_result, dict):
            transitions[-1].result = dict(last_action_result)
        return transitions


    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)


    def _sanitize_exception(exc):
        extracted = traceback.extract_tb(exc.__traceback__)
        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]
        lines = ["Traceback (most recent call last):"]
        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
        lines.append(f"{exc.__class__.__name__}: {exc}")
        return "\n".join(lines)


    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name or "").split(".", 1)[0]
        if root not in SAFE_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in the sandbox.")
        return builtins.__import__(name, globals, locals, fromlist, level)


    def _set_limits(timeout_seconds):
        if resource is None:
            return
        cpu_limit = max(1, int(timeout_seconds)) + 1
        for limit, value in (
            (getattr(resource, "RLIMIT_CPU", None), cpu_limit),
            (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
            (getattr(resource, "RLIMIT_NOFILE", None), 32),
        ):
            if limit is None:
                continue
            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    def _normalize_actions(actions):
        if isinstance(actions, str):
            items = [actions]
        elif isinstance(actions, dict):
            items = [actions]
        elif isinstance(actions, (list, tuple)):
            items = list(actions)
        else:
            raise TypeError(
                "action(actions) expects a string, an action object, or a list of action strings/objects."
            )
        if not items:
            raise ValueError("action(actions) requires at least one action.")

        normalized = []
        for index, item in enumerate(items, start=1):
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError(f"Action {index} is empty.")
                normalized.append({"action": action_name})
                continue
            if isinstance(item, dict):
                action_name = str(item.get("action", "")).strip()
                if not action_name:
                    raise ValueError(f"Action {index} is missing an `action` field.")
                entry = {"action": action_name}
                if action_name.upper() == "MOUSE" and ("x" in item or "y" in item):
                    raise ValueError(
                        f"Action {index} uses legacy MOUSE x/y fields; use row and col."
                    )
                if "row" in item:
                    entry["row"] = item.get("row")
                if "col" in item:
                    entry["col"] = item.get("col")
                normalized.append(entry)
                continue
            raise TypeError(f"Action {index} must be a string or a dict.")
        return normalized


    def main():
        initial = _recv()
        global COLOR_CHARS
        COLOR_CHARS = str(initial.get("color_chars") or "")
        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        sandbox_cwd = str(initial.get("sandbox_cwd", "")).strip()
        if sandbox_cwd:
            os.chdir(sandbox_cwd)
        _set_limits(timeout_seconds)

        action_results = []
        stdout = io.StringIO()
        runtime_globals = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
            "result": None,
        }
        runtime_globals["__builtins__"]["__import__"] = _safe_import

        def _refresh_state(state_payload):
            current_frame = _frame_from_payload(state_payload.get("current_frame"))
            history = _history_from_payload(state_payload.get("history"))
            last_action_result = state_payload.get("last_action_result")
            action_result = (
                dict(last_action_result) if isinstance(last_action_result, dict) else {}
            )
            transitions = _transitions_from_history(history, action_result)
            last_transition = transitions[-1] if transitions else None

            runtime_globals["current_frame"] = current_frame
            runtime_globals["latest_frame"] = current_frame
            runtime_globals["history"] = history
            runtime_globals["transitions"] = transitions
            runtime_globals["last_transition"] = last_transition
            runtime_globals["previous_frame"] = (
                last_transition.before_frame if last_transition is not None else None
            )
            runtime_globals["last_action_frame"] = (
                last_transition.after_frame if last_transition is not None else None
            )
            runtime_globals["last_action"] = last_transition.action if last_transition is not None else None
            runtime_globals["valid_actions"] = [str(item) for item in state_payload.get("valid_actions", [])]
            runtime_globals["last_action_result"] = action_result
            animation_payload = state_payload.get("last_animation")
            runtime_globals["last_animation"] = (
                AnimationView(animation_payload)
                if isinstance(animation_payload, dict)
                else None
            )

        def action(actions):
            normalized_actions = _normalize_actions(actions)
            _send({"type": "action", "actions": normalized_actions})
            reply = _recv()
            if reply.get("type") == "action_error":
                raise RuntimeError(str(reply.get("error", "action failed")))
            if reply.get("type") != "action_result":
                raise RuntimeError("Invalid action response from sandbox host.")
            action_result = reply.get("action_result") or {}
            action_results.append(action_result)
            _refresh_state(reply.get("state") or {})
            return action_result

        runtime_globals["action"] = action
        _refresh_state(initial.get("state") or {})

        try:
            compiled = compile(str(initial.get("code", "")), "<python_tool>", "exec")
            with contextlib.redirect_stdout(stdout):
                exec(compiled, runtime_globals, runtime_globals)
            _send(
                {
                    "type": "final",
                    "stdout": stdout.getvalue(),
                    "result": _json_safe(runtime_globals.get("result")),
                    "action_results": _json_safe(action_results),
                }
            )
        except Exception as exc:
            _send(
                {
                    "type": "error",
                    "error": _sanitize_exception(exc),
                    "stdout": stdout.getvalue(),
                    "action_results": _json_safe(action_results),
                }
            )


    if __name__ == "__main__":
        main()
    """
).replace("__SEGMENTATION_SOURCE__\n", inspect.getsource(_segmentation))


def _sanitize_host_error_text(text: str) -> str:
    if not str(text or "").strip():
        return "Sandbox process exited unexpectedly."
    return "Sandbox process exited unexpectedly."


def _sandbox_env() -> dict[str, str]:
    return {
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PATH": os.environ.get("PATH", ""),
    }


def _send_json_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        try:
            process.kill()
        except OSError:
            pass


def _wait_for_process_exit(process: subprocess.Popen[str], *, timeout: float = 1.0) -> None:
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
    except OSError:
        return

    try:
        process.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        pass


def run_sandboxed_python(
    *,
    code: str,
    timeout_seconds: int,
    initial_state: dict[str, Any],
    action_handler: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rgb_python_tool_") as sandbox_dir:
        host_action_results: list[dict[str, Any]] = []
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _SANDBOX_BOOTSTRAP],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=sandbox_dir,
                env=_sandbox_env(),
                start_new_session=True,
            )
        except OSError:
            return {
                "error": "Sandbox process could not start.",
                "stdout": "",
                "action_results": [],
            }
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        stdout_queue: queue.Queue[str | None] = queue.Queue()

        def _stdout_reader() -> None:
            for raw_line in process.stdout:
                stdout_queue.put(raw_line)
            stdout_queue.put(None)

        threading.Thread(target=_stdout_reader, daemon=True).start()

        _send_json_line(
            process.stdin,
            {
                "code": code,
                "timeout_seconds": timeout_seconds,
                "sandbox_cwd": sandbox_dir,
                "state": initial_state,
                "color_chars": ARC_COLOR_CHARS,
            },
        )

        deadline = time.monotonic() + max(1, int(timeout_seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": f"Tool timed out after {timeout_seconds}s",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                line = stdout_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                stderr = process.stderr.read()
                _wait_for_process_exit(process)
                return {
                    "error": _sanitize_host_error_text(stderr),
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stderr = process.stderr.read()
                _kill_process_group(process)
                _wait_for_process_exit(process)
                return {
                    "error": "Sandbox process returned an invalid response.",
                    "stdout": "",
                    "action_results": list(host_action_results),
                }

            msg_type = str(message.get("type", "")).strip()
            if msg_type == "action":
                try:
                    action_result_payload = action_handler(list(message.get("actions") or []))
                except Exception:  # noqa: BLE001
                    _send_json_line(
                        process.stdin,
                        {
                            "type": "action_error",
                            "error": "action failed in sandbox host.",
                        },
                    )
                    continue
                raw_action_result = action_result_payload.get("action_result") or {}
                if isinstance(raw_action_result, dict):
                    host_action_results.append(dict(raw_action_result))
                _send_json_line(
                    process.stdin,
                    {
                        "type": "action_result",
                        "action_result": raw_action_result,
                        "state": action_result_payload.get("state") or {},
                    },
                )
                continue

            if msg_type in {"final", "error"}:
                _wait_for_process_exit(process)
                return {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": str(message.get("error", "") or ""),
                    "action_results": list(message.get("action_results") or host_action_results),
                }

            _wait_for_process_exit(process)
            return {
                "error": "Sandbox process returned an unknown message type.",
                "stdout": "",
                "action_results": list(host_action_results),
            }
