"""TAAF solver adapter for the existing tool-using harness."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import html
import json
import math
import os
import re
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import arcengine
import numpy as np
import taaf.game
from taaf.solver import Solver

from inference.agent.action_names import (
    to_engine_action,
    to_model_action,
    to_model_actions,
)
from inference.agent.runtime_state import (
    Frame,
    HistoryEntry,
    RUNTIME_STATE_FILENAME,
    write_runtime_state,
)
from inference.agent.tool_agent import ToolAgent
from inference.framework.kaggle import (
    DEFAULT_QWEN_MODEL_DATASET_SOURCE,
    DEFAULT_SERVED_MODEL_NAME,
    DEFAULT_VLLM_MAX_MODEL_LEN,
    DEFAULT_VLLM_PORT,
    DEFAULT_VLLM_TENSOR_PARALLEL_SIZE,
    DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE,
    DEFAULT_WHEELHOUSE_STAMP_TEXT,
    DuckKaggleVllmConfig,
    duck_kaggle_dataset_sources,
    duck_kaggle_setup_command,
    duck_kaggle_teardown_command,
)
from inference.utils.viewer_artifacts import (
    append_raw_events_sidecar,
    reset_raw_events_sidecar,
)

AnalyzerFactory = Callable[[taaf.game.Game, int], Any]

ANALYZER_RETRY_BACKOFF_SECONDS = 1.0
DEFAULT_CANCEL_DRAIN_TIMEOUT_SECONDS = 120.0
_LOCAL_SERVER_PROCESS_ENV_KEYS = (
    "LOCAL_ANALYZER_API_KEY",
    "OPENAI_API_KEY",
    "LOCAL_ANALYZER_BASE_URL",
    "OPENAI_BASE_URL",
    "LOCAL_ANALYZER_PROVIDER",
    "OPENAI_PROVIDER",
)


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _nonnegative_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


@dataclass
class _LocalServerRuntime:
    index: int
    repo_dir: Path
    api_key_file: Path
    env_overrides: dict[str, str]
    base_url: str
    api_key: str = ""


def _analyzer_reported_tokens(analyzer: Any) -> int:
    value = (
        getattr(analyzer, "generated_tokens", None)
        if hasattr(analyzer, "generated_tokens")
        else getattr(analyzer, "total_tokens", 0)
    )
    return max(0, int(value or 0))


def artifact_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _grid_from_state(state: taaf.game.GameState | None) -> tuple[tuple[int, ...], ...]:
    if state is None:
        return ()
    data = state.frame.data
    rows = data.tolist() if hasattr(data, "tolist") else data
    return tuple(tuple(int(cell) for cell in row) for row in rows)


def _solved_frame_grid(
    state: taaf.game.GameState,
) -> tuple[tuple[int, ...], ...] | None:
    """Board at the instant a level was solved, or None when not a transition.

    On a level transition arcengine returns several frames; the last one is
    already the next level's opening board, so ``GameState.frame`` drops the
    winning board. Return the frame just before the transition.
    """
    if os.environ.get("ARC3_WINFRAME", "1") != "1":
        return None
    if not state.just_won_level:
        return None
    if state.raw.state == arcengine.GameState.WIN:
        return None
    frames = state.raw.frame
    if frames is None or len(frames) < 2:
        return None
    data = frames[-2]
    rows = data.tolist() if hasattr(data, "tolist") else data
    return tuple(tuple(int(cell) for cell in row) for row in rows)


def _level_number(game: taaf.game.Game) -> int:
    state = game.current_state
    completed = int(state.levels_completed)
    if state.won:
        return max(1, int(game.number_of_levels))
    return max(1, min(int(game.number_of_levels), completed + 1))


def _engine_action_names(game: taaf.game.Game) -> list[str]:
    names: list[str] = []
    for action_id in game.current_state.available_actions:
        try:
            name = arcengine.GameAction.from_id(int(action_id)).name
        except Exception:
            continue
        if name == "RESET":
            continue
        if name not in names:
            names.append(name)
    return names


def _model_mouse_action_data(
    action_data: dict[str, Any] | None = None,
) -> dict[str, int]:
    data = action_data or {}
    return {"row": int(data.get("y", 0)), "col": int(data.get("x", 0))}


def _format_action_display(
    action_name: str, action_data: dict[str, Any] | None = None
) -> str:
    if action_name == "ACTION6":
        data = _model_mouse_action_data(action_data)
        return f"MOUSE(row={data['row']}, col={data['col']})"
    return to_model_action(action_name)


def _animation_action_family(action_name: str) -> str:
    return "MOUSE" if action_name == "ACTION6" else to_model_action(action_name)


def _sample_evenly(indices: list[int], limit: int) -> list[int]:
    ordered = sorted(set(int(index) for index in indices))
    if limit <= 0 or not ordered:
        return []
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[-1]]
    positions = [
        round(slot * (len(ordered) - 1) / (limit - 1))
        for slot in range(limit)
    ]
    return [ordered[position] for position in sorted(set(positions))]


_ANIMATION_COLOR_CHARS = "WwgGcBMPRbSYOrNp"


def _animation_text_token_estimate(text: str) -> int:
    """Match the tool agent's conservative JSON-character token estimate."""
    rendered = json.dumps(str(text), ensure_ascii=True)
    return max(1, (len(rendered) + 2) // 3)


def _animation_render_grid(grid: np.ndarray) -> str:
    array = np.asarray(grid)
    if array.ndim != 2 or not array.size:
        return "(empty)"
    return "\n".join(
        "".join(
            _ANIMATION_COLOR_CHARS[max(0, min(15, int(value)))]
            for value in row
        )
        for row in array
    )


def _animation_temporal_anchor_reduce(
    frames: list[np.ndarray], block_size: int
) -> list[np.ndarray]:
    """Reduce with one fixed, maximally time-varying pixel per block.

    The chosen source coordinate is shared by every frame, avoiding the
    artificial flicker caused by selecting a different representative in each
    frame. Ties prefer the coordinate nearest the block centre.
    """
    if not frames:
        return []
    arrays = [np.asarray(frame) for frame in frames]
    shape = arrays[0].shape
    if len(shape) != 2 or any(frame.shape != shape for frame in arrays):
        return [frame.copy() for frame in arrays]
    rows, cols = shape
    block = max(1, int(block_size))
    output_rows = math.ceil(rows / block)
    output_cols = math.ceil(cols / block)
    reduced = [
        np.empty((output_rows, output_cols), dtype=arrays[0].dtype)
        for _ in arrays
    ]
    stack = np.stack(arrays, axis=0)
    for out_row, row_start in enumerate(range(0, rows, block)):
        row_end = min(rows, row_start + block)
        for out_col, col_start in enumerate(range(0, cols, block)):
            col_end = min(cols, col_start + block)
            region = stack[:, row_start:row_end, col_start:col_end]
            flattened = region.reshape(len(arrays), -1)
            if len(arrays) > 1:
                transition_counts = np.count_nonzero(
                    flattened[1:] != flattened[:-1], axis=0
                )
            else:
                transition_counts = np.zeros(flattened.shape[1], dtype=int)
            centre_row = (row_end - row_start - 1) / 2.0
            centre_col = (col_end - col_start - 1) / 2.0
            best_index = min(
                range(flattened.shape[1]),
                key=lambda index: (
                    -int(transition_counts[index]),
                    abs(index // (col_end - col_start) - centre_row)
                    + abs(index % (col_end - col_start) - centre_col),
                    index,
                ),
            )
            for frame_index, output in enumerate(reduced):
                output[out_row, out_col] = flattened[frame_index, best_index]
    return reduced


def _animation_visible_transition_count(frames: list[np.ndarray]) -> int:
    """Count transitions that remain visible in a reduced storyboard."""
    return sum(
        not np.array_equal(previous, current)
        for previous, current in zip(frames, frames[1:])
    )


def _animation_novel_frame_order(
    reduced_frames: list[np.ndarray], mandatory: set[int]
) -> list[int]:
    """Greedy farthest-frame order used when the storyboard needs sampling."""
    selected = set(index for index in mandatory if 0 <= index < len(reduced_frames))
    remaining = set(range(len(reduced_frames))) - selected
    min_distances = {
        index: min(
            (
                int(np.count_nonzero(reduced_frames[index] != reduced_frames[other]))
                for other in selected
            ),
            default=0,
        )
        for index in remaining
    }
    min_temporal_gaps = {
        index: min((abs(index - other) for other in selected), default=0)
        for index in remaining
    }
    order: list[int] = []
    while remaining:
        chosen = max(
            remaining,
            key=lambda index: (
                min_distances[index],
                min_temporal_gaps[index],
                -index,
            ),
        )
        remaining.remove(chosen)
        selected.add(chosen)
        order.append(chosen)
        for index in remaining:
            distance = int(
                np.count_nonzero(reduced_frames[index] != reduced_frames[chosen])
            )
            min_distances[index] = min(min_distances[index], distance)
            min_temporal_gaps[index] = min(
                min_temporal_gaps[index], abs(index - chosen)
            )
    return order


def _animation_bbox_iou(
    left: tuple[int, int, int, int] | None,
    right: tuple[int, int, int, int] | None,
) -> float:
    if left is None or right is None:
        return 0.0
    top = max(left[0], right[0])
    bottom = min(left[1], right[1])
    west = max(left[2], right[2])
    east = min(left[3], right[3])
    intersection = max(0, bottom - top + 1) * max(0, east - west + 1)
    left_area = max(0, left[1] - left[0] + 1) * max(0, left[3] - left[2] + 1)
    right_area = max(0, right[1] - right[0] + 1) * max(0, right[3] - right[2] + 1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _animation_set_jaccard(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _animation_tail_is_similar(current: dict[str, Any], prior: dict[str, Any]) -> bool:
    if current.get("action_family") != prior.get("action_family"):
        return False
    current_frames = max(1, int(current.get("changed_frames") or 0))
    prior_frames = max(1, int(prior.get("changed_frames") or 0))
    duration_ratio = min(current_frames, prior_frames) / max(current_frames, prior_frames)
    if duration_ratio < 0.5:
        return False
    cell_overlap = _animation_set_jaccard(
        set(current.get("motion_cells") or set()),
        set(prior.get("motion_cells") or set()),
    )
    bbox_overlap = _animation_bbox_iou(current.get("bbox"), prior.get("bbox"))
    palette_overlap = _animation_set_jaccard(
        set(current.get("palette_transitions") or set()),
        set(prior.get("palette_transitions") or set()),
    )
    return bool(
        (cell_overlap >= 0.35 or bbox_overlap >= 0.60)
        and palette_overlap >= 0.50
    )


def _build_animation_storyboard(
    *,
    previous_frame: np.ndarray,
    returned_frames: list[np.ndarray],
    changed_indices: list[int],
    spatial_masks: list[np.ndarray],
    spatial_change_counts: list[int],
    action_display: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Create a coarse 4x4-to-8x8-block legacy storyboard."""
    previous = np.asarray(previous_frame)
    if previous.ndim != 2 or not returned_frames:
        return {}
    arrays = [np.asarray(frame) for frame in returned_frames]
    if any(frame.shape != previous.shape for frame in arrays):
        return {}

    meaningful_masks = [
        np.asarray(mask, dtype=bool)
        for mask in spatial_masks
        if np.asarray(mask).shape == previous.shape
    ]
    union_mask = np.zeros(previous.shape, dtype=bool)
    for mask in meaningful_masks:
        union_mask |= mask
    coordinates = np.argwhere(union_mask)
    if coordinates.size:
        top = max(0, int(coordinates[:, 0].min()) - 1)
        bottom = min(previous.shape[0] - 1, int(coordinates[:, 0].max()) + 1)
        left = max(0, int(coordinates[:, 1].min()) - 1)
        right = min(previous.shape[1] - 1, int(coordinates[:, 1].max()) + 1)
    else:
        top, bottom, left, right = 0, previous.shape[0] - 1, 0, previous.shape[1] - 1

    selected_raw_indices = [
        index for index in changed_indices if 0 <= index < len(arrays)
    ]
    if not selected_raw_indices:
        selected_raw_indices = [0, len(arrays) - 1]
    records: list[tuple[str, int, np.ndarray]] = [
        ("before", -1, previous[top : bottom + 1, left : right + 1])
    ]
    records.extend(
        (
            f"f{index}",
            index,
            arrays[index][top : bottom + 1, left : right + 1],
        )
        for index in selected_raw_indices
    )
    raw_story_frames = [record[2] for record in records]
    # Animation storyboards are meant to convey broad motion, not preserve
    # pixel-perfect identity. Try the most efficient allowed reduction first,
    # tolerate merged intermediate states, and accept it when at least one
    # transition remains visible. Never send a block smaller than 4x4.
    chosen_block = 0
    reduced_story_frames: list[np.ndarray] = []
    reduced_transition_count = 0
    for block_size in range(8, 3, -1):
        candidate = _animation_temporal_anchor_reduce(raw_story_frames, block_size)
        candidate_transitions = _animation_visible_transition_count(candidate)
        if candidate_transitions < 1:
            continue
        chosen_block = block_size
        reduced_story_frames = candidate
        reduced_transition_count = candidate_transitions
        break
    if not reduced_story_frames:
        return {}

    peak_raw_index = max(
        selected_raw_indices,
        key=lambda index: (
            spatial_change_counts[index]
            if 0 <= index < len(spatial_change_counts)
            else 0,
            -index,
        ),
    )
    peak_story_index = next(
        (
            index
            for index, record in enumerate(records)
            if record[1] == peak_raw_index
        ),
        len(records) - 1,
    )
    mandatory = {0, 1 if len(records) > 1 else 0, peak_story_index, len(records) - 1}

    def render(indices: list[int], *, sampled: bool) -> str:
        ordered = sorted(set(indices))
        out_shape = reduced_story_frames[0].shape
        lines = [
            "AUTOMATIC ANIMATION STORYBOARD",
            f"Action: {action_display}",
            (
                f"Motion crop: rows {top}-{bottom}, cols {left}-{right}. "
                f"Intentionally downsized for a coarse motion overview: "
                f"{out_shape[0]}x{out_shape[1]} from "
                f"{bottom - top + 1}x{right - left + 1}, using one fixed "
                f"maximally time-varying pixel per {chosen_block}x{chosen_block} block "
                "(largest motion-preserving block selected from 4x4 through 8x8). "
                "Fine details and similar intermediate states may be merged; use the "
                "settled board for exact coordinates."
            ),
            f"Visible transitions after spatial reduction: {reduced_transition_count}.",
            (
                f"Frames shown: {len(ordered)} of {len(records)} meaningful states"
                + (" (temporally sampled to fit the token budget)." if sampled else ".")
            ),
            "Legend: W white, w light-gray, g gray, G dark-gray, c charcoal, B black, M magenta, P pink, R red, b blue, S sky-blue, Y yellow, O orange, r dark-red, N light-green, p purple.",
            "Read the panels in time order; original frame numbers are preserved.",
        ]
        for index in ordered:
            lines.extend(
                [
                    f"[{records[index][0]}]",
                    _animation_render_grid(reduced_story_frames[index]),
                ]
            )
        return "\n".join(lines)

    all_indices = list(range(len(records)))
    storyboard = render(all_indices, sampled=False)
    sampled = _animation_text_token_estimate(storyboard) > max_tokens
    shown = all_indices
    if sampled:
        shown = sorted(mandatory)
        storyboard = render(shown, sampled=True)
        for candidate in _animation_novel_frame_order(
            reduced_story_frames, set(shown)
        ):
            proposed = sorted([*shown, candidate])
            proposed_storyboard = render(proposed, sampled=True)
            if _animation_text_token_estimate(proposed_storyboard) > max_tokens:
                continue
            shown = proposed
            storyboard = proposed_storyboard
        while len(shown) > 1 and _animation_text_token_estimate(storyboard) > max_tokens:
            removable = [index for index in shown if index not in {0, len(records) - 1}]
            if not removable:
                removable = [shown[0]]
            shown.remove(removable[len(removable) // 2])
            storyboard = render(shown, sampled=True)

    return {
        "animation_storyboard": storyboard,
        "animation_storyboard_token_estimate": _animation_text_token_estimate(storyboard),
        "animation_storyboard_source_frame_count": len(records),
        "animation_storyboard_frame_count": len(shown),
        "animation_storyboard_sampled": sampled,
        "animation_storyboard_frame_indices": [records[index][1] for index in shown],
        "animation_storyboard_resolution": list(reduced_story_frames[0].shape),
        "animation_storyboard_crop": [top, bottom, left, right],
        "animation_storyboard_block_size": chosen_block,
        "animation_storyboard_reduced_transition_count": reduced_transition_count,
        "animation_storyboard_method": "fixed_temporal_variance_anchor_4_to_8_permissive",
    }


def _animation_component_bbox(
    cells: set[tuple[int, int]],
) -> tuple[int, int, int, int] | None:
    if not cells:
        return None
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _animation_mask_components(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    """Return 8-connected changed-cell components for one transition."""
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2 or not array.size:
        return []
    rows, cols = array.shape
    remaining = {
        (int(row), int(col))
        for row, col in np.argwhere(array)
    }
    components: list[set[tuple[int, int]]] = []
    while remaining:
        seed = remaining.pop()
        component = {seed}
        stack = [seed]
        while stack:
            row, col = stack.pop()
            for next_row in range(max(0, row - 1), min(rows, row + 2)):
                for next_col in range(max(0, col - 1), min(cols, col + 2)):
                    coordinate = (next_row, next_col)
                    if coordinate not in remaining:
                        continue
                    remaining.remove(coordinate)
                    component.add(coordinate)
                    stack.append(coordinate)
        components.append(component)
    return sorted(components, key=lambda cells: (-len(cells), min(cells)))


def _animation_bboxes_link(
    left: tuple[int, int, int, int] | None,
    right: tuple[int, int, int, int] | None,
    *,
    margin: int = 2,
) -> bool:
    if left is None or right is None:
        return False
    return not (
        left[1] + margin < right[0]
        or right[1] + margin < left[0]
        or left[3] + margin < right[2]
        or right[3] + margin < left[2]
    )


def _animation_bbox_center(
    bbox: tuple[int, int, int, int],
) -> tuple[float, float]:
    return (bbox[0] + bbox[1]) / 2.0, (bbox[2] + bbox[3]) / 2.0


def _animation_track_regions(
    *,
    previous_frame: np.ndarray,
    returned_frames: list[np.ndarray],
    masks: list[np.ndarray],
    min_transition_frames: int,
) -> list[dict[str, Any]]:
    """Link changed components across time and drop one-transition blips.

    A normal HUD update is disconnected and changes in only one transition, so
    it never becomes a local animation region. Location is deliberately not
    used as a HUD heuristic: persistent edge motion remains eligible.
    """
    arrays = [np.asarray(frame) for frame in returned_frames]
    prior = np.asarray(previous_frame)
    tracks: list[dict[str, Any]] = []
    for frame_index, (frame, mask) in enumerate(zip(arrays, masks)):
        used_tracks: set[int] = set()
        for cells in _animation_mask_components(mask):
            bbox = _animation_component_bbox(cells)
            if bbox is None:
                continue
            centre = _animation_bbox_center(bbox)
            candidates: list[tuple[tuple[float, ...], int]] = []
            for track_index, track in enumerate(tracks):
                if track_index in used_tracks:
                    continue
                gap = frame_index - int(track["last_frame"])
                if gap < 1 or gap > 2:
                    continue
                prior_bbox = track["last_bbox"]
                if not _animation_bboxes_link(prior_bbox, bbox):
                    continue
                prior_cells = track["last_cells"]
                overlap = len(cells & prior_cells)
                iou = _animation_bbox_iou(prior_bbox, bbox)
                prior_centre = _animation_bbox_center(prior_bbox)
                distance = abs(centre[0] - prior_centre[0]) + abs(
                    centre[1] - prior_centre[1]
                )
                candidates.append(((float(overlap), iou, -distance), track_index))
            if candidates:
                _, track_index = max(candidates)
                track = tracks[track_index]
                used_tracks.add(track_index)
            else:
                track_index = len(tracks)
                track = {
                    "frames": [],
                    "frame_cells": {},
                    "frame_bboxes": {},
                    "cells": set(),
                    "palette_transitions": set(),
                    "change_sum": 0,
                }
                tracks.append(track)
                used_tracks.add(track_index)

            track["frames"].append(frame_index)
            track["frame_cells"][frame_index] = set(cells)
            track["frame_bboxes"][frame_index] = bbox
            track["cells"].update(cells)
            track["change_sum"] += len(cells)
            if prior.shape == frame.shape:
                track["palette_transitions"].update(
                    (int(prior[row, col]), int(frame[row, col]))
                    for row, col in cells
                )
            track["last_frame"] = frame_index
            track["last_bbox"] = bbox
            track["last_cells"] = set(cells)
        prior = frame

    minimum = max(2, int(min_transition_frames))
    persistent = [
        track
        for track in tracks
        if len(set(track.get("frames") or [])) >= minimum
    ]
    for track in persistent:
        track["bbox"] = _animation_component_bbox(set(track["cells"]))
    return sorted(
        persistent,
        key=lambda track: (
            -len(set(track.get("frames") or [])),
            -int(track.get("change_sum") or 0),
            -len(set(track.get("cells") or set())),
            track.get("bbox") or (0, 0, 0, 0),
        ),
    )


def _animation_region_behavior(track: dict[str, Any]) -> str:
    bboxes = list((track.get("frame_bboxes") or {}).values())
    if not bboxes:
        return "transform"
    heights = [bbox[1] - bbox[0] + 1 for bbox in bboxes]
    widths = [bbox[3] - bbox[2] + 1 for bbox in bboxes]
    centres = [_animation_bbox_center(bbox) for bbox in bboxes]
    row_span = max(row for row, _ in centres) - min(row for row, _ in centres)
    col_span = max(col for _, col in centres) - min(col for _, col in centres)
    transitions = set(track.get("palette_transitions") or set())
    reverses = any(
        left != right and (right, left) in transitions
        for left, right in transitions
    )
    stable_location = row_span <= 2.0 and col_span <= 2.0
    # A grow-then-shrink cycle also contains reverse color transitions. Size
    # variation is therefore more specific than blink and must win first.
    if max(heights) - min(heights) >= 2 or max(widths) - min(widths) >= 2:
        return "resize"
    if reverses and stable_location:
        return "blink"
    if row_span >= 2.0 or col_span >= 2.0:
        return "translate"
    return "transform"


def _animation_padded_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    *,
    padding: int = 1,
) -> tuple[int, int, int, int]:
    rows, cols = shape
    return (
        max(0, bbox[0] - padding),
        min(rows - 1, bbox[1] + padding),
        max(0, bbox[2] - padding),
        min(cols - 1, bbox[3] + padding),
    )


def _animation_centered_bounds(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    window_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    rows, cols = shape
    height = min(rows, max(1, int(window_shape[0])))
    width = min(cols, max(1, int(window_shape[1])))
    centre_row, centre_col = _animation_bbox_center(bbox)
    top = int(round(centre_row - (height - 1) / 2.0))
    left = int(round(centre_col - (width - 1) / 2.0))
    top = min(max(0, top), rows - height)
    left = min(max(0, left), cols - width)
    return top, top + height - 1, left, left + width - 1


def _animation_aspect_reduce(
    frames: list[np.ndarray], max_side: int
) -> tuple[list[np.ndarray], int]:
    if not frames:
        return [], 1
    rows, cols = np.asarray(frames[0]).shape
    limit = max(1, int(max_side))
    # Never cross the user's information-preservation floor: v5 may use a
    # smaller uniform block for a local crop, but 8x8 source blocks are the
    # coarsest allowed scale.
    block_size = min(
        8,
        max(1, math.ceil(rows / limit), math.ceil(cols / limit)),
    )
    return _animation_temporal_anchor_reduce(frames, block_size), block_size


def _animation_region_storyboard(
    *,
    region_id: int,
    behavior: str,
    track: dict[str, Any],
    previous_frame: np.ndarray,
    returned_frames: list[np.ndarray],
    mode: str,
    max_side: int,
    max_frames: int,
) -> dict[str, Any]:
    arrays = [np.asarray(frame) for frame in returned_frames]
    if not arrays or mode not in {"fixed", "tracked"}:
        return {}
    frame_indices = sorted(set(int(index) for index in track.get("frames") or []))
    frame_indices = _sample_evenly(frame_indices, max(1, int(max_frames) - 1))
    if not frame_indices:
        return {}
    bboxes = track["frame_bboxes"]
    union_bbox = track.get("bbox")
    if union_bbox is None:
        return {}
    board_shape = tuple(int(value) for value in np.asarray(previous_frame).shape)

    if mode == "fixed":
        fixed_bounds = _animation_padded_bbox(union_bbox, board_shape, padding=1)
        bounds = [fixed_bounds, *[fixed_bounds for _ in frame_indices]]
        explanation = (
            "The source window stays fixed in world coordinates, so translation and "
            "size changes remain spatially honest."
        )
    else:
        component_heights = [bbox[1] - bbox[0] + 1 for bbox in bboxes.values()]
        component_widths = [bbox[3] - bbox[2] + 1 for bbox in bboxes.values()]
        window_shape = (
            min(board_shape[0], max(component_heights) + 2),
            min(board_shape[1], max(component_widths) + 2),
        )
        first_bbox = bboxes[frame_indices[0]]
        bounds = [
            _animation_centered_bounds(first_bbox, board_shape, window_shape),
            *[
                _animation_centered_bounds(bboxes[index], board_shape, window_shape)
                for index in frame_indices
            ],
        ]
        explanation = (
            "The source window follows the changed region, exposing local blinking or "
            "shape changes while absolute motion is described numerically."
        )

    source_frames = [np.asarray(previous_frame), *[arrays[index] for index in frame_indices]]
    crops = [
        frame[top : bottom + 1, left : right + 1]
        for frame, (top, bottom, left, right) in zip(source_frames, bounds)
    ]
    reduced, block_size = _animation_aspect_reduce(crops, max_side)
    if not reduced:
        return {}
    output_shape = tuple(int(value) for value in reduced[0].shape)
    source_shape = tuple(int(value) for value in crops[0].shape)
    labels = ["before", *[f"f{index}" for index in frame_indices]]
    lines = [
        f"LOCAL ANIMATION REGION {region_id}",
        f"Mode: {mode}. Likely behavior: {behavior}.",
        explanation,
        (
            f"Aspect ratio preserved: {source_shape[0]}x{source_shape[1]} source "
            f"to {output_shape[0]}x{output_shape[1]} using one uniform "
            f"{block_size}x{block_size} block scale; no axis was stretched."
        ),
        "Use the settled board for exact final coordinates.",
    ]
    for label, index, bbox, window, grid in zip(
        labels,
        [-1, *frame_indices],
        [bboxes[frame_indices[0]], *[bboxes[item] for item in frame_indices]],
        bounds,
        reduced,
    ):
        height = bbox[1] - bbox[0] + 1
        width = bbox[3] - bbox[2] + 1
        lines.extend(
            [
                (
                    f"[{label}] changed-region rows {bbox[0]}-{bbox[1]}, "
                    f"cols {bbox[2]}-{bbox[3]}, size {height}x{width}; "
                    f"source window rows {window[0]}-{window[1]}, "
                    f"cols {window[2]}-{window[3]}"
                ),
                _animation_render_grid(grid),
            ]
        )
    return {
        "storyboard": "\n".join(lines),
        "resolution": list(output_shape),
        "source_shape": list(source_shape),
        "block_size": block_size,
        "frame_indices": [-1, *frame_indices],
        "visible_transitions": _animation_visible_transition_count(reduced),
    }


def _animation_region_payload(
    *,
    region_id: int,
    track: dict[str, Any],
    previous_frame: np.ndarray,
    returned_frames: list[np.ndarray],
    max_frames: int,
) -> dict[str, Any]:
    bbox = track.get("bbox")
    if bbox is None:
        return {}
    frame_bboxes = list(track["frame_bboxes"].values())
    heights = [item[1] - item[0] + 1 for item in frame_bboxes]
    widths = [item[3] - item[2] + 1 for item in frame_bboxes]
    behaviour = _animation_region_behavior(track)
    transitions = len(set(track.get("frames") or []))
    summary = (
        f"Region {region_id} covers rows {bbox[0]}-{bbox[1]}, cols {bbox[2]}-{bbox[3]}; "
        f"it changed across {transitions} frame transitions and {len(track['cells'])} "
        f"distinct cells. Its changed-region size ranged from "
        f"{min(heights)}x{min(widths)} to {max(heights)}x{max(widths)}. "
        f"The motion most resembles {behaviour}."
    )
    board_shape = tuple(int(value) for value in np.asarray(previous_frame).shape)
    inspection_bbox = _animation_padded_bbox(bbox, board_shape, padding=2)
    top, bottom, left, right = inspection_bbox
    selected_indices = _sample_evenly(
        sorted(set(track.get("frames") or [])),
        max(1, int(max_frames) - 1),
    )
    arrays = [np.asarray(frame) for frame in returned_frames]
    inspection_frames = [
        {
            "index": -1,
            "changed_bbox": list(track["frame_bboxes"][selected_indices[0]]),
            "grid": np.asarray(previous_frame)[
                top : bottom + 1, left : right + 1
            ].tolist(),
        },
        *[
            {
                "index": index,
                "changed_bbox": list(track["frame_bboxes"][index]),
                "grid": arrays[index][top : bottom + 1, left : right + 1].tolist(),
            }
            for index in selected_indices
        ],
    ]
    return {
        "region_id": region_id,
        "summary": summary,
        "behavior": behaviour,
        "bbox": list(bbox),
        "transition_frames": transitions,
        "frame_indices": sorted(set(track.get("frames") or [])),
        "unique_cells": len(track["cells"]),
        "change_sum": int(track.get("change_sum") or 0),
        "min_changed_region_size": [min(heights), min(widths)],
        "max_changed_region_size": [max(heights), max(widths)],
        "palette_transitions": [
            list(item) for item in sorted(track.get("palette_transitions") or set())
        ],
        "inspection_bbox": list(inspection_bbox),
        "inspection_frame_indices": [-1, *selected_indices],
        "inspection_frames": inspection_frames,
        "inspection_max_block_size": 8,
        "inspection_default_max_tokens": 2000,
    }


def _animation_regions_summary(regions: list[dict[str, Any]]) -> str:
    if not regions:
        return ""
    lead = (
        f"Localized animation detected in {len(regions)} persistent region"
        f"{'s' if len(regions) != 1 else ''}."
    )
    return " ".join([lead, *[str(region.get("summary") or "") for region in regions]])


def _is_engine_game_over(game: taaf.game.Game) -> bool:
    return game.current_state.raw.state == arcengine.GameState.GAME_OVER


def _is_run_complete(game: taaf.game.Game) -> bool:
    return game.current_state.raw.state == arcengine.GameState.WIN


def _write_transcript_html(transcript_path: Path, html_path: Path, title: str) -> None:
    if not transcript_path.exists():
        return
    html_path.parent.mkdir(parents=True, exist_ok=True)
    text = transcript_path.read_text(encoding="utf-8")
    body = (
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{background:#1e1e1e;color:#e0e0e0;font-family:-apple-system,system-ui,sans-serif;"
        "padding:20px;max-width:1100px;margin:0 auto;line-height:1.4;}"
        "h1{color:#fff;}pre{white-space:pre-wrap;background:#111;padding:16px;border-radius:6px;"
        "border:1px solid #333;overflow:auto;}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1><pre>{html.escape(text)}</pre>"
        "</body></html>\n"
    )
    html_path.write_text(body, encoding="utf-8")


@dataclass
class _HarnessGameSession:
    solver: "HarnessSolver"
    game: taaf.game.Game
    analyzer: Any
    game_index: int
    pass_index: int
    state_path: Path
    transcript_path: Path
    analysis_html_relpath: str
    stop_event: threading.Event
    viewer_data_path: Path
    started_at: float = field(default_factory=time.monotonic)
    history_entries: list[HistoryEntry] = field(default_factory=list)
    viewer_events: list[dict[str, Any]] = field(default_factory=list)
    analysis_step: int = 0
    animation_checkpoint_enabled: bool = field(
        default_factory=lambda: bool(
            _nonnegative_env_int("ARC3_ANIMATION_CHECKPOINT_ENABLED", 0)
        )
    )
    animation_checkpoint_min_changed: int = field(
        default_factory=lambda: _positive_env_int(
            "ARC3_ANIMATION_CHECKPOINT_MIN_CHANGED", 2
        )
    )
    animation_exposed_keyframes: int = field(
        default_factory=lambda: _positive_env_int(
            "ARC3_ANIMATION_EXPOSED_KEYFRAMES", 12
        )
    )
    animation_baseline_min_samples: int = field(
        default_factory=lambda: _positive_env_int(
            "ARC3_ANIMATION_BASELINE_MIN_SAMPLES", 5
        )
    )
    animation_family_min_samples: int = field(
        default_factory=lambda: _positive_env_int(
            "ARC3_ANIMATION_FAMILY_MIN_SAMPLES", 5
        )
    )
    animation_hud_border: int = field(
        default_factory=lambda: _nonnegative_env_int(
            "ARC3_ANIMATION_HUD_BORDER", 0
        )
    )
    animation_min_spatial_frames: int = field(
        default_factory=lambda: _positive_env_int(
            "ARC3_ANIMATION_MIN_SPATIAL_FRAMES", 4
        )
    )
    animation_min_spatial_unique_cells: int = field(
        default_factory=lambda: _positive_env_int(
            "ARC3_ANIMATION_MIN_SPATIAL_UNIQUE_CELLS", 8
        )
    )
    animation_min_spatial_change_sum: int = field(
        default_factory=lambda: _positive_env_int(
            "ARC3_ANIMATION_MIN_SPATIAL_CHANGE_SUM", 32
        )
    )
    animation_storyboard_max_tokens: int = field(
        default_factory=lambda: _positive_env_int(
            "ARC3_ANIMATION_STORYBOARD_MAX_TOKENS", 2000
        )
    )
    animation_region_min_transition_frames: int = field(
        default_factory=lambda: _positive_env_int(
            "ARC3_ANIMATION_REGION_MIN_TRANSITION_FRAMES", 2
        )
    )
    animation_region_max_count: int = field(
        default_factory=lambda: _positive_env_int(
            "ARC3_ANIMATION_REGION_MAX_COUNT", 4
        )
    )
    animation_region_max_frames: int = field(
        default_factory=lambda: _positive_env_int(
            "ARC3_ANIMATION_REGION_MAX_FRAMES", 12
        )
    )
    animation_checkpoint_max_per_level: int = field(
        default_factory=lambda: _positive_env_int(
            "ARC3_ANIMATION_CHECKPOINT_MAX_PER_LEVEL", 3
        )
    )
    animation_checkpoint_cooldown_actions: int = field(
        default_factory=lambda: _nonnegative_env_int(
            "ARC3_ANIMATION_CHECKPOINT_COOLDOWN_ACTIONS", 5
        )
    )
    animation_game_history: list[int] = field(default_factory=list)
    animation_game_family_history: dict[str, list[int]] = field(default_factory=dict)
    animation_level_history: dict[int, list[int]] = field(default_factory=dict)
    animation_family_history: dict[tuple[int, str], list[int]] = field(
        default_factory=dict
    )
    animation_checkpointed_levels: set[int] = field(default_factory=set)
    animation_tail_signatures: dict[int, list[dict[str, Any]]] = field(
        default_factory=dict
    )
    animation_last_checkpoint_action: dict[int, int] = field(default_factory=dict)
    turn_animation_checkpoint: dict[str, Any] | None = None
    last_engine_action: str | None = None
    token_baseline: int = 0
    _viewer_events_flushed: int = field(default=0, init=False, repr=False)

    def current_frame(self) -> Frame:
        return Frame(
            grid=_grid_from_state(self.game.current_state),
            step=self.action_count,
            level=_level_number(self.game),
        )

    def write_runtime_state(self) -> None:
        write_runtime_state(
            self.state_path,
            current_frame=self.current_frame(),
            history=self.history_entries,
        )

    def seed_initial_history(self) -> None:
        if not self.history_entries:
            self.history_entries.append(
                HistoryEntry(action="", frame=self.current_frame())
            )

    @property
    def action_count(self) -> int:
        run = self.game.game_run
        return len(run.history) if run is not None else 0

    def runtime_limit_reached(self) -> bool:
        if self.solver.max_runtime_s_per_game is None:
            return False
        return (
            time.monotonic() - self.started_at
        ) >= self.solver.max_runtime_s_per_game

    def timing_payload(self) -> dict[str, float | None]:
        elapsed = max(0.0, time.monotonic() - self.started_at)
        if self.solver.max_runtime_s_per_game is None:
            remaining = None
        else:
            remaining = max(0.0, self.solver.max_runtime_s_per_game - elapsed)
        return {"run_elapsed_seconds": elapsed, "time_remaining_seconds": remaining}

    def request_timeout_seconds(self) -> float | None:
        candidates: list[float] = []
        configured = getattr(self.analyzer, "_timeout", None)
        try:
            if configured is not None:
                candidates.append(float(configured))
        except (TypeError, ValueError):
            pass
        if self.solver.max_runtime_s_per_game is not None:
            remaining = self.timing_payload()["time_remaining_seconds"]
            if remaining is not None:
                candidates.append(float(remaining))
        soft_remaining = self.solver.soft_time_remaining_seconds()
        if soft_remaining is not None:
            candidates.append(soft_remaining)
        if not candidates:
            return None
        return max(0.1, min(candidates))

    def should_stop(self) -> bool:
        run = self.game.game_run
        if run is None or run.state != "playing":
            return True
        if self.stop_event.is_set():
            return True
        if _is_run_complete(self.game):
            return True
        if self.runtime_limit_reached():
            return True
        if (
            self.solver.max_actions_per_game is not None
            and self.action_count >= self.solver.max_actions_per_game
        ):
            return True
        return False

    def _animation_threshold(
        self, *, source_level: int, action_family: str
    ) -> tuple[int | None, float, float, str]:
        game_prior = self.animation_game_history
        if len(game_prior) < self.animation_baseline_min_samples:
            return None, 0.0, 0.0, "game_warmup"

        level_family_prior = self.animation_family_history.get(
            (source_level, action_family), []
        )
        level_prior = self.animation_level_history.get(source_level, [])
        game_family_prior = self.animation_game_family_history.get(
            action_family, []
        )
        if len(level_family_prior) >= self.animation_family_min_samples:
            prior = level_family_prior[-32:]
            source = "action_family"
        elif len(level_prior) >= self.animation_baseline_min_samples:
            prior = level_prior[-32:]
            source = "game_level"
        elif len(game_family_prior) >= self.animation_family_min_samples:
            prior = game_family_prior[-32:]
            source = "game_action_family_fallback"
        else:
            prior = game_prior[-32:]
            source = "game_fallback"
        median = float(statistics.median(prior))
        mad = float(statistics.median(abs(value - median) for value in prior))
        robust_delta = max(1, math.ceil(3.0 * 1.4826 * mad))
        threshold = max(
            self.animation_checkpoint_min_changed,
            math.ceil(median * 2.0),
            math.ceil(median + robust_delta),
        )
        return threshold, median, mad, source

    def _animation_payload(
        self,
        *,
        previous_frame_data: Any,
        new_state: taaf.game.GameState,
        source_level: int,
        action_name: str,
        terminal: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        returned_frames = list(new_state.raw.frame)
        total_frames = len(returned_frames)
        changed_indices: list[int] = []
        prior_frame = np.asarray(previous_frame_data)
        unique_frames: set[bytes] = set()
        spatial_changed_frames = 0
        spatial_change_sum = 0
        spatial_peak_changed_cells = 0
        spatial_unique_cells: set[int] = set()
        spatial_motion_cells: set[tuple[int, int]] = set()
        palette_transitions: set[tuple[int, int]] = set()
        spatial_masks: list[np.ndarray] = []
        spatial_change_counts: list[int] = []
        for index, raw_frame in enumerate(returned_frames):
            frame_data = np.asarray(raw_frame)
            unique_frames.add(frame_data.tobytes())
            if prior_frame.shape == frame_data.shape:
                difference = np.not_equal(prior_frame, frame_data)
            else:
                difference = np.ones(frame_data.shape, dtype=bool)
            if bool(np.any(difference)):
                changed_indices.append(index)
            # V5 does not classify HUD by location. The local-region tracker
            # below rejects disconnected components that exist for only one
            # transition, while persistent edge motion remains eligible.
            meaningful_difference = np.asarray(difference, dtype=bool).copy()
            changed_cells = int(np.count_nonzero(meaningful_difference))
            spatial_masks.append(meaningful_difference)
            spatial_change_counts.append(changed_cells)
            if changed_cells:
                spatial_changed_frames += 1
                spatial_change_sum += changed_cells
                spatial_peak_changed_cells = max(
                    spatial_peak_changed_cells, changed_cells
                )
                changed_coordinates = np.argwhere(meaningful_difference)
                for row, col in changed_coordinates:
                    coordinate = (int(row), int(col))
                    spatial_motion_cells.add(coordinate)
                    spatial_unique_cells.add(
                        coordinate[0] * max(1, frame_data.shape[1]) + coordinate[1]
                    )
                if prior_frame.shape == frame_data.shape:
                    palette_transitions.update(
                        (int(prior_frame[row, col]), int(frame_data[row, col]))
                        for row, col in changed_coordinates
                    )
            prior_frame = frame_data

        changed_frames = len(changed_indices)
        action_family = _animation_action_family(action_name)
        threshold, baseline_median, baseline_mad, baseline_source = (
            self._animation_threshold(
                source_level=source_level,
                action_family=action_family,
            )
        )
        temporal_outlier = bool(
            self.animation_checkpoint_enabled
            and not terminal
            and threshold is not None
            and changed_frames >= threshold
        )
        large_spatial_gate_passed = bool(
            spatial_changed_frames >= self.animation_min_spatial_frames
            and len(spatial_unique_cells)
            >= self.animation_min_spatial_unique_cells
            and spatial_change_sum >= self.animation_min_spatial_change_sum
        )
        persistent_tracks = _animation_track_regions(
            previous_frame=np.asarray(previous_frame_data),
            returned_frames=[np.asarray(frame) for frame in returned_frames],
            masks=spatial_masks,
            min_transition_frames=self.animation_region_min_transition_frames,
        )
        selected_tracks = persistent_tracks[: self.animation_region_max_count]
        animation_regions = [
            payload
            for region_id, track in enumerate(selected_tracks)
            for payload in [
                _animation_region_payload(
                    region_id=region_id,
                    track=track,
                    previous_frame=np.asarray(previous_frame_data),
                    returned_frames=[np.asarray(frame) for frame in returned_frames],
                    max_frames=self.animation_region_max_frames,
                )
            ]
            if payload
        ]
        local_spatial_gate_passed = bool(animation_regions)
        spatial_gate_passed = bool(
            large_spatial_gate_passed or local_spatial_gate_passed
        )
        signature_motion_cells = {
            coordinate
            for track in selected_tracks
            for coordinate in set(track.get("cells") or set())
        }
        signature_palette_transitions = {
            transition
            for track in selected_tracks
            for transition in set(track.get("palette_transitions") or set())
        }
        if not signature_motion_cells and large_spatial_gate_passed:
            signature_motion_cells = set(spatial_motion_cells)
            signature_palette_transitions = set(palette_transitions)
        motion_bbox: tuple[int, int, int, int] | None = None
        if signature_motion_cells:
            motion_rows = [row for row, _ in signature_motion_cells]
            motion_cols = [col for _, col in signature_motion_cells]
            motion_bbox = (
                min(motion_rows),
                max(motion_rows),
                min(motion_cols),
                max(motion_cols),
            )
        signature = {
            "action_family": action_family,
            "changed_frames": changed_frames,
            "motion_cells": set(signature_motion_cells),
            "bbox": motion_bbox,
            "palette_transitions": set(signature_palette_transitions),
        }
        prior_signatures = self.animation_tail_signatures.setdefault(source_level, [])
        similar_tail = any(
            _animation_tail_is_similar(signature, prior)
            for prior in prior_signatures
        )
        checkpoint_count = len(prior_signatures)
        checkpoint_already_used = (
            checkpoint_count >= self.animation_checkpoint_max_per_level
        )
        last_checkpoint_action = self.animation_last_checkpoint_action.get(source_level)
        cooldown_remaining = 0
        if last_checkpoint_action is not None:
            cooldown_remaining = max(
                0,
                self.animation_checkpoint_cooldown_actions
                - max(0, self.action_count - last_checkpoint_action),
            )
        outlier = bool(
            temporal_outlier
            and spatial_gate_passed
            and not similar_tail
            and not checkpoint_already_used
            and cooldown_remaining == 0
        )
        if outlier:
            prior_signatures.append(signature)
            self.animation_last_checkpoint_action[source_level] = self.action_count
            if len(prior_signatures) >= self.animation_checkpoint_max_per_level:
                self.animation_checkpointed_levels.add(source_level)

        animation_reminder = bool(
            temporal_outlier and spatial_gate_passed and not outlier and not terminal
        )

        if terminal:
            suppression_reason = "terminal_action"
        elif threshold is None:
            suppression_reason = "game_warmup"
        elif not temporal_outlier:
            suppression_reason = "not_temporal_outlier"
        elif not spatial_gate_passed:
            suppression_reason = "insufficient_non_hud_motion"
        elif similar_tail:
            suppression_reason = "similar_tail_continued"
        elif checkpoint_already_used:
            suppression_reason = "level_checkpoint_quota_used"
        elif cooldown_remaining:
            suppression_reason = "checkpoint_cooldown"
        else:
            suppression_reason = ""

        self.animation_game_history.append(changed_frames)
        self.animation_game_family_history.setdefault(action_family, []).append(
            changed_frames
        )
        self.animation_level_history.setdefault(source_level, []).append(changed_frames)
        self.animation_family_history.setdefault(
            (source_level, action_family), []
        ).append(changed_frames)

        candidate_indices = (
            sorted(set([0, max(0, total_frames - 1), *changed_indices]))
            if total_frames
            else []
        )
        sampled_indices = _sample_evenly(
            candidate_indices, self.animation_exposed_keyframes
        )
        step = self.action_count
        level = _level_number(self.game)

        def frame_payload(data: Any, *, index: int) -> dict[str, Any]:
            return {
                "index": index,
                "step": step,
                "level": level,
                "grid": np.asarray(data).tolist(),
            }

        metadata = {
            "animation_frame_count": total_frames,
            "animation_changed_frame_count": changed_frames,
            "animation_unique_frame_count": len(unique_frames),
            "animation_outlier": outlier,
            "animation_temporal_outlier": temporal_outlier,
            "animation_checkpoint_threshold": threshold,
            "animation_baseline_median": baseline_median,
            "animation_baseline_mad": baseline_mad,
            "animation_baseline_source": baseline_source,
            "animation_action_family": action_family,
            "animation_source_level": source_level,
            "animation_hud_border": 0,
            "animation_spatial_changed_frames": spatial_changed_frames,
            "animation_spatial_unique_cells": len(spatial_unique_cells),
            "animation_spatial_change_sum": spatial_change_sum,
            "animation_spatial_peak_changed_cells": spatial_peak_changed_cells,
            "animation_large_spatial_gate_passed": large_spatial_gate_passed,
            "animation_local_spatial_gate_passed": local_spatial_gate_passed,
            "animation_spatial_gate_passed": spatial_gate_passed,
            "animation_region_count": len(animation_regions),
            "animation_region_behaviors": [
                region.get("behavior") for region in animation_regions
            ],
            "animation_checkpoint_already_used": checkpoint_already_used,
            "animation_checkpoint_count": checkpoint_count + (1 if outlier else 0),
            "animation_checkpoint_max_per_level": self.animation_checkpoint_max_per_level,
            "animation_checkpoint_cooldown_remaining": cooldown_remaining,
            "animation_similar_to_prior_tail": similar_tail,
            "animation_checkpoint_suppression_reason": suppression_reason,
        }
        view = {
            **metadata,
            "before_frame": frame_payload(previous_frame_data, index=-1),
            "keyframes": [
                frame_payload(returned_frames[index], index=index)
                for index in sampled_indices
            ],
            "sampled_frame_indices": sampled_indices,
            "regions": animation_regions,
        }
        if outlier:
            has_broad_storyboard = False
            if large_spatial_gate_passed:
                storyboard = _build_animation_storyboard(
                    previous_frame=np.asarray(previous_frame_data),
                    returned_frames=[np.asarray(frame) for frame in returned_frames],
                    changed_indices=changed_indices,
                    spatial_masks=spatial_masks,
                    spatial_change_counts=spatial_change_counts,
                    action_display=action_name,
                    max_tokens=self.animation_storyboard_max_tokens,
                )
                if storyboard:
                    metadata.update(storyboard)
                    has_broad_storyboard = True
            if animation_regions:
                metadata["animation_region_summary"] = _animation_regions_summary(
                    animation_regions
                )
            if has_broad_storyboard and animation_regions:
                metadata["animation_checkpoint_mode"] = "broad_and_local"
            elif has_broad_storyboard:
                metadata["animation_checkpoint_mode"] = "broad"
            elif animation_regions:
                metadata["animation_checkpoint_mode"] = "local"
            else:
                metadata["animation_checkpoint_mode"] = "description_only"
                metadata["animation_region_summary"] = (
                    "A multi-frame animation was detected, but it could not be reduced "
                    "to a motion-preserving storyboard or a stable local component. "
                    "Re-check the settled board before continuing."
                )
            view["animation_checkpoint_mode"] = metadata[
                "animation_checkpoint_mode"
            ]
        if animation_reminder:
            if suppression_reason == "similar_tail_continued":
                reason = "it closely matched an animation already shown on this level"
            elif suppression_reason == "checkpoint_cooldown":
                reason = f"the animation checkpoint cooldown has {cooldown_remaining} action(s) remaining"
            elif suppression_reason == "level_checkpoint_quota_used":
                reason = "this level has reached its novel-animation checkpoint limit"
            else:
                reason = "the host kept the sequence moving"
            metadata["animation_reminder"] = True
            metadata["animation_reminder_detail"] = (
                f"Animation reminder: {action_family} produced {total_frames} returned "
                f"frames with {changed_frames} actual changes, but the queued action "
                f"sequence continued uninterrupted because {reason}. Re-check the settled "
                "board or inspect `last_animation.regions` if this motion contradicts "
                "the current plan."
            )
        return metadata, view

    def _animation_stop_detail(self, *, remaining: int) -> str:
        checkpoint = self.turn_animation_checkpoint or {}
        action = str(checkpoint.get("action_display") or "The last action")
        total = int(checkpoint.get("animation_frame_count") or 0)
        changed = int(checkpoint.get("animation_changed_frame_count") or 0)
        median = float(checkpoint.get("animation_baseline_median") or 0.0)
        threshold = int(checkpoint.get("animation_checkpoint_threshold") or 0)
        spatial_frames = int(
            checkpoint.get("animation_spatial_changed_frames") or 0
        )
        spatial_unique = int(
            checkpoint.get("animation_spatial_unique_cells") or 0
        )
        spatial_total = int(
            checkpoint.get("animation_spatial_change_sum") or 0
        )
        region_count = int(checkpoint.get("animation_region_count") or 0)
        region_summary = str(
            checkpoint.get("animation_region_summary") or ""
        ).strip()
        checkpoint_mode = str(
            checkpoint.get("animation_checkpoint_mode") or "description_only"
        )
        suffix = (
            f" {remaining} later queued action{'s were' if remaining != 1 else ' was'} not tried."
            if remaining
            else " Any later action() call in this model turn will be left unexecuted."
        )
        if checkpoint_mode == "broad_and_local":
            observation_guidance = (
                "A compact, automatically downsized broad-motion storyboard is included "
                "in the tool result. Persistent local regions are also described. Read "
                "the overview first; if a local detail matters, inspect only its useful "
                "absolute crop with `last_animation.region(0).inspect()` or its `rows=` "
                "and `cols=` arguments. "
            )
        elif checkpoint_mode == "broad":
            observation_guidance = (
                "A compact, automatically downsized broad-motion storyboard is included "
                "in the tool result. Read it for motion direction and use the settled "
                "board for exact coordinates. "
            )
        elif checkpoint_mode == "local":
            observation_guidance = (
                "The tool result describes the persistent local motion without printing "
                "its frames automatically. If it matters, inspect only the useful "
                "absolute crop with `last_animation.region(0).inspect()` or its `rows=` "
                "and `cols=` arguments. "
            )
        else:
            observation_guidance = (
                "No reliable compact animation view was available, so re-check the "
                "settled board before continuing. "
            )
        return (
            f"The host paused this model turn after {action} produced {total} returned "
            f"animation frames, including {changed} actual frame changes. This action family "
            f"usually produced about {median:g} changing frames here; the outlier threshold "
            f"was {threshold}. Motion appeared in {spatial_frames} frames across "
            f"{spatial_unique} distinct cells ({spatial_total} cumulative cell changes), "
            f"forming {region_count} persistent local region"
            f"{'s' if region_count != 1 else ''}.{suffix} The action itself did run. "
            f"{region_summary} {observation_guidance}"
            "The local inspector chooses an aspect-preserving scale only after a request and "
            "never uses source blocks larger than 8x8. This "
            "was an observation checkpoint, not the 8-action batch checkpoint, a blocked "
            "move, or a game rule."
        )

    def play(self) -> None:
        run = self.game.game_run
        assert run is not None, "TAAF starts games before invoking the solver."
        run.solver_analysis_html = self.analysis_html_relpath
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript_path.touch(exist_ok=True)
        self.token_baseline = _analyzer_reported_tokens(self.analyzer)
        self.seed_initial_history()
        self.write_runtime_state()
        self._append_initial_viewer_event()
        self.write_viewer_payload()
        try:
            retry_analysis_step: int | None = None
            while not self.should_stop():
                if (
                    _is_engine_game_over(self.game)
                    and self.last_engine_action != "RESET"
                ):
                    self._execute_auto_reset()
                    continue

                if retry_analysis_step is None:
                    self.analysis_step += 1
                    analysis_step = self.analysis_step
                    self.turn_animation_checkpoint = None
                else:
                    analysis_step = retry_analysis_step

                self.write_runtime_state()
                transcript_before = self._read_transcript_bytes()
                try:
                    result = self.analyzer.analyze(
                        self.state_path,
                        self.action_count,
                        valid_actions=_engine_action_names(self.game),
                        step_env=self.step_env,
                        transcript_path=self.transcript_path,
                        analysis_step=analysis_step,
                        request_timeout_seconds=self.request_timeout_seconds(),
                        should_stop=self.should_stop,
                    )
                finally:
                    transcript_delta = self._transcript_delta_since(transcript_before)
                    if transcript_delta.strip():
                        self._append_analysis_viewer_event(
                            analysis_step, transcript_delta
                        )
                        self.write_viewer_payload()
                if result is None:
                    raise RuntimeError("Analyzer did not return a result.")
                if result.retryable_failure:
                    retry_analysis_step = analysis_step
                    if self.should_stop():
                        break
                    time.sleep(ANALYZER_RETRY_BACKOFF_SECONDS)
                    continue

                retry_analysis_step = None
                if getattr(result, "yielded_control", False):
                    retry_analysis_step = analysis_step
                    continue
                if not result.step_executed:
                    continue
        except Exception as exc:
            if run.final_score is None:
                run.solver_note = f"error: {type(exc).__name__}: {exc}"
                if run.state == "playing":
                    run.state = "crashed"
                self._finish_if_needed()
        finally:
            total_tokens = _analyzer_reported_tokens(self.analyzer)
            if run.solver_note is None:
                run.solver_note = f"tokens={total_tokens}"
            self._finish_if_needed()
            self.state_path.unlink(missing_ok=True)
            self._write_analysis_html()
            self.write_viewer_payload()

    def _finish_if_needed(self) -> None:
        run = self.game.game_run
        if run is not None and run.final_score is None:
            if self.stop_event.is_set() and run.state == "playing":
                run.state = "cancelled"
            self.game.finish_game()

    def _write_analysis_html(self) -> None:
        if self.solver.job_dir is None:
            return
        _write_transcript_html(
            self.transcript_path,
            self.solver.job_dir / self.analysis_html_relpath,
            f"{self.game.game_run.game_id if self.game.game_run else self.game_index} analysis",
        )

    def _read_transcript_bytes(self) -> bytes:
        try:
            return self.transcript_path.read_bytes()
        except OSError:
            return b""

    def _transcript_delta_since(self, previous_transcript: bytes) -> str:
        try:
            current_size = self.transcript_path.stat().st_size
            previous_size = len(previous_transcript)
            with self.transcript_path.open("rb") as file:
                if current_size >= previous_size:
                    current_prefix = file.read(previous_size)
                    if current_prefix == previous_transcript:
                        return file.read().decode("utf-8", errors="replace").strip()
                    file.seek(0)
                return file.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _base_viewer_event(self, frame: Frame) -> dict[str, Any]:
        run = self.game.game_run
        raw_state = self.game.current_state.raw.state
        return {
            "board": [list(row) for row in frame.grid],
            "board_ascii": frame.ascii,
            "score": int(self.game.current_state.levels_completed),
            "state": raw_state.name,
            "level": frame.level,
            "run_status": run.state if run is not None else "playing",
        }

    def _append_initial_viewer_event(self) -> None:
        if self.viewer_events:
            return
        frame = self.current_frame()
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "initial",
                "title": "Initial State",
                "action_num": self.action_count,
                "analysis_step": None,
                "action_display": "RESET",
                "reward": 0.0,
            }
        )

    def _append_analysis_viewer_event(
        self, analysis_step: int, transcript: str
    ) -> None:
        frame = self.current_frame()
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "analysis",
                "title": f"Analysis Step {analysis_step}",
                "action_num": self.action_count,
                "analysis_step": analysis_step,
                "transcript": transcript,
            }
        )

    def _append_action_viewer_event(
        self, payload: dict[str, Any], frame: Frame
    ) -> None:
        self.viewer_events.append(
            {
                **self._base_viewer_event(frame),
                "type": "action",
                "title": f"Action {int(payload.get('action_num') or self.action_count)}",
                "action_num": int(payload.get("action_num") or self.action_count),
                "analysis_step": self.analysis_step,
                "action_name": payload.get("action_name"),
                "action_display": payload.get("action_display"),
                "reward": payload.get("reward"),
                "board_changed": payload.get("board_changed"),
                "done": payload.get("done"),
                "level_completed": payload.get("level_completed"),
                "game_over": payload.get("game_over"),
                "run_complete": payload.get("run_complete"),
                "batch_index": payload.get("batch_index"),
                "batch_size": payload.get("batch_size"),
            }
        )

    def write_viewer_payload(self) -> None:
        if self.solver.job_dir is None:
            return
        self.viewer_data_path.parent.mkdir(parents=True, exist_ok=True)
        run = self.game.game_run
        last_event_source = next(
            (
                event
                for event in reversed(self.viewer_events)
                if event.get("type") == "action"
            ),
            self.viewer_events[-1] if self.viewer_events else {},
        )
        last_event = dict(last_event_source)
        last_event.pop("board", None)
        last_event.pop("board_ascii", None)
        last_event.pop("transcript", None)
        payload = {
            "game_id": run.game_id if run is not None else str(self.game_index),
            "agent_name": self.solver.label,
            "status": run.state if run is not None else "playing",
            "pass_index": self.pass_index,
            "pass_label": str(self.pass_index),
            "eventCount": len(self.viewer_events),
            "lastEvent": last_event,
            "viewer_steps": [],
            "replay_url": self.analysis_html_relpath,
        }
        if run is not None:
            payload.update(
                {
                    "levels_completed": run.levels_completed,
                    "total_levels": run.number_of_levels,
                    "actions_per_level": list(run.actions_per_level),
                    "final_score": run.final_score,
                }
            )
        if self._viewer_events_flushed == 0:
            reset_raw_events_sidecar(self.viewer_data_path)
        append_raw_events_sidecar(
            self.viewer_data_path, self.viewer_events[self._viewer_events_flushed :]
        )
        self._viewer_events_flushed = len(self.viewer_events)
        tmp_path = self.viewer_data_path.with_suffix(
            f"{self.viewer_data_path.suffix}.tmp"
        )
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.viewer_data_path)

    def _normalize_actions(
        self, arguments: dict[str, Any]
    ) -> tuple[list[arcengine.ActionInput] | None, str | None]:
        has_single = bool(str(arguments.get("action", "")).strip())
        has_batch = arguments.get("actions") is not None
        if has_single and has_batch:
            return None, "Use either `action` or `actions`, not both."

        if has_batch:
            raw_actions = arguments.get("actions")
            if not isinstance(raw_actions, list):
                return None, "`actions` must be a JSON array of action objects."
            if not raw_actions:
                return None, "`actions` must contain at least one action."
        else:
            if not has_single:
                return None, "step_env requires `action` or `actions`."
            raw_actions = [
                {
                    "action": arguments.get("action"),
                    "row": arguments.get("row"),
                    "col": arguments.get("col"),
                }
            ]

        actions: list[arcengine.ActionInput] = []
        for index, raw_action in enumerate(raw_actions, start=1):
            if not isinstance(raw_action, dict):
                return None, f"Action {index} must be a JSON object."
            action_name = to_engine_action(raw_action.get("action"))
            if not action_name:
                return (
                    None,
                    f"Unknown action at index {index}: {raw_action.get('action')!r}",
                )
            action_id = arcengine.GameAction.from_name(action_name)
            data: dict[str, Any] = {}
            if action_id == arcengine.GameAction.ACTION6:
                try:
                    row = max(0, min(63, int(raw_action["row"])))
                    column = max(0, min(63, int(raw_action["col"])))
                    data = {
                        "x": column,
                        "y": row,
                    }
                except (KeyError, TypeError, ValueError):
                    return (
                        None,
                        f"MOUSE action at index {index} requires integer row and col arguments.",
                    )
            actions.append(arcengine.ActionInput(id=action_id, data=data))
        return actions, None

    def _error_payload(self, message: str) -> dict[str, Any]:
        return {
            "executed": False,
            "error": message,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            **self.timing_payload(),
        }

    def _terminal_payload(
        self, requested_actions: list[arcengine.ActionInput]
    ) -> dict[str, Any]:
        raw_state = self.game.current_state.raw.state
        is_game_over = raw_state == arcengine.GameState.GAME_OVER
        is_win = raw_state == arcengine.GameState.WIN
        requested = [
            _format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]
        stop_reason = (
            "run_complete" if is_win else "game_over" if is_game_over else "stopped"
        )
        return {
            "executed": False,
            "error": "No action was executed because the current game state is terminal or stopping.",
            "action_num": self.action_count,
            "level": _level_number(self.game),
            "score": int(self.game.current_state.levels_completed),
            "state": raw_state.name,
            "valid_actions": [],
            "board_changed": False,
            "done": is_win,
            "level_completed": False,
            "game_over": is_game_over,
            "run_complete": is_win,
            "batched": len(requested_actions) > 1,
            "requested_count": len(requested_actions),
            "executed_count": 0,
            "requested_actions": requested,
            "executed_actions": [],
            "stopped_early": True,
            "stop_reason": stop_reason,
            **self.timing_payload(),
        }

    def _animation_checkpoint_payload(
        self, requested_actions: list[arcengine.ActionInput]
    ) -> dict[str, Any]:
        requested = [
            _format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]
        checkpoint = dict(self.turn_animation_checkpoint or {})
        return {
            "executed": False,
            "action_num": self.action_count,
            "level": _level_number(self.game),
            "score": int(self.game.current_state.levels_completed),
            "state": self.game.current_state.raw.state.name,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            "board_changed": False,
            "done": False,
            "level_completed": False,
            "game_over": False,
            "run_complete": False,
            "batched": len(requested_actions) > 1,
            "requested_count": len(requested_actions),
            "executed_count": 0,
            "requested_actions": requested,
            "executed_actions": [],
            "unexecuted_actions": requested,
            "stopped_early": True,
            "stop_reason": "animation_outlier",
            "stop_detail": self._animation_stop_detail(
                remaining=len(requested_actions)
            ),
            **{
                key: value
                for key, value in checkpoint.items()
                if key.startswith("animation_")
            },
            **self.timing_payload(),
        }

    def step_env(self, arguments: dict[str, Any]) -> dict[str, Any]:
        requested_actions, error = self._normalize_actions(arguments)
        if error is not None or requested_actions is None:
            return self._error_payload(error or "Could not parse action request.")
        if self.should_stop() or _is_engine_game_over(self.game):
            return self._terminal_payload(requested_actions)
        if self.turn_animation_checkpoint is not None:
            return self._animation_checkpoint_payload(requested_actions)

        executed_payloads: list[dict[str, Any]] = []
        total_reward = 0.0
        stop_reason: str | None = None
        batch_size = len(requested_actions)
        requested_displays = [
            _format_action_display(action.id.name, dict(action.data))
            for action in requested_actions
        ]

        # Isolated cap-8 ablation on the pristine TAAF harness: execute at most
        # eight actions from one action(...) request, then return the settled
        # board and the unexecuted suffix to the same Python REPL.
        batch_checkpoint_limit = 8
        for batch_index, action in enumerate(requested_actions, start=1):
            if batch_index > batch_checkpoint_limit:
                stop_reason = "batch_checkpoint"
                break
            if self.should_stop():
                stop_reason = "stopped"
                break
            if action.id.value not in self.game.current_state.available_actions:
                message = f"{_format_action_display(action.id.name, dict(action.data))} is not valid right now."
                if executed_payloads:
                    stop_reason = "invalid_action"
                    break
                return self._error_payload(message)

            try:
                payload = self._execute_action(
                    action,
                    batch_index=batch_index,
                    batch_size=batch_size,
                    flush_viewer_payload=False,
                )
            except Exception as exc:
                if executed_payloads:
                    stop_reason = "action_error"
                    break
                return self._error_payload(f"{type(exc).__name__}: {exc}")
            executed_payloads.append(payload)
            total_reward += float(payload.get("reward", 0.0) or 0.0)

            if payload.get("run_complete"):
                stop_reason = "run_complete"
                break
            if payload.get("game_over"):
                stop_reason = "game_over"
                break
            if payload.get("level_completed"):
                stop_reason = "level_completed"
                break
            if payload.get("animation_outlier"):
                self.turn_animation_checkpoint = {
                    key: value
                    for key, value in payload.items()
                    if key.startswith("animation_") or key == "action_display"
                }
                stop_reason = "animation_outlier"
                break

        if not executed_payloads:
            return self._error_payload("No action was executed.")

        final_payload = dict(executed_payloads[-1])
        final_payload["reward"] = total_reward
        final_payload["last_reward"] = executed_payloads[-1].get("reward", 0.0)
        final_payload["batched"] = batch_size > 1
        final_payload["requested_count"] = batch_size
        final_payload["executed_count"] = len(executed_payloads)
        final_payload["requested_actions"] = requested_displays
        final_payload["executed_actions"] = [
            str(item.get("action_display") or item.get("action_name") or "")
            for item in executed_payloads
        ]
        final_payload["action_outcomes"] = [
            {
                "action_num": item.get("action_num"),
                "action": item.get("action_display") or item.get("action_name"),
                "board_changed": bool(item.get("board_changed")),
                "reward": item.get("reward", 0.0),
                "level_completed": bool(item.get("level_completed")),
                "game_over": bool(item.get("game_over")),
                "animation_frames": item.get("animation_frame_count"),
                "animation_changed_frames": item.get(
                    "animation_changed_frame_count"
                ),
                "animation_reminder": bool(item.get("animation_reminder")),
                "animation_reminder_detail": item.get("animation_reminder_detail"),
            }
            for item in executed_payloads
        ]
        animation_reminders = [
            {
                "action_num": item.get("action_num"),
                "action": item.get("action_display") or item.get("action_name"),
                "detail": item.get("animation_reminder_detail"),
            }
            for item in executed_payloads
            if item.get("animation_reminder")
        ]
        if animation_reminders:
            final_payload["animation_reminder"] = True
            final_payload["animation_reminders"] = animation_reminders
            final_payload["animation_reminder_detail"] = " ".join(
                str(item.get("detail") or "").strip()
                for item in animation_reminders
                if str(item.get("detail") or "").strip()
            )
        final_payload["board_changed"] = any(
            bool(item.get("board_changed")) for item in executed_payloads
        )
        final_payload["stopped_early"] = len(executed_payloads) < batch_size
        if stop_reason is not None:
            final_payload["stop_reason"] = stop_reason
        if stop_reason == "batch_checkpoint":
            final_payload["batch_checkpoint_limit"] = batch_checkpoint_limit
            final_payload["unexecuted_actions"] = requested_displays[len(executed_payloads):]
        if stop_reason == "animation_outlier":
            final_payload["unexecuted_actions"] = requested_displays[
                len(executed_payloads):
            ]
            final_payload["stop_detail"] = self._animation_stop_detail(
                remaining=len(final_payload["unexecuted_actions"])
            )
        self.write_viewer_payload()
        return final_payload

    def _execute_auto_reset(self) -> None:
        action = arcengine.ActionInput(id=arcengine.GameAction.RESET, data={})
        self._execute_action(action, batch_index=1, batch_size=1, generated_tokens=0)

    def _execute_action(
        self,
        action: arcengine.ActionInput,
        *,
        batch_index: int,
        batch_size: int,
        generated_tokens: int | None = None,
        flush_viewer_payload: bool = True,
    ) -> dict[str, Any]:
        previous_grid = _grid_from_state(self.game.current_state)
        previous_frame_data = self.game.current_state.frame.data.copy()
        previous_completed = int(self.game.current_state.levels_completed)
        source_level = _level_number(self.game)
        if generated_tokens is None:
            current_tokens = _analyzer_reported_tokens(self.analyzer)
            generated_tokens = max(0, current_tokens - self.token_baseline)
            self.token_baseline = current_tokens

        new_state = self.game.execute_action(
            action, generated_tokens=generated_tokens, uncached_input_tokens=0
        )
        self.last_engine_action = action.id.name
        action_display = _format_action_display(action.id.name, dict(action.data))
        current_frame = Frame(
            grid=_grid_from_state(new_state),
            step=self.action_count,
            level=_level_number(self.game),
        )
        # winframe: on a level transition the engine returns
        # [.., solved board, next-level board] and GameState.frame keeps only
        # frame[-1], so the board at the moment of the win never reaches the
        # model. Record it as its own history entry, before the next-level one.
        solved_frame_grid = _solved_frame_grid(new_state)
        if solved_frame_grid is not None:
            self.history_entries.append(
                HistoryEntry(
                    action=f"{action_display} [level {source_level} solved]",
                    frame=Frame(
                        grid=solved_frame_grid,
                        step=self.action_count,
                        level=source_level,
                    ),
                )
            )
        self.history_entries.append(
            HistoryEntry(action=action_display, frame=current_frame)
        )
        self.write_runtime_state()

        completed = int(new_state.levels_completed)
        reward = float(completed - previous_completed) / max(
            1.0, float(self.game.number_of_levels)
        )
        raw_state = new_state.raw.state
        board_changed = previous_grid != _grid_from_state(new_state)
        level_completed = bool(
            new_state.just_won_level and raw_state != arcengine.GameState.WIN
        )
        terminal = bool(
            level_completed
            or raw_state == arcengine.GameState.GAME_OVER
            or raw_state == arcengine.GameState.WIN
            or action.id == arcengine.GameAction.RESET
        )
        animation_metadata, animation_view = self._animation_payload(
            previous_frame_data=previous_frame_data,
            new_state=new_state,
            source_level=source_level,
            action_name=action.id.name,
            terminal=terminal,
        )
        payload = {
            "executed": True,
            "action_num": self.action_count,
            "level": _level_number(self.game),
            "score": completed,
            "reward": reward,
            "state": raw_state.name,
            "valid_actions": to_model_actions(_engine_action_names(self.game)),
            "board_changed": board_changed,
            "done": raw_state == arcengine.GameState.WIN,
            "level_completed": level_completed,
            "game_over": raw_state == arcengine.GameState.GAME_OVER,
            "run_complete": raw_state == arcengine.GameState.WIN,
            "action_name": action.id.name,
            "action_data": (
                _model_mouse_action_data(action.data)
                if action.id == arcengine.GameAction.ACTION6
                else dict(action.data)
            ),
            "action_display": action_display,
            "batch_index": batch_index,
            "batch_size": batch_size,
            **animation_metadata,
            "_animation_view": animation_view,
            **self.timing_payload(),
        }
        self._append_action_viewer_event(payload, current_frame)
        if flush_viewer_payload:
            self.write_viewer_payload()
        return payload


@dataclass
class HarnessSolver(Solver):
    """Run the existing tool-using harness as a TAAF ``Solver``."""

    label: str = "HarnessSolver"
    model: str = ""
    analyzer_timeout: float | None = 120.0
    max_actions_per_game: int | None = None
    max_runtime_s_per_game: float | None = None
    concurrency: int = 16
    save_request_logs: bool = False
    start_local_server: bool = False
    local_server_config: str = ""
    local_server_api_key_file: str = ""
    local_server_repo_dir: str = ""
    local_server_port: int | None = None
    local_server_tensor_parallel_size: int | None = None
    local_server_count: int = 1
    kaggle_enable_vllm: bool = field(default=True, repr=False)
    kaggle_wheelhouse_dataset_source: str = field(
        default=DEFAULT_VLLM_WHEELHOUSE_DATASET_SOURCE, repr=False
    )
    kaggle_model_dataset_source: str = field(
        default=DEFAULT_QWEN_MODEL_DATASET_SOURCE, repr=False
    )
    kaggle_served_model_name: str = field(default=DEFAULT_SERVED_MODEL_NAME, repr=False)
    kaggle_vllm_port: int = field(default=DEFAULT_VLLM_PORT, repr=False)
    kaggle_vllm_max_model_len: int = field(
        default=DEFAULT_VLLM_MAX_MODEL_LEN, repr=False
    )
    kaggle_vllm_tensor_parallel_size: int = field(
        default=DEFAULT_VLLM_TENSOR_PARALLEL_SIZE, repr=False
    )
    kaggle_wheelhouse_stamp_text: str = field(
        default=DEFAULT_WHEELHOUSE_STAMP_TEXT, repr=False
    )
    cancel_drain_timeout_s: float = DEFAULT_CANCEL_DRAIN_TIMEOUT_SECONDS
    analyzer_factory: AnalyzerFactory | None = field(
        default=None, repr=False, compare=False
    )
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _local_server_started: bool = field(
        default=False, init=False, repr=False, compare=False
    )
    _local_server_env_overrides: dict[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _local_server_cwd: str = field(default="", init=False, repr=False, compare=False)
    _local_server_api_key: str = field(
        default="", init=False, repr=False, compare=False
    )
    _local_server_base_url: str = field(
        default="", init=False, repr=False, compare=False
    )
    _local_servers: list[_LocalServerRuntime] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    _local_server_original_env: dict[str, str | None] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    # Custom pool sized to self.concurrency: asyncio.to_thread routes onto
    # Python's default executor, capped at min(32, cpu+4) — which would
    # silently cap real concurrency below self.concurrency.
    _worker_pool: ThreadPoolExecutor | None = field(default=None, init=False, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["analyzer_factory"] = None
        state.pop("_stop_event", None)
        state.pop("_local_server_started", None)
        state.pop("_local_server_env_overrides", None)
        state.pop("_local_server_cwd", None)
        state.pop("_local_server_api_key", None)
        state.pop("_local_server_base_url", None)
        state.pop("_local_servers", None)
        state.pop("_local_server_original_env", None)
        state.pop("_worker_pool", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._stop_event = threading.Event()
        self._local_server_started = False
        self._local_server_env_overrides = {}
        self._local_server_cwd = ""
        self._local_server_api_key = ""
        self._local_server_base_url = ""
        self._local_servers = []
        self._local_server_original_env = {}
        self._worker_pool = None

    def __deepcopy__(self, memo: dict[int, Any]) -> "HarnessSolver":
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_stop_event":
                object.__setattr__(new, key, threading.Event())
            elif key == "analyzer_factory":
                object.__setattr__(new, key, value)
            elif key == "_local_servers":
                object.__setattr__(new, key, [])
            elif key == "_local_server_original_env":
                object.__setattr__(new, key, {})
            elif key == "_worker_pool":
                object.__setattr__(new, key, None)
            else:
                object.__setattr__(new, key, copy.deepcopy(value, memo))
        return new

    @property
    def kaggle_dataset_sources(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return duck_kaggle_dataset_sources(self._kaggle_vllm_config())

    @property
    def kaggle_setup_commands(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return [duck_kaggle_setup_command(self._kaggle_vllm_config())]

    @property
    def kaggle_teardown_commands(self) -> list[str]:
        if not self.kaggle_enable_vllm:
            return []
        return [duck_kaggle_teardown_command()]

    def _kaggle_vllm_config(self) -> DuckKaggleVllmConfig:
        return DuckKaggleVllmConfig(
            wheelhouse_dataset_source=self.kaggle_wheelhouse_dataset_source,
            model_dataset_source=self.kaggle_model_dataset_source,
            served_model_name=self.kaggle_served_model_name,
            vllm_port=self.kaggle_vllm_port,
            max_model_len=self.kaggle_vllm_max_model_len,
            tensor_parallel_size=self.kaggle_vllm_tensor_parallel_size,
            wheelhouse_stamp_text=self.kaggle_wheelhouse_stamp_text,
        )

    def _setup(self) -> None:
        if self.start_local_server:
            self._start_local_servers()
        self._worker_pool = ThreadPoolExecutor(
            max_workers=max(1, int(self.concurrency)),
            thread_name_prefix="harness-game",
        )

    def _teardown(self) -> None:
        if self._local_server_started:
            self._stop_local_servers()
        if self._worker_pool is not None:
            self._worker_pool.shutdown(wait=False)
            self._worker_pool = None

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        self._stop_event.clear()
        semaphore = asyncio.Semaphore(max(1, int(self.concurrency)))
        pass_indices_by_game_id: dict[str, int] = {}
        loop = asyncio.get_running_loop()
        pool = self._worker_pool

        async def run_one(index: int, pass_index: int, game: taaf.game.Game) -> None:
            async with semaphore:
                args = (game, index, pass_index, self._local_server_for_game_index(index))
                if pool is not None:
                    await loop.run_in_executor(pool, functools.partial(self._play_one, *args))
                else:
                    # _setup wasn't called (direct test invocation).
                    await asyncio.to_thread(self._play_one, *args)

        tasks: list[asyncio.Task[None]] = []
        for index, game in enumerate(games):
            game_id = game.game_run.game_id if game.game_run is not None else str(index)
            pass_index = pass_indices_by_game_id.get(game_id, 0)
            pass_indices_by_game_id[game_id] = pass_index + 1
            tasks.append(asyncio.create_task(run_one(index, pass_index, game)))
        try:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tasks), return_exceptions=True
            )
        except asyncio.CancelledError:
            self._stop_event.set()
            await self._drain_game_tasks(tasks)
            self._finish_remaining(games)
            raise

    async def _drain_game_tasks(self, tasks: list[asyncio.Task[None]]) -> None:
        if not tasks:
            return
        timeout = max(0.0, float(self.cancel_drain_timeout_s))
        if timeout == 0.0:
            return
        done, _pending = await asyncio.wait(tasks, timeout=timeout)
        if done:
            await asyncio.gather(*done, return_exceptions=True)

    def _start_local_servers(self) -> None:
        server_count = self._resolved_local_server_count()
        started: list[_LocalServerRuntime] = []
        self._capture_local_server_process_env()
        try:
            for server_index in range(server_count):
                runtime = self._local_server_settings(
                    server_index=server_index, server_count=server_count
                )
                print(
                    "Starting local inference server inside solver setup "
                    f"(server {server_index + 1}/{server_count})"
                )
                subprocess.run(
                    ["make", "server"],
                    cwd=runtime.repo_dir,
                    env=self._local_server_env(runtime.env_overrides),
                    check=True,
                )
                if runtime.api_key_file.is_file():
                    runtime.api_key = runtime.api_key_file.read_text(
                        encoding="utf-8"
                    ).strip()
                started.append(runtime)
        except Exception:
            self._local_servers = started
            self._local_server_started = bool(started)
            with contextlib.suppress(Exception):
                self._stop_local_servers()
            raise

        self._local_servers = started
        self._local_server_started = bool(started)
        if started:
            first = started[0]
            self._local_server_cwd = str(first.repo_dir)
            self._local_server_env_overrides = first.env_overrides
            self._local_server_api_key = first.api_key
            self._local_server_base_url = first.base_url
            if first.api_key:
                os.environ["LOCAL_ANALYZER_API_KEY"] = first.api_key
                os.environ["OPENAI_API_KEY"] = first.api_key
            if first.base_url:
                os.environ["LOCAL_ANALYZER_BASE_URL"] = first.base_url
                os.environ["OPENAI_BASE_URL"] = first.base_url
                os.environ["LOCAL_ANALYZER_PROVIDER"] = "vllm"
                os.environ["OPENAI_PROVIDER"] = "vllm"

    def _stop_local_servers(self) -> None:
        runtimes = list(reversed(self._local_servers))
        if not runtimes and self._local_server_env_overrides:
            repo_dir = (
                Path(self._local_server_cwd)
                if self._local_server_cwd
                else self._local_server_repo_dir()
            )
            runtimes = [
                _LocalServerRuntime(
                    index=0,
                    repo_dir=repo_dir,
                    api_key_file=Path(
                        self._local_server_env_overrides.get("SERVER_API_KEY_FILE", "")
                    ),
                    env_overrides=self._local_server_env_overrides,
                    base_url=self._local_server_base_url,
                    api_key=self._local_server_api_key,
                )
            ]
        try:
            for runtime in runtimes:
                subprocess.run(
                    ["make", "stop-server"],
                    cwd=runtime.repo_dir,
                    env=self._local_server_env(runtime.env_overrides),
                    check=False,
                )
        finally:
            self._local_servers = []
            self._local_server_started = False
            self._restore_local_server_process_env()

    def _capture_local_server_process_env(self) -> None:
        self._local_server_original_env = {
            key: os.environ.get(key) for key in _LOCAL_SERVER_PROCESS_ENV_KEYS
        }

    def _restore_local_server_process_env(self) -> None:
        if not self._local_server_original_env:
            return
        for key, value in self._local_server_original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._local_server_original_env = {}

    def _local_server_settings(
        self, *, server_index: int, server_count: int
    ) -> _LocalServerRuntime:
        config_path = self.local_server_config.strip()
        if not config_path:
            raise ValueError(
                "local_server_config is required when start_local_server is enabled."
            )

        repo_dir = self._local_server_repo_dir()
        run_dir = (self.job_dir or Path.cwd()).resolve()
        api_key_file = self._local_server_api_key_path(
            server_index=server_index,
            server_count=server_count,
            run_dir=run_dir,
        )
        pid_path = run_dir / (
            "server.pid" if server_count <= 1 else f"server-{server_index}.pid"
        )
        log_path = run_dir / (
            "server.log" if server_count <= 1 else f"server-{server_index}.log"
        )
        port = self._local_server_port(server_index=server_index)
        base_url = f"http://127.0.0.1:{port}/v1" if port is not None else ""
        env_overrides = {
            "CONFIG_PATH": config_path,
            "SERVER_API_KEY_FILE": str(api_key_file),
            "SERVER_PID": str(pid_path),
            "SERVER_LOG": str(log_path),
            "SERVER_TAIL_ON_WAIT": "true",
            "UV_PROJECT_ENVIRONMENT": str(repo_dir / ".venv"),
        }
        venv_python = self._local_server_venv_python(repo_dir)
        if venv_python is not None:
            env_overrides["SERVER_VENV_PYTHON"] = str(venv_python)
            env_overrides["PYTHON"] = str(venv_python)
        if port is not None:
            env_overrides.update(
                {
                    "SERVER_PORT": str(port),
                    "LOCAL_ANALYZER_BASE_URL": base_url,
                    "OPENAI_BASE_URL": base_url,
                    "LOCAL_ANALYZER_PROVIDER": "vllm",
                    "OPENAI_PROVIDER": "vllm",
                }
            )
        if self.local_server_tensor_parallel_size is not None:
            env_overrides["SERVER_TENSOR_PARALLEL_SIZE"] = str(
                int(self.local_server_tensor_parallel_size)
            )
        if server_count > 1:
            env_overrides["CUDA_VISIBLE_DEVICES"] = (
                self._cuda_visible_device_for_server(server_index)
            )
        return _LocalServerRuntime(
            index=server_index,
            repo_dir=repo_dir,
            api_key_file=api_key_file,
            env_overrides=env_overrides,
            base_url=base_url,
        )

    def _resolved_local_server_count(self) -> int:
        if not self.start_local_server:
            return 0
        return max(1, int(self.local_server_count or 1))

    def _local_server_port(self, *, server_index: int) -> int | None:
        if self.local_server_port is None:
            return None
        return int(self.local_server_port) + int(server_index)

    def _local_server_api_key_path(
        self, *, server_index: int, server_count: int, run_dir: Path
    ) -> Path:
        default_name = (
            "server-api-key" if server_count <= 1 else f"server-{server_index}-api-key"
        )
        base_path = self._resolve_local_server_path(
            self.local_server_api_key_file, default=run_dir / default_name
        )
        if server_count <= 1 or not str(self.local_server_api_key_file or "").strip():
            return base_path
        suffix = base_path.suffix
        stem = base_path.name[: -len(suffix)] if suffix else base_path.name
        return base_path.with_name(f"{stem}-{server_index}{suffix}")

    def _cuda_visible_device_for_server(self, server_index: int) -> str:
        visible_devices = [
            device.strip()
            for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if device.strip()
        ]
        if server_index < len(visible_devices):
            return visible_devices[server_index]
        return str(server_index)

    def _local_server_repo_dir(self) -> Path:
        repo_dir = (
            Path(self.local_server_repo_dir).expanduser()
            if self.local_server_repo_dir
            else Path(__file__).parents[2]
        )
        repo_dir = repo_dir.resolve()
        if not repo_dir.is_dir():
            raise ValueError(f"local_server_repo_dir does not exist: {repo_dir}")
        return repo_dir

    def _local_server_venv_python(self, repo_dir: Path) -> Path | None:
        repo_venv_python = repo_dir / ".venv" / "bin" / "python"
        if repo_venv_python.is_file():
            return repo_venv_python
        return None

    def _resolve_local_server_path(self, raw_value: str, *, default: Path) -> Path:
        raw = str(raw_value or "").strip()
        if not raw:
            return default.resolve()
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        return (self._local_server_repo_dir() / path).resolve()

    def _local_server_env(
        self, overrides: dict[str, str] | None = None
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(overrides or self._local_server_env_overrides)
        return env

    def soft_time_remaining_seconds(self) -> float | None:
        if self.soft_end_time is None:
            return None
        now = (
            datetime.now(self.soft_end_time.tzinfo)
            if self.soft_end_time.tzinfo
            else datetime.now()
        )
        return max(0.0, (self.soft_end_time - now).total_seconds())

    def _local_server_for_game_index(
        self, game_index: int
    ) -> _LocalServerRuntime | None:
        if not self._local_servers:
            return None
        return self._local_servers[int(game_index) % len(self._local_servers)]

    def _make_analyzer(
        self,
        game: taaf.game.Game,
        index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> Any:
        if self.analyzer_factory is not None:
            return self.analyzer_factory(game, index)
        return ToolAgent(
            model=self.model,
            timeout=self.analyzer_timeout,
            save_request_logs=self.save_request_logs,
            api_key=(
                local_server.api_key
                if local_server is not None
                else self._local_server_api_key
            )
            or None,
            base_url=(
                local_server.base_url
                if local_server is not None
                else self._local_server_base_url
            )
            or None,
            provider="vllm" if local_server is not None else None,
        )

    def _play_one(
        self,
        game: taaf.game.Game,
        index: int,
        pass_index: int,
        local_server: _LocalServerRuntime | None = None,
    ) -> None:
        try:
            assert game.game_run is not None
            run = game.game_run
            run_stem = self._run_stem(run.game_id, pass_index)
            state_path = self._artifacts_dir() / f"{run_stem}_{RUNTIME_STATE_FILENAME}"
            viewer_data_path = self._artifacts_dir() / f"{run_stem}_viewer_data.json"
            transcript_path = self._transcripts_dir() / f"{run_stem}.txt"
            analysis_relpath = f"solver_analysis/{run_stem}.html"
            analyzer = self._make_analyzer(game, index, local_server)
            session = _HarnessGameSession(
                solver=self,
                game=game,
                analyzer=analyzer,
                game_index=index,
                pass_index=pass_index,
                state_path=state_path,
                transcript_path=transcript_path,
                analysis_html_relpath=analysis_relpath,
                stop_event=self._stop_event,
                viewer_data_path=viewer_data_path,
            )
            session.play()
        except Exception as exc:
            self._finish_after_error(game, exc)

    def _artifacts_dir(self) -> Path:
        root = self.job_dir or Path.cwd() / "taaf_harness_artifacts"
        path = root / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _transcripts_dir(self) -> Path:
        root = self.job_dir or Path.cwd() / "taaf_harness_artifacts"
        path = root / "transcripts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_stem(self, game_id: str, index: int) -> str:
        return f"{artifact_stem(game_id)}_p{index}"

    def _finish_remaining(self, games: list[taaf.game.Game]) -> None:
        for game in games:
            run = game.game_run
            if run is not None and run.final_score is None:
                try:
                    if self._stop_event.is_set() and run.state == "playing":
                        run.state = "cancelled"
                    game.finish_game()
                except Exception:
                    pass

    def _finish_after_error(self, game: taaf.game.Game, exc: Exception) -> None:
        run = game.game_run
        if run is None or run.final_score is not None:
            return
        run.solver_note = f"error: {type(exc).__name__}: {exc}"
        if run.state == "playing":
            run.state = "crashed"
        with contextlib.suppress(Exception):
            game.finish_game()
        if run.final_score is None:
            with contextlib.suppress(Exception):
                run.final_score = run._compute_final_score()
