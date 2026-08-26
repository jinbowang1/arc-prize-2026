# ARC-AGI-3 Solver — Qwen3.8-27B-FP8 — 25-Game P1 Public Eval

Public/offline evaluation is overridden to the same 25 public games × 1 pass shape. Competition reruns still use the live private game list from the Kaggle gateway.

#----
import contextlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO
from urllib.request import urlopen


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


NOTEBOOK_START_EPOCH = time.time()
RUN_AS_SUBMISSION = False
RUN_AS_SUBMISSION = RUN_AS_SUBMISSION or _env_bool("KAGGLE_IS_COMPETITION_RERUN", False)
ENABLE_GPU = True

os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if RUN_AS_SUBMISSION else "0"
os.environ.setdefault("MPLBACKEND", "Agg")

if ENABLE_GPU:
    cuda_library_path = "/usr/local/nvidia/lib64"
    existing = [entry for entry in os.environ.get("LIBRARY_PATH", "").split(os.pathsep) if entry]
    os.environ["LIBRARY_PATH"] = os.pathsep.join(
        [cuda_library_path, *[entry for entry in existing if entry != cuda_library_path]]
    )

print(f"TAAF RUN_AS_SUBMISSION={RUN_AS_SUBMISSION}")
if ENABLE_GPU:
    print(f"taaf.kaggle: LIBRARY_PATH={os.environ['LIBRARY_PATH']}")
#----
wheelhouse = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels")
if wheelhouse.exists():
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-warn-conflicts",
            "--disable-pip-version-check",
            "--find-links",
            str(wheelhouse),
            "arc-agi",
        ]
    )
elif os.getenv("TAAF_KAGGLE_BUNDLE_DIR"):
    print(f"Competition wheelhouse not found at {wheelhouse}; assuming local debug dependencies are installed.")
else:
    raise RuntimeError(f"Competition wheelhouse not found at {wheelhouse}.")
#----
# Qwen3.8 / Kaggle input configuration
DATASET_SOURCES: list[str] = [
    "jakobbrggen/taaf-kaggle-source-anim-20260807-anim",
    "driessmit1/arc3-vllm-h100-wheelhouse-v3",
]
KERNEL_SOURCES: list[str] = []

# New private Kaggle Model (Version 1).
QWEN_MODEL_OWNER = "foysalemonshanto"
QWEN_MODEL_SLUG = "qwen3-8-27b-fp8-repacked-v1"
QWEN_MODEL_REF = f"{QWEN_MODEL_OWNER}/{QWEN_MODEL_SLUG}"
QWEN_MODEL_VARIATION = "hf-fp8"
QWEN_MODEL_VERSION = "1"
QWEN_SERVED_MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"
QWEN_MODEL_PATH = Path(
    f"/kaggle/input/models/{QWEN_MODEL_OWNER}/{QWEN_MODEL_SLUG}/"
    f"pytorch/{QWEN_MODEL_VARIATION}/{QWEN_MODEL_VERSION}"
)

DATASET_BUNDLE_MARKER = "taaf-kaggle-bundle.json"
WORKING_DIR = Path(os.getenv("TAAF_KAGGLE_WORKING_DIR", "/kaggle/working")).resolve()
SETUP_ENV_PATH = WORKING_DIR / "taaf_setup_env.json"
SOFT_DEADLINE_BUFFER_S = 600.0
WORKING_DIR.mkdir(parents=True, exist_ok=True)

# Keep the whole run offline. vLLM/Transformers must use the mounted files only.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _split_ref(ref: str) -> tuple[str, str]:
    owner, slug = ref.split("/", 1)
    return owner, slug


def _dataset_mount_candidates(ref: str) -> list[Path]:
    owner, slug = _split_ref(ref)
    return [Path("/kaggle/input") / slug, Path("/kaggle/input/datasets") / owner / slug]


def _kernel_mount_candidates(ref: str) -> list[Path]:
    owner, slug = _split_ref(ref)
    return [Path("/kaggle/usr/lib/notebooks") / owner / slug]


def _first_existing(candidates: list[Path]) -> Path | None:
    return next((candidate for candidate in candidates if candidate.exists()), None)


def _find_taaf_bundle() -> Path:
    explicit = os.getenv("TAAF_KAGGLE_BUNDLE_DIR", "").strip()
    if explicit:
        path = Path(explicit)
        if (path / DATASET_BUNDLE_MARKER).is_file():
            return path

    # Prefer the attached bundle whose marker actually exists.
    for root in [Path("/kaggle/input/datasets"), Path("/kaggle/input"), Path.cwd()]:
        if root.exists():
            for marker in root.rglob(DATASET_BUNDLE_MARKER):
                return marker.parent

    raise RuntimeError("Could not find TAAF Kaggle source bundle dataset.")


def _load_setup_env() -> dict[str, str]:
    if not SETUP_ENV_PATH.is_file():
        return {}
    data = json.loads(SETUP_ENV_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{SETUP_ENV_PATH} must contain a JSON object.")
    return {str(key): str(value) for key, value in data.items()}


def _write_setup_env_updates(updates: dict[str, str]) -> None:
    data = _load_setup_env()
    data.update(updates)
    SETUP_ENV_PATH.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


BUNDLE_DIR = _find_taaf_bundle()
print(f"TAAF source bundle: {BUNDLE_DIR}")

# Verify the Qwen3.8 Kaggle Model before any expensive setup work starts.
if not QWEN_MODEL_PATH.is_dir():
    raise FileNotFoundError(
        "Qwen3.8 Kaggle Model is not attached.\n"
        f"Expected path:\n{QWEN_MODEL_PATH}\n\n"
        "Attach: Qwen3.8 27B FP8 Repacked → PyTorch → hf-fp8 → Version 1"
    )

_required_qwen_files = [
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "outside.safetensors",
    "mtp.safetensors",
    "chat_template.jinja",
]
_missing_qwen_files = [
    name for name in _required_qwen_files if not (QWEN_MODEL_PATH / name).is_file()
]
if _missing_qwen_files:
    raise FileNotFoundError(
        "Qwen3.8 mount is incomplete; missing: " + ", ".join(_missing_qwen_files)
    )

_qwen_layer_shards = sorted(QWEN_MODEL_PATH.glob("model-layers-*.safetensors"))
_qwen_safetensors = sorted(QWEN_MODEL_PATH.glob("*.safetensors"))
if len(_qwen_layer_shards) != 16 or len(_qwen_safetensors) != 18:
    raise RuntimeError(
        "Unexpected Qwen3.8 checkpoint layout: "
        f"{len(_qwen_layer_shards)} layer shards, "
        f"{len(_qwen_safetensors)} safetensors files."
    )

# Tell setup commands and solver code where Kaggle mounted every attached input.
kaggle_input_paths: dict[str, str] = {}
for index, ref in enumerate(DATASET_SOURCES):
    candidates = _dataset_mount_candidates(ref)
    resolved = BUNDLE_DIR if index == 0 else _first_existing(candidates)
    kaggle_input_paths[ref] = str(resolved or candidates[0])

for ref in KERNEL_SOURCES:
    candidates = _kernel_mount_candidates(ref)
    kaggle_input_paths[ref] = str(_first_existing(candidates) or candidates[0])

# The bundled setup resolver asks for owner/slug. Give it a model ref that maps
# directly to the full Kaggle Model version directory.
kaggle_input_paths[QWEN_MODEL_REF] = str(QWEN_MODEL_PATH)

setup_env = {
    "TAAF_KAGGLE_INPUT_PATHS": json.dumps(kaggle_input_paths, sort_keys=True),
    "TAAF_KAGGLE_DATASET_SOURCES": json.dumps(DATASET_SOURCES),
    "TAAF_KAGGLE_KERNEL_SOURCES": json.dumps(KERNEL_SOURCES),
    "TAAF_QWEN_MODEL_REF": QWEN_MODEL_REF,
    "TAAF_QWEN_MODEL_PATH": str(QWEN_MODEL_PATH),
    "TAAF_QWEN_SERVED_MODEL_NAME": QWEN_SERVED_MODEL_NAME,
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
os.environ.update(setup_env)
_write_setup_env_updates(setup_env)

print("\n✅ Qwen3.8 input configuration ready")
print(f"Model ref:       {QWEN_MODEL_REF}")
print(f"Physical path:   {QWEN_MODEL_PATH}")
print(f"Served model:    {QWEN_SERVED_MODEL_NAME}")
print(f"Safetensors:     {len(_qwen_safetensors)}")
print(f"Layer shards:    {len(_qwen_layer_shards)}")
print(f"TAAF input map:  {setup_env['TAAF_KAGGLE_INPUT_PATHS']}")

#----
# Audit the attached inputs that matter for this run.
print("=== TAAF bundle ===")
print(BUNDLE_DIR)
print("Exists:", BUNDLE_DIR.exists())

print("\n=== vLLM wheelhouse ===")
_vllm_wheelhouse = Path(
    "/kaggle/input/datasets/driessmit1/arc3-vllm-h100-wheelhouse-v3"
)
print(_vllm_wheelhouse)
print("Exists:", _vllm_wheelhouse.exists())

print("\n=== Qwen3.8 Kaggle Model ===")
print(QWEN_MODEL_PATH)
print("Exists:", QWEN_MODEL_PATH.exists())
print("Safetensors:", len(list(QWEN_MODEL_PATH.glob("*.safetensors"))))
print(
    "Repacked layer shards:",
    len(list(QWEN_MODEL_PATH.glob("model-layers-*.safetensors"))),
)

#----
import re


def _source_path_entries(bundle_dir: Path) -> list[Path]:
    src_root = bundle_dir / "src"
    if not src_root.is_dir():
        return []

    entries: list[Path] = []
    for repo in sorted(src_root.iterdir(), reverse=True):
        if not repo.is_dir():
            continue
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(candidate)
    return entries


def _command_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["TAAF_KAGGLE_BUNDLE_DIR"] = str(BUNDLE_DIR)
    env["TAAF_KAGGLE_WORKING_DIR"] = str(WORKING_DIR)
    env["TAAF_KAGGLE_SETUP_ENV"] = str(SETUP_ENV_PATH)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env.update(_load_setup_env())
    return env


def _replace_python_assignment(
    command: str,
    variable_name: str,
    value: str,
) -> tuple[str, int]:
    """Replace a top-level Python string assignment inside the setup here-doc."""
    pattern = rf"(?m)^{re.escape(variable_name)}\s*=\s*(['\"])[^\r\n]*?\1\s*$"
    replacement = f"{variable_name} = {value!r}"
    return re.subn(pattern, replacement, command, count=1)


def _patch_qwen38_setup_commands(commands: list[str]) -> list[str]:
    """
    Preserve the TAAF deployment setup but replace its model identity with the
    Qwen3.8 Kaggle Model. This avoids copying/forking the large bundled setup
    script and keeps the wheelhouse/GPU/vLLM behavior from the source bundle.
    """
    patched: list[str] = []
    replacement_counts = {
        "MODEL_OWNER": 0,
        "MODEL_SLUG": 0,
        "SERVED_MODEL_NAME": 0,
    }

    replacements = {
        "MODEL_OWNER": QWEN_MODEL_OWNER,
        "MODEL_SLUG": QWEN_MODEL_SLUG,
        "SERVED_MODEL_NAME": QWEN_SERVED_MODEL_NAME,
    }

    for raw_command in commands:
        command = str(raw_command)

        for variable_name, value in replacements.items():
            command, count = _replace_python_assignment(
                command,
                variable_name,
                value,
            )
            replacement_counts[variable_name] += count

        # Make offline behavior explicit in the child process as well.
        if "def vllm_env()" in command:
            command = command.replace(
                "'VLLM_NO_USAGE_STATS': '1',",
                "'VLLM_NO_USAGE_STATS': '1',\n"
                "            'HF_HUB_OFFLINE': '1',\n"
                "            'TRANSFORMERS_OFFLINE': '1',",
                1,
            )

        patched.append(command)

    missing = [
        name for name, count in replacement_counts.items() if count == 0
    ]
    if missing:
        raise RuntimeError(
            "Could not update the bundled TAAF setup for Qwen3.8. "
            "Missing assignment(s): "
            + ", ".join(missing)
            + ". The attached TAAF bundle's setup_commands.json has changed."
        )

    print("taaf.kaggle: Qwen3.8 setup patch =", replacement_counts, flush=True)
    return patched


def _run_shell_commands(filename: str, *, label: str, check: bool) -> None:
    path = BUNDLE_DIR / filename
    if not path.is_file():
        return

    commands = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(commands, list):
        raise RuntimeError(f"{path} must contain a JSON list of shell commands.")

    if filename == "setup_commands.json":
        commands = _patch_qwen38_setup_commands(commands)

    env = _command_env()
    for command in commands:
        print(f"taaf.kaggle: {label} command: {command}", flush=True)
        result = subprocess.run(
            str(command),
            shell=True,
            check=check,
            cwd=WORKING_DIR,
            env=env,
        )
        if not check and result.returncode != 0:
            print(
                f"taaf.kaggle: {label} command exited with {result.returncode}",
                flush=True,
            )

        # Setup commands may export additional runtime settings.
        env.update(_load_setup_env())
        os.environ.update(env)


# Make bundled TAAF repos importable for this notebook and child Python processes.
source_entries = _source_path_entries(BUNDLE_DIR)
for entry in source_entries:
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

if source_entries:
    import sysconfig

    pth_path = Path(sysconfig.get_paths()["purelib"]) / "taaf_kaggle_sources.pth"
    pth_path.write_text(
        "".join(f"{entry}\n" for entry in source_entries),
        encoding="utf-8",
    )
    print(
        f"taaf.kaggle: wrote {pth_path} ({len(source_entries)} source roots)",
        flush=True,
    )

# Run the TAAF deployment setup, patched to use Qwen3.8.
_run_shell_commands("setup_commands.json", label="setup", check=True)

# Setup commands may export PYTHONPATH through TAAF_KAGGLE_SETUP_ENV.
pythonpath_entries = [
    entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry
]
for entry in reversed(pythonpath_entries):
    if entry not in sys.path:
        sys.path.insert(0, entry)

# Fail early if the analyzer is still exposing an old model identity.
_actual_model_id = os.environ.get("INFERENCE_ANALYZER_MODEL", "")
if _actual_model_id != QWEN_SERVED_MODEL_NAME:
    raise RuntimeError(
        "TAAF setup completed, but the analyzer model ID is wrong: "
        f"{_actual_model_id!r}; expected {QWEN_SERVED_MODEL_NAME!r}"
    )

print("\n✅ TAAF/vLLM setup completed for Qwen3.8")
print("Model path:", QWEN_MODEL_PATH)
print("Analyzer model:", _actual_model_id)
print("Analyzer endpoint:", os.environ.get("LOCAL_ANALYZER_BASE_URL"))

#----
def _soft_end_time(max_runtime_s: float, *, run_as_submission: bool) -> datetime | None:
    if run_as_submission or max_runtime_s <= 0:
        return None
    budget = max(1.0, max_runtime_s)
    buffer = min(SOFT_DEADLINE_BUFFER_S, budget / 2)
    start = datetime.fromtimestamp(NOTEBOOK_START_EPOCH)
    return start + timedelta(seconds=budget - buffer)


def _competition_games():
    import arc_agi

    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=os.environ.get("ARC_BASE_URL", "http://gateway:8001/"),
        environments_dir="",
    )
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=spec.arc_base_url,
        environments_dir="",
    )
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError("Competition Arcade exposed zero environments.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


@contextlib.contextmanager
def _tee_to_file(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _Tee(original_stdout, log_file)
    sys.stderr = _Tee(original_stderr, log_file)
    try:
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        n = 0
        for stream in self._streams:
            n = stream.write(data)
        return n

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)
#----
true_submission = _env_bool("KAGGLE_IS_COMPETITION_RERUN", False)
run_as_submission = _env_bool("TAAF_RUN_AS_SUBMISSION", False) or true_submission
os.environ["ONLY_RESET_LEVELS"] = "true"
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if run_as_submission else "0"
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if run_as_submission else "0"

with open(BUNDLE_DIR / "deploy_target.pkl", "rb") as file:
    target = pickle.load(file)
target.actual_run_as_submission = run_as_submission
target.is_competition_rerun = true_submission
soft_end = _soft_end_time(float(getattr(target, "max_runtime_s", 0.0) or 0.0), run_as_submission=run_as_submission)

with open(BUNDLE_DIR / "benchmark_initial.pkl", "rb") as file:
    bm = pickle.load(file)
bm.job_dir = WORKING_DIR
#----
# Inline customization hook — Q38 P1-style public evaluation.
#
# Q38 P1 runs the full 25 public ARC-AGI-3 games once each (25 games × 1 pass).
# This override applies only to the public/offline notebook run. Competition reruns
# still replace bm.games from Kaggle's live gateway in the final run cell.

print("Benchmark analyzer model:", os.environ.get("INFERENCE_ANALYZER_MODEL"))
print("Qwen3.8 model path:", os.environ.get("TAAF_QWEN_MODEL_PATH"))

Q38_P1_PUBLIC_GAME_IDS = [
    "ar25-0c556536",
    "bp35-0a0ad940",
    "cd82-fb555c5d",
    "cn04-2fe56bfb",
    "dc22-fdcac232",
    "ft09-0d8bbf25",
    "g50t-5849a774",
    "ka59-38d34dbb",
    "lf52-271a04aa",
    "lp85-305b61c3",
    "ls20-9607627b",
    "m0r0-492f87ba",
    "r11l-495a7899",
    "re86-8af5384d",
    "s5i5-18d95033",
    "sb26-7fbdac44",
    "sc25-635fd71a",
    "sk48-d8078629",
    "sp80-589a99af",
    "su15-1944f8ab",
    "tn36-ef4dde99",
    "tr87-cd924810",
    "tu93-0768757b",
    "vc33-5430563c",
    "wa30-ee6fef47",
]

if not true_submission:
    if len(Q38_P1_PUBLIC_GAME_IDS) != 25 or len(set(Q38_P1_PUBLIC_GAME_IDS)) != 25:
        raise RuntimeError("Q38 P1 public game list must contain exactly 25 unique games.")
    if not bm.games:
        raise RuntimeError("benchmark_initial.pkl contains no template public game.")

    import taaf.game_api

    template_game = bm.games[0]
    arcade_spec = getattr(template_game, "arcade_spec", None)
    if arcade_spec is None:
        arcade_spec = getattr(template_game, "_arcade_spec", None)
    if arcade_spec is None:
        raise RuntimeError(
            "Could not recover the public ArcadeSpec from benchmark_initial.pkl; "
            "cannot construct the 25-game Q38 P1 evaluation set."
        )

    bm.games = [
        taaf.game_api.GameAPI(env_name=game_id, arcade_spec=arcade_spec)
        for game_id in Q38_P1_PUBLIC_GAME_IDS
    ]
    bm.n_passes = 1
    bm.game_weights = None

    # These already match Q38 P1 in the source notebook; set them explicitly so the
    # intended evaluation configuration is visible and stable.
    if hasattr(bm.solver, "concurrency"):
        bm.solver.concurrency = 28
    if hasattr(bm.solver, "max_runtime_s_per_game"):
        bm.solver.max_runtime_s_per_game = 7920.0

    bm.label = f"{bm.label}-25g-p1"
    print(f"Public evaluation override: {len(bm.games)} games × {bm.n_passes} pass = {len(bm.games) * bm.n_passes} runs")
    print("Public evaluation concurrency:", getattr(bm.solver, "concurrency", None))
    print("Public per-game runtime cap (s):", getattr(bm.solver, "max_runtime_s_per_game", None))

#----
run_context = contextlib.nullcontext() if run_as_submission else _tee_to_file(WORKING_DIR / "stdout.log")
with run_context:
    preamble = (BUNDLE_DIR / "preamble.txt").read_text(encoding="utf-8")
    print(preamble)
    print(f"deploy.kaggle: working_dir             = {WORKING_DIR}")
    print(f"deploy.kaggle: run_as_submission       = {run_as_submission}")
    print(f"deploy.kaggle: competition_rerun       = {true_submission}")
    print(f"deploy.kaggle: soft_end_time           = {soft_end}")
    print("---")

    bundled_git_status = BUNDLE_DIR / "git_status.txt"
    if bundled_git_status.is_file():
        (WORKING_DIR / "git_status.txt").write_text(
            bundled_git_status.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    if true_submission:
        # Competition reruns use Kaggle's live gateway instead of the bundled offline games.
        os.environ.setdefault("ARC_API_KEY", "test-key-123")
        os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
        os.environ.setdefault("SCHEME", "http")
        os.environ.setdefault("HOST", "gateway")
        os.environ.setdefault("PORT", "8001")
        os.environ.setdefault("OPERATION_MODE", "competition")
        os.environ.setdefault("ENVIRONMENTS_DIR", "")
        os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))

        deadline = time.monotonic() + 600.0
        last_error = ""
        while time.monotonic() < deadline:
            try:
                with urlopen("http://gateway:8001/api/games", timeout=10) as response:
                    if response.status < 500:
                        break
            except Exception as exc:
                last_error = repr(exc)
            time.sleep(5)
        else:
            raise RuntimeError(f"Kaggle gateway did not become ready: {last_error}")

        bm.games = _competition_games()
        bm.n_passes = 1
        bm.game_weights = None

    try:
        await bm.run(
            soft_end_time=soft_end,
            runtime_environment=target,
            minimal_diagnostics=run_as_submission,
        )
        if not true_submission and Path("/kaggle/input").exists():
            try:
                import pandas as pd

                submission = pd.DataFrame(
                    data=[["1_0", "1", True, 1]],
                    columns=["row_id", "game_id", "end_of_game", "score"],
                )
                submission.to_parquet(WORKING_DIR / "submission.parquet", index=False)
            except Exception as exc:
                print(f"taaf.kaggle: could not write offline dummy submission: {exc!r}", flush=True)
    finally:
        _run_shell_commands("teardown_commands.json", label="teardown", check=False)
#----
from html import escape

from IPython.display import HTML, display

diagnostics_html = WORKING_DIR / "diagnostics.html"
if diagnostics_html.is_file():
    # Isolate the full document in an iframe so its styles don't leak into the notebook.
    display(
        HTML(
            f'<iframe srcdoc="{escape(diagnostics_html.read_text(), quote=True)}" '
            'width="100%" height="900" style="border:0"></iframe>'
        )
    )
else:
    print("No diagnostics.html — minimal diagnostics (real submission) suppresses it.")