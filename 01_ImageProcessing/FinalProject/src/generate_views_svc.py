from __future__ import annotations

import argparse
import os
import platform
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SVC_REPO_PATH_FILES = (".svc_repo_path", ".wsl_svc_repo_path")
SVC_VENV_PATH_FILES = (".svc_venv_path", ".wsl_svc_venv_path")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_generated_views(folder: Path) -> None:
    ensure_dir(folder)
    for pattern in ("view_*.png", "source_input.png"):
        for path in folder.glob(pattern):
            try:
                path.unlink()
            except PermissionError:
                print(f"Warning: locked file was not removed: {path}")


def prepare_single_image_input(image_path: Path, work_dir: Path) -> Path:
    ensure_dir(work_dir)
    prepared_path = work_dir / image_path.name
    image = Image.open(image_path).convert("RGB")
    image.save(prepared_path)
    return prepared_path


def is_windows() -> bool:
    return platform.system().lower() == "windows"


def wsl_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    text = str(resolved)
    drive, rest = os.path.splitdrive(text)
    if not drive:
        return text.replace("\\", "/")
    drive_letter = drive.rstrip(":").lower()
    rest = rest.replace("\\", "/").lstrip("/")
    return f"/mnt/{drive_letter}/{rest}"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def read_first_existing_path_file(repo: Path, names: tuple[str, ...]) -> str | None:
    for name in names:
        path_file = repo / name
        if path_file.exists():
            value = path_file.read_text(encoding="utf-8").strip()
            if value:
                return value
    return None


def iter_path_file_values(repo: Path, names: tuple[str, ...]):
    for name in names:
        path_file = repo / name
        if path_file.exists():
            value = path_file.read_text(encoding="utf-8").strip()
            if value:
                yield value


def python_from_venv_dir(venv_dir: str, backend: str) -> str | None:
    python_path = Path(venv_dir).expanduser() / "bin" / "python"
    if python_path.exists():
        return wsl_path(python_path) if backend == "wsl" else str(python_path)
    return None


def resolve_svc_repo(repo: Path, backend: str) -> Path:
    repo = repo.expanduser().resolve()
    recorded_repo = read_first_existing_path_file(repo, SVC_REPO_PATH_FILES)
    if recorded_repo and backend != "wsl":
        recorded_path = Path(recorded_repo).expanduser()
        if (recorded_path / "demo.py").exists():
            return recorded_path.resolve()
    return repo


def find_python(repo: Path, requested: str, backend: str, fallback_repo: Path | None = None) -> str:
    if requested != "auto":
        if backend == "wsl" and os.path.splitdrive(requested)[0]:
            return wsl_path(Path(requested))
        return requested

    for path_repo in (repo, fallback_repo):
        if path_repo is None:
            continue
        for venv_dir in iter_path_file_values(path_repo, SVC_VENV_PATH_FILES):
            python_path = python_from_venv_dir(venv_dir, backend)
            if python_path:
                return python_path
            print(f"Warning: recorded SVC Python does not exist: {Path(venv_dir).expanduser() / 'bin' / 'python'}")

    if backend == "wsl":
        candidates = [
            repo / ".venv-wsl" / "bin" / "python",
            repo / ".venv" / "bin" / "python",
            repo / "venv" / "bin" / "python",
        ]
    else:
        candidates = [
            repo / ".venv" / "Scripts" / "python.exe",
            repo / "venv" / "Scripts" / "python.exe",
            repo / ".venv" / "bin" / "python",
            repo / "venv" / "bin" / "python",
        ]
    for candidate in candidates:
        if candidate.exists():
            return wsl_path(candidate) if backend == "wsl" else str(candidate)
    if backend == "wsl":
        return "python3"
    return sys.executable


def run_command(
    command: list[str],
    cwd: Path | None = None,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=check,
    )


def query_cuda_memory() -> tuple[int, int] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    if not first_line:
        return None
    try:
        free_text, total_text = [part.strip() for part in first_line.split(",", maxsplit=1)]
        return int(free_text), int(total_text)
    except ValueError:
        return None


def check_cuda_memory(min_free_gb: float) -> None:
    if min_free_gb <= 0:
        return
    memory = query_cuda_memory()
    if memory is None:
        return
    free_mib, total_mib = memory
    required_mib = int(min_free_gb * 1024)
    if free_mib < required_mib:
        raise RuntimeError(
            "Not enough free GPU memory for Stable Virtual Camera.\n"
            f"Free VRAM: {free_mib / 1024:.1f} GB / {total_mib / 1024:.1f} GB\n"
            f"Required by preflight: {min_free_gb:.1f} GB\n\n"
            "Run `nvidia-smi` and stop other GPU-heavy processes, then retry.\n"
            "If you intentionally want to try anyway, pass `--min-free-vram-gb 0`."
        )


def run_wsl_command(
    command: list[str],
    repo: Path,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    linux_repo = wsl_path(repo)
    exports = ""
    if env:
        exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items()) + " "
    shell_command = f"cd {shlex.quote(linux_repo)} && {exports}{shell_join(command)}"
    return subprocess.run(
        ["wsl", "bash", "-lc", shell_command],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=check,
    )


def decode_process_output(data: bytes) -> str:
    if not data:
        return ""
    if data.count(b"\x00") > max(1, len(data) // 8):
        return data.decode("utf-16le", errors="replace").replace("\x00", "")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp949", errors="replace")


def check_wsl_available() -> None:
    if shutil.which("wsl") is None:
        raise RuntimeError("wsl.exe was not found. Install WSL first with: wsl --install -d Ubuntu-22.04")
    result = subprocess.run(
        ["wsl", "bash", "-lc", "printf ready"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        wsl_output = decode_process_output(result.stderr or result.stdout).strip()
        raise RuntimeError(
            "WSL is not ready on this computer.\n\n"
            "Install Ubuntu 22.04 from an Administrator PowerShell:\n"
            "  wsl --install -d Ubuntu-22.04\n\n"
            "After rebooting/opening Ubuntu, run:\n"
            "  nvidia-smi\n"
            "  cd /mnt/c/Users/zcz84/Desktop/ImageProcessing\n"
            "  bash scripts/setup_svc_wsl.sh\n\n"
            f"WSL output:\n{wsl_output}"
        )


def check_svc_environment(python_exe: str, repo: Path, backend: str) -> None:
    if backend != "wsl" and not shutil.which(python_exe) and not Path(python_exe).exists():
        raise RuntimeError(
            "Stable Virtual Camera Python executable was not found.\n"
            f"Python used: {python_exe}\n\n"
            "Run the Ubuntu setup again from the project folder:\n"
            "  bash scripts/setup_svc_ubuntu.sh"
        )

    check = (
        "import importlib.util, sys; "
        "print(sys.executable); "
        "missing=[m for m in ['fire','seva','torch'] if importlib.util.find_spec(m) is None]; "
        "print(','.join(missing))"
    )
    if backend == "wsl":
        check_wsl_available()
        result = run_wsl_command([python_exe, "-c", check], repo)
    else:
        result = run_command([python_exe, "-c", check], cwd=repo)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    executable = lines[0] if lines else python_exe
    missing = lines[1].split(",") if len(lines) > 1 and lines[1] else []
    if result.returncode != 0 or missing:
        missing_text = ", ".join(missing) if missing else "unknown import check failure"
        raise RuntimeError(
            "Stable Virtual Camera environment is not ready.\n"
            f"Backend: {backend}\n"
            f"Python used: {executable}\n"
            f"Missing modules: {missing_text}\n\n"
            "Recommended setup from Ubuntu/Linux:\n"
            "  bash scripts/setup_svc_ubuntu.sh\n\n"
            "If you are launching from Windows, run the same setup inside WSL Ubuntu first.\n\n"
            "Native Windows may fail later because the official Stable Virtual Camera repo recommends WSL for Windows users."
        )


def resolve_svc_hf_weight_name(weight_name: str, model_version: float) -> str:
    if model_version <= 1:
        return weight_name
    base, ext = os.path.splitext(weight_name)
    version_suffix = f"v{model_version}"
    if base.endswith(version_suffix):
        return weight_name
    return f"{base}{version_suffix}{ext}"


def check_huggingface_access(
    python_exe: str,
    repo: Path,
    backend: str,
    weight_name: str,
    vae_repo: str,
) -> None:
    check = r"""
from huggingface_hub import hf_hub_download

targets = [
    ("__VAE_REPO__", "vae/diffusion_pytorch_model.safetensors"),
    ("stabilityai/stable-virtual-camera", "__WEIGHT_NAME__"),
]

for repo_id, filename in targets:
    try:
        hf_hub_download(repo_id=repo_id, filename=filename, dry_run=True)
    except Exception as exc:
        raise SystemExit(f"{repo_id}/{filename}: {type(exc).__name__}: {exc}")
print("huggingface-access-ok")
""".replace("__VAE_REPO__", vae_repo).replace("__WEIGHT_NAME__", weight_name)

    if backend == "wsl":
        result = run_wsl_command([python_exe, "-c", check], repo)
    else:
        result = run_command([python_exe, "-c", check], cwd=repo)

    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "Hugging Face access is not ready.\n\n"
            "Log in inside the same SVC environment and accept both gated model terms:\n"
            "  source \"$(cat stable-virtual-camera/.svc_venv_path 2>/dev/null || cat stable-virtual-camera/.wsl_svc_venv_path)/bin/activate\"\n"
            "  hf auth login --force\n"
            "  hf auth whoami\n\n"
            "Open and accept access if prompted:\n"
            "  https://huggingface.co/stabilityai/stable-virtual-camera\n"
            f"  https://huggingface.co/{vae_repo}\n\n"
            f"Underlying error:\n{details}"
        )


def choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    return "wsl" if is_windows() else "native"


def run_stable_virtual_camera(args: argparse.Namespace, prepared_image: Path) -> Path:
    backend = choose_backend(args.backend)
    requested_repo = args.svc_repo.expanduser().resolve()
    repo = resolve_svc_repo(requested_repo, backend)
    if not (repo / "demo.py").exists():
        raise FileNotFoundError(f"Stable Virtual Camera repo not found or incomplete: {repo}")

    python_exe = find_python(repo, args.python, backend, fallback_repo=requested_repo)
    check_svc_environment(python_exe, repo, backend)
    check_huggingface_access(
        python_exe,
        repo,
        backend,
        resolve_svc_hf_weight_name(args.weight_name, args.model_version),
        args.vae_repo,
    )
    if backend == "native":
        check_cuda_memory(args.min_free_vram_gb)

    data_path = wsl_path(prepared_image.parent) if backend == "wsl" else str(prepared_image.parent.resolve())
    svc_env = os.environ.copy()
    svc_env["SEVA_VAE_REPO"] = args.vae_repo
    svc_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command = [
        python_exe,
        "demo.py",
        "--data_path",
        data_path,
        "--task",
        "img2trajvid_s-prob",
        "--replace_or_include_input",
        "True",
        "--traj_prior",
        args.traj_prior,
        "--cfg",
        args.cfg,
        "--guider",
        args.guider,
        "--num_targets",
        str(args.target_count),
        "--L_short",
        str(args.short_side),
        "--use_traj_prior",
        "True",
        "--chunk_strategy",
        args.chunk_strategy,
        "--camera_scale",
        str(args.camera_scale),
        "--version",
        str(args.model_version),
        "--weight_name",
        args.weight_name,
        "--seed",
        str(args.seed),
        "--save_subdir",
        args.save_subdir,
        "--video_save_fps",
        str(args.fps),
    ]

    print(f"Running Stable Virtual Camera via {backend}:")
    print(shell_join(command))
    if backend == "wsl":
        result = run_wsl_command(command, repo, env={"SEVA_VAE_REPO": args.vae_repo})
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            raise subprocess.CalledProcessError(result.returncode, ["wsl", "bash", "-lc", shell_join(command)])
    else:
        subprocess.run(command, cwd=repo, env=svc_env, check=True)

    scene_name = prepared_image.stem
    output_scene = repo / "work_dirs" / "demo" / "img2trajvid_s-prob" / args.save_subdir / scene_name
    if not output_scene.exists():
        raise FileNotFoundError(f"Stable Virtual Camera output folder was not found: {output_scene}")
    return output_scene


def collect_outputs(output_scene: Path, out_dir: Path, include_input: bool) -> None:
    clear_generated_views(out_dir)

    generated = sorted((output_scene / "samples-rgb").glob("*.png"))
    if not generated:
        raise FileNotFoundError(f"No generated PNG files found in {output_scene / 'samples-rgb'}")

    index = 1
    if include_input:
        input_images = sorted((output_scene / "input").glob("*.png"))
        if input_images:
            shutil.copyfile(input_images[0], out_dir / "source_input.png")

    for image_path in generated:
        shutil.copyfile(image_path, out_dir / f"view_{index:03d}.png")
        index += 1

    print(f"Copied {len(generated)} generated views to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate novel views locally with Stable Virtual Camera v1.1.")
    parser.add_argument("--image", type=Path, required=True, help="Input image.")
    parser.add_argument("--out", type=Path, required=True, help="Output folder for generated views.")
    parser.add_argument(
        "--svc-repo",
        type=Path,
        default=Path("stable-virtual-camera"),
        help="Path to the cloned Stability-AI/stable-virtual-camera repository.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "wsl", "native"],
        default="auto",
        help="Execution backend. On Windows, 'auto' uses WSL because SVC recommends WSL.",
    )
    parser.add_argument(
        "--python",
        default="auto",
        help="Python executable inside the SVC environment. 'auto' uses .venv if present.",
    )
    parser.add_argument("--target-count", type=int, default=40, help="Number of target views to generate.")
    parser.add_argument("--short-side", type=int, default=576, help="SVC L_short preprocessing size.")
    parser.add_argument("--traj-prior", default="orbit", help="SVC preset trajectory, e.g. orbit, spiral, pan-left.")
    parser.add_argument("--camera-scale", type=float, default=2.0, help="SVC camera motion scale.")
    parser.add_argument("--cfg", default="4.0,2.0", help="SVC two-pass CFG values.")
    parser.add_argument("--guider", default="1,2", help="SVC guider setting.")
    parser.add_argument("--chunk-strategy", default="interp", help="SVC chunk strategy.")
    parser.add_argument("--model-version", type=float, default=1.1, help="Stable Virtual Camera model version.")
    parser.add_argument(
        "--vae-repo",
        default="Manojb/stable-diffusion-2-1-base",
        help="Diffusers repo containing the SD 2.1-compatible VAE used by SVC.",
    )
    parser.add_argument(
        "--weight-name",
        default="model.safetensors",
        help="Base checkpoint file name. SVC appends v1.1 automatically when --model-version is 1.1.",
    )
    parser.add_argument("--seed", type=int, default=23, help="Random seed.")
    parser.add_argument("--fps", type=float, default=10.0, help="Saved video FPS.")
    parser.add_argument("--save-subdir", default="imageprocessing", help="SVC work_dirs/demo subfolder.")
    parser.add_argument("--include-input", action="store_true", help="Also copy source_input.png.")
    parser.add_argument(
        "--min-free-vram-gb",
        type=float,
        default=12.0,
        help="Minimum free GPU memory required before launching SVC. Use 0 to skip the check.",
    )
    args = parser.parse_args()

    if not args.image.exists():
        parser.error(f"Input image does not exist: {args.image}")
    if args.image.suffix.lower() not in IMAGE_EXTENSIONS:
        parser.error(f"Unsupported image extension: {args.image.suffix}")
    if args.target_count < 1:
        parser.error("--target-count must be at least 1.")
    return args


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)
    temp_input_dir = args.out / "_svc_input"
    prepared_image = prepare_single_image_input(args.image, temp_input_dir)
    output_scene = run_stable_virtual_camera(args, prepared_image)
    collect_outputs(output_scene, args.out, args.include_input)


if __name__ == "__main__":
    main()
