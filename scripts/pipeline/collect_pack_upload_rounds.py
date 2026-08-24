#!/usr/bin/env python3
"""Run seeded collection rounds, archive each round, and upload to ModelScope."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BATCH_ENTRY = PROJECT_ROOT / "scripts/pipeline/run_full_physics_batch.py"
DEFAULT_COMMIT_MESSAGE = "原始随机化方式"
GIB = 1024**3


@dataclass(frozen=True)
class BatchStats:
    """Completion and task-success counts read from batch_summary.jsonl."""

    attempted: int
    succeeded: int
    failed: int


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _non_negative_int(raw_value: str) -> int:
    value = int(raw_value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _positive_float(raw_value: str) -> float:
    value = float(raw_value)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return value


def _non_negative_float(raw_value: str) -> float:
    value = float(raw_value)
    if value < 0.0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return value


def _success_rate_threshold(raw_value: str) -> float:
    value = float(raw_value)
    if not 0.0 <= value < 100.0:
        raise argparse.ArgumentTypeError("must be in the range [0, 100)")
    return value


def _space_safety_factor(raw_value: str) -> float:
    value = float(raw_value)
    if value < 1.0:
        raise argparse.ArgumentTypeError("must be at least 1.0")
    return value


def _default_output_root() -> Path:
    configured = os.environ.get("PCT_SCENE_OUTPUT")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "pct_scene_outputs"


def normalize_dataset_repo_id(raw_value: str) -> str:
    """Accept owner/name or a ModelScope dataset URL and return owner/name."""

    value = raw_value.strip().rstrip("/")
    if not value:
        raise ValueError("dataset URL/repository ID cannot be empty")

    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"unsupported dataset URL: {raw_value!r}")
        parts = [part for part in parsed.path.split("/") if part]
        if parts and parts[0] in {"datasets", "dataset"}:
            parts = parts[1:]
    else:
        parts = [part for part in value.split("/") if part]
        if parts and parts[0] in {"datasets", "dataset"}:
            parts = parts[1:]

    if len(parts) != 2:
        raise ValueError(
            "dataset must be owner/name or a URL like "
            "https://modelscope.cn/datasets/owner/name"
        )
    return "/".join(parts)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "按连续 seed 分轮运行 full-physics 批采；每轮完成后打包为 .tar，"
            "再上传到指定 ModelScope dataset。"
        ),
    )
    parser.add_argument(
        "--dataset-url",
        required=True,
        help=(
            "远端 ModelScope dataset，支持 owner/name 或完整 URL，例如 "
            "NaturalHorse/liangzhu_sim_primitive。"
        ),
    )
    parser.add_argument(
        "--episodes-per-round",
        type=_positive_int,
        default=400,
        help="每轮采集的 episode 数；默认 400。",
    )
    parser.add_argument(
        "--rounds",
        type=_positive_int,
        default=1,
        help="采集轮数；默认 1。",
    )
    parser.add_argument(
        "--min-success-rate",
        type=_success_rate_threshold,
        default=80.0,
        help="允许打包上传的最低成功率（百分数，不含等号）；默认要求严格大于 80。",
    )
    parser.add_argument(
        "--start-seed",
        type=_non_negative_int,
        default=601,
        help="第一轮第一条轨迹的 seed；默认 601。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_output_root(),
        help="输出根目录；默认使用 PCT_SCENE_OUTPUT，否则为 ~/pct_scene_outputs。",
    )
    space_source_group = parser.add_mutually_exclusive_group()
    space_source_group.add_argument(
        "--space-reference",
        type=Path,
        default=None,
        help=(
            "用于估算单条数据大小的既有完整批次目录；默认从 output-root 中的"
            "无视频完整批次里选择单条占用最大的参考。"
        ),
    )
    space_source_group.add_argument(
        "--estimated-gib-per-episode",
        type=_positive_float,
        default=None,
        help="直接指定每条 episode 的预计原始占用（GiB），覆盖历史批次自动估算。",
    )
    parser.add_argument(
        "--space-safety-factor",
        type=_space_safety_factor,
        default=1.1,
        help="原始目录和 tar 空间估算的安全系数；默认 1.1。",
    )
    parser.add_argument(
        "--free-space-reserve-gib",
        type=_non_negative_float,
        default=5.0,
        help="完成原始目录和 tar 后必须额外保留的磁盘空间；默认 5 GiB。",
    )
    parser.add_argument(
        "--name-prefix",
        default="liangzhu",
        help=(
            "每轮目录和 tar 文件名前缀；默认 liangzhu。最终名称格式为 "
            "<prefix>_seed<seed>_n<count>。"
        ),
    )
    parser.add_argument(
        "--scene-profile",
        default="liangzhu",
        help="传给批采入口的 scene profile；默认 liangzhu。",
    )
    parser.add_argument(
        "--navigation-visual-mode",
        choices=("auto", "collision", "full"),
        default="full",
        help="传给批采入口的视觉模式；默认 full。",
    )
    parser.add_argument(
        "--isaac-python",
        default=os.environ.get("ISAAC_PYTHON", ""),
        help="Isaac Sim Python 路径；默认读取 ISAAC_PYTHON。",
    )
    parser.add_argument(
        "--ms-bin",
        default="ms",
        help="ModelScope CLI 命令或绝对路径；默认 ms。",
    )
    parser.add_argument(
        "--commit-message",
        default=DEFAULT_COMMIT_MESSAGE,
        help=f"ModelScope commit message；默认 {DEFAULT_COMMIT_MESSAGE!r}。",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="上传目标分支；不传时使用 ms CLI 默认分支。",
    )
    parser.add_argument(
        "--remote-prefix",
        default="",
        help="tar 在 dataset 仓库内的可选目录前缀，例如 raw/v1。",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "允许复用已经完整采集的轮次目录和有效 tar；不完整的已有目录仍会拒绝，"
            "避免混入旧数据。"
        ),
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="上传成功后仍保留原始轮次目录和 tar；默认会删除两者以释放磁盘空间。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印每轮采集、打包和上传命令，不执行。",
    )
    return parser


def _display_command(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def _run(command: Sequence[str], *, env: dict[str, str] | None = None) -> int:
    print(f"\n$ {_display_command(command)}", flush=True)
    return subprocess.run([str(part) for part in command], env=env, check=False).returncode


def _read_batch_stats(summary_path: Path, expected_count: int) -> BatchStats:
    if not summary_path.is_file():
        raise RuntimeError(f"batch summary does not exist: {summary_path}")

    by_index: dict[int, dict[str, object]] = {}
    with summary_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                episode_index = int(record["episode_index"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"invalid batch summary row {summary_path}:{line_number}: {error}"
                ) from error
            by_index[episode_index] = record

    expected_indices = set(range(expected_count))
    actual_indices = set(by_index)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        unexpected = sorted(actual_indices - expected_indices)
        details: list[str] = []
        if missing:
            preview = ", ".join(str(value) for value in missing[:10])
            details.append(f"missing indices: {preview}{' ...' if len(missing) > 10 else ''}")
        if unexpected:
            preview = ", ".join(str(value) for value in unexpected[:10])
            details.append(
                f"unexpected indices: {preview}{' ...' if len(unexpected) > 10 else ''}"
            )
        raise RuntimeError(
            f"batch is incomplete ({len(actual_indices)}/{expected_count}); "
            + "; ".join(details)
        )

    succeeded = sum(bool(record.get("success")) for record in by_index.values())
    return BatchStats(
        attempted=len(by_index),
        succeeded=succeeded,
        failed=len(by_index) - succeeded,
    )


def _infer_batch_stats(summary_path: Path) -> BatchStats:
    indices: set[int] = set()
    with summary_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            indices.add(int(record["episode_index"]))
    if not indices:
        raise RuntimeError(f"batch summary is empty: {summary_path}")
    expected_count = max(indices) + 1
    return _read_batch_stats(summary_path, expected_count)


def _directory_size_bytes(path: Path) -> int:
    result = subprocess.run(
        ["du", "-sb", "--", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise RuntimeError(f"cannot measure directory size for {path}: {detail}")
    try:
        return int(result.stdout.split(maxsplit=1)[0])
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"unexpected du output for {path}: {result.stdout!r}") from error


def _find_space_reference(output_root: Path) -> tuple[Path, BatchStats]:
    candidates: list[tuple[float, Path, BatchStats]] = []
    for summary_path in output_root.glob("*/batch_summary.jsonl"):
        try:
            stats = _infer_batch_stats(summary_path)
        except (
            OSError,
            RuntimeError,
            ValueError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ):
            continue
        reference = summary_path.parent
        # This orchestrator does not enable --overview or --record-video. Old
        # batches containing rendered videos are therefore not comparable and
        # can overestimate the current per-episode footprint by several times.
        has_recorded_video = any(reference.glob("episode_*/overview_videos/*")) or any(
            reference.glob("episode_*/recording_videos/*")
        )
        if has_recorded_video:
            continue
        bytes_per_episode = _directory_size_bytes(reference) / stats.attempted
        candidates.append((bytes_per_episode, reference, stats))
    if not candidates:
        raise RuntimeError(
            "cannot estimate disk usage: no compatible complete video-free batch was "
            "found under "
            f"{output_root}; pass --space-reference or --estimated-gib-per-episode"
        )
    # Use the largest observed per-episode footprint among compatible runs so
    # one unusually short episode cannot make the preflight optimistic.
    _, path, stats = max(candidates, key=lambda item: (item[0], str(item[1])))
    return path, stats


def _estimate_episode_bytes(args: argparse.Namespace, output_root: Path) -> tuple[float, str]:
    if args.estimated_gib_per_episode is not None:
        return (
            float(args.estimated_gib_per_episode) * GIB,
            f"--estimated-gib-per-episode={args.estimated_gib_per_episode}",
        )

    if args.space_reference is not None:
        reference = args.space_reference.expanduser().resolve()
        summary_path = reference / "batch_summary.jsonl"
        if not reference.is_dir():
            raise RuntimeError(f"space reference is not a directory: {reference}")
        stats = _infer_batch_stats(summary_path)
    else:
        reference, stats = _find_space_reference(output_root)

    size_bytes = _directory_size_bytes(reference)
    return size_bytes / stats.attempted, str(reference)


def _format_gib(byte_count: float) -> str:
    return f"{byte_count / GIB:.2f} GiB"


def _require_success_rate(stats: BatchStats, threshold_percent: float) -> float:
    success_rate = 100.0 * stats.succeeded / stats.attempted
    if success_rate <= threshold_percent:
        raise RuntimeError(
            f"success rate gate failed: {success_rate:.2f}% is not greater than "
            f"{threshold_percent:.2f}%; keeping local data and stopping"
        )
    return success_rate


def _check_free_space(
    *,
    output_root: Path,
    needed_bytes: float,
    reserve_bytes: float,
    context: str,
) -> None:
    free_bytes = shutil.disk_usage(output_root).free
    required_free = needed_bytes + reserve_bytes
    print(
        f"space check ({context}): free={_format_gib(free_bytes)}, "
        f"workload={_format_gib(needed_bytes)}, reserve={_format_gib(reserve_bytes)}, "
        f"required={_format_gib(required_free)}",
        flush=True,
    )
    if free_bytes < required_free:
        shortfall = required_free - free_bytes
        raise RuntimeError(
            f"insufficient disk space before {context}: short by {_format_gib(shortfall)}"
        )


def _round_name(prefix: str, first_seed: int, episode_count: int) -> str:
    cleaned = prefix.strip().strip("/\\")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("name prefix cannot be empty, '.' or '..'")
    if "/" in cleaned or "\\" in cleaned:
        raise ValueError("name prefix must not contain path separators")
    return f"{cleaned}_seed{first_seed}_n{episode_count}"


def _remote_path(prefix: str, archive: Path) -> str:
    cleaned = prefix.strip().strip("/")
    return f"{cleaned}/{archive.name}" if cleaned else archive.name


def _require_command(command: str, *, label: str) -> None:
    path = Path(command).expanduser()
    if path.parent != Path("."):
        if not path.is_file():
            raise RuntimeError(f"{label} does not exist: {path}")
        return
    if shutil.which(command) is None:
        raise RuntimeError(f"{label} is not available on PATH: {command}")


def _resolve_ms_bin(configured_command: str) -> str:
    """Resolve ms-hub CLI from PATH or active/base Conda environments."""

    configured_path = Path(configured_command).expanduser()
    if configured_path.parent != Path("."):
        if configured_path.is_file():
            return str(configured_path.resolve())
        raise RuntimeError(f"ModelScope CLI does not exist: {configured_path}")

    discovered = shutil.which(configured_command)
    if discovered:
        return discovered
    if configured_command != "ms":
        raise RuntimeError(f"ModelScope CLI is not available on PATH: {configured_command}")

    # Depending on the installed version, the console entry point can be named
    # `ms`, `ms-hub`, or `modelscope`.
    for alternative_name in ("ms-hub", "modelscope"):
        alternative_command = shutil.which(alternative_name)
        if alternative_command:
            print(
                f"ModelScope `ms` command was not on PATH; using {alternative_command}",
                flush=True,
            )
            return alternative_command

    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_bin = Path(conda_prefix).expanduser() / "bin"
        candidates.extend(
            [conda_bin / "ms", conda_bin / "ms-hub", conda_bin / "modelscope"]
        )

    # Do not resolve this path first: a Conda environment's python executable
    # may be a symlink to the base interpreter, while its CLI lives beside the
    # symlink in the active environment's bin directory.
    active_python_bin = Path(sys.executable).expanduser().parent
    candidates.extend(
        [
            active_python_bin / "ms",
            active_python_bin / "ms-hub",
            active_python_bin / "modelscope",
        ]
    )

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        conda_base_bin = Path(conda_exe).expanduser().resolve().parent
        candidates.extend(
            [
                conda_base_bin / "ms",
                conda_base_bin / "ms-hub",
                conda_base_bin / "modelscope",
            ]
        )

    python_path = Path(sys.executable).resolve()
    if "envs" in python_path.parts:
        envs_index = python_path.parts.index("envs")
        conda_root = Path(*python_path.parts[:envs_index])
        candidates.extend(
            [
                conda_root / "bin/ms",
                conda_root / "bin/ms-hub",
                conda_root / "bin/modelscope",
            ]
        )

    candidates.extend(
        [
            Path.home() / "miniconda3/bin/ms",
            Path.home() / "miniconda3/bin/ms-hub",
            Path.home() / "miniconda3/bin/modelscope",
            Path.home() / "anaconda3/bin/ms",
            Path.home() / "anaconda3/bin/ms-hub",
            Path.home() / "anaconda3/bin/modelscope",
        ]
    )
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            print(
                f"ModelScope CLI was not on PATH; using {candidate}",
                flush=True,
            )
            return str(candidate)

    raise RuntimeError(
        "ModelScope CLI (`ms`, `ms-hub`, or `modelscope`) is not available on PATH "
        "and was not found in the active/base Conda environments; pass --ms-bin "
        "with the executable used for login"
    )


def _collect_round(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    first_seed: int,
) -> BatchStats:
    summary_path = output_dir / "batch_summary.jsonl"
    if output_dir.exists():
        if not args.resume:
            raise RuntimeError(
                f"round output already exists: {output_dir}; use --resume only if it is complete"
            )
        stats = _read_batch_stats(summary_path, args.episodes_per_round)
        print(
            f"[resume] collection already complete: {stats.attempted} attempted, "
            f"{stats.succeeded} succeeded, {stats.failed} failed",
            flush=True,
        )
        return stats

    command = [
        args.isaac_python,
        "-B",
        str(BATCH_ENTRY),
        "--scene-profile",
        args.scene_profile,
        "--output-dir",
        str(output_dir),
        "--num-episodes",
        str(args.episodes_per_round),
        "--seed",
        str(first_seed),
        "--headless",
        "--navigation-visual-mode",
        args.navigation_visual_mode,
        "--continue-on-failure",
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return_code = _run(command, env=env)

    # The batch entry returns 1 when one or more task episodes fail. That is not
    # an incomplete collection: package it if every requested index has a row.
    stats = _read_batch_stats(summary_path, args.episodes_per_round)
    if return_code != 0:
        print(
            f"[warning] batch exited with {return_code}, but all {stats.attempted} "
            "episode attempts have summaries; packaging the completed round.",
            flush=True,
        )
    return stats


def _archive_round(
    *,
    output_root: Path,
    output_dir: Path,
    archive_path: Path,
    resume: bool,
) -> None:
    if archive_path.exists():
        if not resume:
            raise RuntimeError(
                f"archive already exists: {archive_path}; use --resume to reuse it"
            )
        verify_command = ["tar", "-tf", str(archive_path)]
        print(f"\n$ {_display_command(verify_command)} >/dev/null", flush=True)
        verify_code = subprocess.run(
            verify_command,
            stdout=subprocess.DEVNULL,
            check=False,
        ).returncode
        if verify_code != 0:
            raise RuntimeError(f"existing archive is invalid: {archive_path}")
        print(f"[resume] reusing archive: {archive_path}", flush=True)
        return

    return_code = _run(
        [
            "tar",
            "-cf",
            str(archive_path),
            "-C",
            str(output_root),
            output_dir.name,
        ]
    )
    if return_code != 0:
        raise RuntimeError(f"tar failed with exit code {return_code}")


def _upload_archive(
    *,
    args: argparse.Namespace,
    repo_id: str,
    archive_path: Path,
) -> None:
    command = [
        args.ms_bin,
        "upload",
        repo_id,
        str(archive_path),
        _remote_path(args.remote_prefix, archive_path),
        "--repo-type",
        "dataset",
        "--commit-message",
        args.commit_message,
    ]
    if args.revision:
        command.extend(["--revision", args.revision])
    return_code = _run(command)
    if return_code != 0:
        raise RuntimeError(f"ModelScope upload failed with exit code {return_code}")


def _cleanup_round(
    *,
    output_root: Path,
    output_dir: Path,
    archive_path: Path,
) -> None:
    """Delete one uploaded round while guarding against broad path removal."""

    resolved_root = output_root.resolve()
    resolved_output = output_dir.resolve()
    resolved_archive = archive_path.resolve()
    if (
        output_dir.is_symlink()
        or archive_path.is_symlink()
        or resolved_output.parent != resolved_root
        or resolved_archive.parent != resolved_root
        or resolved_output == resolved_root
        or resolved_archive == resolved_root
        or resolved_output.name in {"", ".", ".."}
        or resolved_archive.suffix != ".tar"
    ):
        raise RuntimeError(
            "refusing unsafe cleanup targets: "
            f"output_dir={resolved_output}, archive={resolved_archive}"
        )

    # Keep the uploaded tar available until the larger source directory is gone.
    # If directory cleanup fails, the archive remains available for recovery.
    if resolved_output.is_dir():
        print(f"deleting uploaded source directory: {resolved_output}", flush=True)
        shutil.rmtree(resolved_output)
    elif resolved_output.exists():
        raise RuntimeError(f"round output is not a directory: {resolved_output}")

    if resolved_archive.is_file():
        print(f"deleting uploaded archive: {resolved_archive}", flush=True)
        resolved_archive.unlink()
    elif resolved_archive.exists():
        raise RuntimeError(f"archive path is not a file: {resolved_archive}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        repo_id = normalize_dataset_repo_id(args.dataset_url)
        output_root = args.output_root.expanduser().resolve()
        _round_name(args.name_prefix, args.start_seed, args.episodes_per_round)

        if not args.dry_run:
            if not BATCH_ENTRY.is_file():
                raise RuntimeError(f"batch entry does not exist: {BATCH_ENTRY}")
            if not args.isaac_python:
                raise RuntimeError(
                    "ISAAC_PYTHON is not set; export it or pass --isaac-python"
                )
            _require_command(args.isaac_python, label="Isaac Python")
            _require_command("tar", label="tar")
            _require_command("du", label="du")
            args.ms_bin = _resolve_ms_bin(args.ms_bin)
            output_root.mkdir(parents=True, exist_ok=True)

            estimated_episode_bytes, estimate_source = _estimate_episode_bytes(
                args,
                output_root,
            )
            print(
                f"space estimate source={estimate_source}; "
                f"average_per_episode={_format_gib(estimated_episode_bytes)}",
                flush=True,
            )
        else:
            estimated_episode_bytes = (
                float(args.estimated_gib_per_episode) * GIB
                if args.estimated_gib_per_episode is not None
                else None
            )

        print(
            f"dataset={repo_id} rounds={args.rounds} "
            f"episodes_per_round={args.episodes_per_round} output_root={output_root}",
            flush=True,
        )

        for round_index in range(args.rounds):
            first_seed = args.start_seed + round_index * args.episodes_per_round
            last_seed = first_seed + args.episodes_per_round - 1
            name = _round_name(args.name_prefix, first_seed, args.episodes_per_round)
            output_dir = output_root / name
            archive_path = output_root / f"{name}.tar"

            print(
                f"\n=== round {round_index + 1}/{args.rounds}: "
                f"seeds {first_seed}..{last_seed} ===",
                flush=True,
            )

            if args.dry_run:
                batch_command = [
                    args.isaac_python or "$ISAAC_PYTHON",
                    "-B",
                    str(BATCH_ENTRY),
                    "--scene-profile",
                    args.scene_profile,
                    "--output-dir",
                    str(output_dir),
                    "--num-episodes",
                    str(args.episodes_per_round),
                    "--seed",
                    str(first_seed),
                    "--headless",
                    "--navigation-visual-mode",
                    args.navigation_visual_mode,
                    "--continue-on-failure",
                ]
                upload_command = [
                    args.ms_bin,
                    "upload",
                    repo_id,
                    str(archive_path),
                    _remote_path(args.remote_prefix, archive_path),
                    "--repo-type",
                    "dataset",
                    "--commit-message",
                    args.commit_message,
                ]
                if args.revision:
                    upload_command.extend(["--revision", args.revision])
                print(
                    f"[space preflight] require room for raw data + tar, "
                    f"factor={args.space_safety_factor}, "
                    f"reserve={args.free_space_reserve_gib:.2f} GiB"
                )
                print(f"$ {_display_command(batch_command)}")
                print(
                    f"[success gate] require success_rate > "
                    f"{args.min_success_rate:.2f}%"
                )
                print(
                    "$ "
                    + _display_command(
                        [
                            "tar",
                            "-cf",
                            str(archive_path),
                            "-C",
                            str(output_root),
                            output_dir.name,
                        ]
                    )
                )
                print(f"$ {_display_command(upload_command)}")
                if not args.keep_local:
                    print(f"[cleanup after successful upload] {output_dir}")
                    print(f"[cleanup after successful upload] {archive_path}")
                continue

            if archive_path.exists() and not output_dir.exists():
                raise RuntimeError(
                    f"archive exists without its round directory: {archive_path}; "
                    "refusing to recollect against a possibly stale archive"
                )

            reserve_bytes = float(args.free_space_reserve_gib) * GIB
            if output_dir.exists():
                if not args.resume:
                    raise RuntimeError(
                        f"round output already exists: {output_dir}; "
                        "use --resume only if it is complete"
                    )
                if archive_path.exists():
                    preflight_workload_bytes = 0.0
                    preflight_context = "resumed upload"
                else:
                    preflight_workload_bytes = (
                        _directory_size_bytes(output_dir) * args.space_safety_factor
                    )
                    preflight_context = "resumed tar creation"
            else:
                if estimated_episode_bytes is None:
                    raise RuntimeError("internal error: disk estimate is unavailable")
                projected_raw_bytes = (
                    estimated_episode_bytes * args.episodes_per_round
                )
                preflight_workload_bytes = (
                    projected_raw_bytes * 2.0 * args.space_safety_factor
                )
                preflight_context = "collection plus tar creation"
                print(
                    f"projected raw round size={_format_gib(projected_raw_bytes)}; "
                    "checking concurrent raw + tar capacity",
                    flush=True,
                )
            _check_free_space(
                output_root=output_root,
                needed_bytes=preflight_workload_bytes,
                reserve_bytes=reserve_bytes,
                context=preflight_context,
            )

            stats = _collect_round(
                args=args,
                output_dir=output_dir,
                first_seed=first_seed,
            )
            success_rate = 100.0 * stats.succeeded / stats.attempted
            print(
                f"round result: {stats.succeeded}/{stats.attempted} succeeded "
                f"({success_rate:.2f}%), {stats.failed} failed",
                flush=True,
            )
            _require_success_rate(stats, args.min_success_rate)

            actual_raw_bytes = _directory_size_bytes(output_dir)
            estimated_episode_bytes = max(
                estimated_episode_bytes or 0.0,
                actual_raw_bytes / stats.attempted,
            )
            if not archive_path.exists():
                _check_free_space(
                    output_root=output_root,
                    needed_bytes=actual_raw_bytes * args.space_safety_factor,
                    reserve_bytes=reserve_bytes,
                    context="tar creation after collection",
                )
            _archive_round(
                output_root=output_root,
                output_dir=output_dir,
                archive_path=archive_path,
                resume=args.resume,
            )
            _upload_archive(args=args, repo_id=repo_id, archive_path=archive_path)
            print(f"uploaded: {archive_path.name} -> {repo_id}", flush=True)
            if args.keep_local:
                print(
                    f"keeping local source and archive because --keep-local was set: "
                    f"{output_dir}, {archive_path}",
                    flush=True,
                )
            else:
                _cleanup_round(
                    output_root=output_root,
                    output_dir=output_dir,
                    archive_path=archive_path,
                )
                print("local source and archive deleted after upload", flush=True)

        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
