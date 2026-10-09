from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque
from urllib.request import urlretrieve

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions


@dataclass(frozen=True)
class AppConfig:
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    closed_eyes_ear: float = 0.24
    perclos_ear: float = 0.26
    yawning_mar: float = 0.60
    head_down_pitch: float = 0.60
    early_drowsy_score: float = 0.30
    drowsy_score: float = 0.52


CONFIG = AppConfig()
DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
DEFAULT_MODEL_PATH = Path("face_landmarker.task")

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 308, 311, 14]
NOSE_TIP = 1


@dataclass(frozen=True)
class FaceMetrics:
    ear: float
    mar: float
    head_pitch_proxy: float


def _distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def eye_aspect_ratio(eye_points: list[np.ndarray]) -> float:
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical = _distance(p2, p6) + _distance(p3, p5)
    horizontal = 2.0 * _distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def mouth_aspect_ratio(mouth_points: list[np.ndarray]) -> float:
    p1, p2, p3, p4, p5, p6 = mouth_points
    vertical = _distance(p2, p6) + _distance(p3, p5)
    horizontal = 2.0 * _distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def head_pitch_proxy(nose: np.ndarray, eye_mid: np.ndarray, mouth_mid: np.ndarray) -> float:
    top_span = nose[1] - eye_mid[1]
    lower_span = mouth_mid[1] - eye_mid[1]
    if lower_span <= 0:
        return 0.0
    return float(top_span / lower_span)


class FaceAnalyzer:
    def __init__(
        self,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_path: Path,
    ) -> None:
        model_path = ensure_model_exists(model_path)
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path.resolve())),
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._face_landmarker = FaceLandmarker.create_from_options(options)

    def analyze(self, frame_bgr: np.ndarray) -> FaceMetrics | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._face_landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        face_landmarks = result.face_landmarks[0]
        height, width = frame_bgr.shape[:2]
        landmarks_px = np.array(
            [[lm.x * width, lm.y * height] for lm in face_landmarks],
            dtype=np.float32,
        )

        left_eye = [landmarks_px[idx] for idx in LEFT_EYE]
        right_eye = [landmarks_px[idx] for idx in RIGHT_EYE]
        mouth = [landmarks_px[idx] for idx in MOUTH]
        nose = landmarks_px[NOSE_TIP]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)
        eye_mid = (left_eye[0] + right_eye[3]) / 2.0
        mouth_mid = (mouth[0] + mouth[3]) / 2.0
        pitch = head_pitch_proxy(nose, eye_mid, mouth_mid)
        return FaceMetrics(ear=ear, mar=mar, head_pitch_proxy=pitch)

    def close(self) -> None:
        self._face_landmarker.close()


def ensure_model_exists(model_path: Path) -> Path:
    if model_path.exists():
        return model_path
    print(f"Downloading Face Landmarker model to {model_path} ...")
    urlretrieve(DEFAULT_MODEL_URL, str(model_path))
    return model_path


@dataclass(frozen=True)
class ClipRiskSummary:
    state: str
    risk_score: float
    confidence: float
    eyes_closed_ratio: float
    perclos_ratio: float
    eye_closure_severity: float
    sustained_eye_closure_ratio: float
    yawning_ratio: float
    mouth_open_severity: float
    head_down_ratio: float
    head_down_severity: float


def summarize_ratios(
    eyes_closed_ratio: float,
    perclos_ratio: float,
    eye_closure_severity: float,
    sustained_eye_closure_ratio: float,
    yawning_ratio: float,
    mouth_open_severity: float,
    head_down_ratio: float,
    head_down_severity: float,
) -> ClipRiskSummary:
    sustained_eye_signal = _clamp01(sustained_eye_closure_ratio / 0.35)
    eye_signal = min(
        1.0,
        0.30 * eyes_closed_ratio
        + 0.30 * perclos_ratio
        + 0.25 * eye_closure_severity
        + 0.15 * sustained_eye_signal,
    )
    mouth_signal = min(1.0, 0.60 * yawning_ratio + 0.40 * mouth_open_severity)
    head_signal = min(1.0, 0.55 * head_down_ratio + 0.45 * head_down_severity)
    risk_score = min(1.0, 0.55 * eye_signal + 0.20 * mouth_signal + 0.25 * head_signal)
    support_signal = 0.65 * eye_signal + 0.20 * mouth_signal + 0.15 * head_signal

    if risk_score >= CONFIG.drowsy_score:
        state = "drowsy"
        margin = _clamp01((risk_score - CONFIG.drowsy_score) / max(1e-6, 1.0 - CONFIG.drowsy_score))
        confidence = _clamp01(0.60 * margin + 0.40 * support_signal)
    elif risk_score >= CONFIG.early_drowsy_score:
        state = "low_vigilance"
        band_pos = _clamp01(
            (risk_score - CONFIG.early_drowsy_score)
            / max(1e-6, CONFIG.drowsy_score - CONFIG.early_drowsy_score)
        )
        center_strength = 1.0 - abs(band_pos - 0.5) / 0.5
        confidence = _clamp01(0.55 * center_strength + 0.45 * support_signal)
    else:
        state = "alert"
        calm_signal = 1.0 - support_signal
        margin = _clamp01((CONFIG.early_drowsy_score - risk_score) / max(1e-6, CONFIG.early_drowsy_score))
        confidence = _clamp01(0.60 * margin + 0.40 * calm_signal)

    return ClipRiskSummary(
        state=state,
        risk_score=risk_score,
        confidence=confidence,
        eyes_closed_ratio=eyes_closed_ratio,
        perclos_ratio=perclos_ratio,
        eye_closure_severity=eye_closure_severity,
        sustained_eye_closure_ratio=sustained_eye_closure_ratio,
        yawning_ratio=yawning_ratio,
        mouth_open_severity=mouth_open_severity,
        head_down_ratio=head_down_ratio,
        head_down_severity=head_down_severity,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a video into heuristic drowsiness clips.")
    parser.add_argument("video_path", type=Path, help="Input video path or directory containing videos")
    parser.add_argument("--output", type=Path, help="Output CSV path")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Face landmarker .task model path")
    parser.add_argument("--clip-seconds", type=float, default=5.0, help="Clip length in seconds")
    parser.add_argument("--stride-seconds", type=float, default=2.5, help="Sliding stride in seconds")
    parser.add_argument("--min-confidence", type=float, default=0.25, help="Keep clips above this confidence")
    parser.add_argument("--alert-confidence", type=float, help="Per-label confidence threshold for alert")
    parser.add_argument("--low-vigilance-confidence", type=float, help="Per-label confidence threshold for low_vigilance")
    parser.add_argument("--drowsy-confidence", type=float, help="Per-label confidence threshold for drowsy")
    parser.add_argument("--keep-all", action="store_true", help="Write all clips including uncertain ones")
    parser.add_argument("--export-clips", action="store_true", help="Export filtered clips into label subfolders")
    parser.add_argument("--clips-dir", type=Path, help="Output directory for exported clips")
    parser.add_argument("--debug-frames", action="store_true", help="Export per-frame EAR/MAR/PITCH debug CSV")
    parser.add_argument("--debug-dir", type=Path, help="Output directory for per-frame debug CSV files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video_path.exists():
        raise FileNotFoundError(f"Input path not found: {args.video_path}")

    video_paths = collect_video_paths(args.video_path)
    if not video_paths:
        raise FileNotFoundError(f"No video files found in: {args.video_path}")

    print(f"[Log] Found {len(video_paths)} video(s) to process.")
    total_rows = 0
    label_thresholds = build_label_thresholds(args)
    for video_path in video_paths:
        print(f"\n[Log] --- Starting to process video: {video_path.name} ---")
        output_path = resolve_output_csv_path(args.output, video_path, args.video_path)
        rows, debug_rows = analyze_video(
            video_path,
            args.clip_seconds,
            args.stride_seconds,
            args.min_confidence,
            label_thresholds,
            args.keep_all,
            args.model_path,
            args.debug_frames,
        )
        write_csv(output_path, rows)
        if args.export_clips:
            clips_dir = resolve_clips_dir(args.clips_dir, video_path, args.video_path)
            export_clips(video_path, rows, clips_dir)
        if args.debug_frames:
            debug_path = resolve_debug_csv_path(args.debug_dir, video_path, args.video_path)
            write_frame_debug_csv(debug_path, debug_rows)
        print(f"Wrote {len(rows)} clip rows to {output_path}")
        total_rows += len(rows)

    if len(video_paths) > 1:
        print(f"Processed {len(video_paths)} videos, total kept clips: {total_rows}")


def analyze_video(
    video_path: Path,
    clip_seconds: float,
    stride_seconds: float,
    min_confidence: float,
    label_thresholds: dict[str, float],
    keep_all: bool,
    model_path: Path,
    debug_frames: bool,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 20.0

    clip_frames = max(1, int(round(clip_seconds * fps)))
    stride_frames = max(1, int(round(stride_seconds * fps)))
    buffer: Deque[dict[str, float | int]] = deque(maxlen=clip_frames)
    next_emit_start = 0
    frame_idx = 0
    rows: list[dict[str, float | int | str]] = []
    debug_rows: list[dict[str, float | int | str]] = []

    analyzer = FaceAnalyzer(
        min_detection_confidence=CONFIG.min_detection_confidence,
        min_tracking_confidence=CONFIG.min_tracking_confidence,
        model_path=model_path,
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print(f"[Log] Finished reading all frames. Total frames processed: {frame_idx}")
                break

            if frame_idx > 0 and frame_idx % 1000 == 0:
                print(f"[Log] Analyzing frame {frame_idx}...")

            metrics = analyzer.analyze(frame)
            if metrics is None:
                frame_idx += 1
                continue

            buffer.append(
                {
                    "frame_idx": frame_idx,
                    "eyes_closed": 1 if metrics.ear < CONFIG.closed_eyes_ear else 0,
                    "perclos_closed": 1 if metrics.ear < CONFIG.perclos_ear else 0,
                    "yawning": 1 if metrics.mar > CONFIG.yawning_mar else 0,
                    "head_down": 1 if metrics.head_pitch_proxy > CONFIG.head_down_pitch else 0,
                    "ear": metrics.ear,
                    "mar": metrics.mar,
                    "pitch": metrics.head_pitch_proxy,
                }
            )

            if debug_frames:
                debug_rows.append(
                    build_frame_debug_row(video_path, frame_idx, fps, metrics)
                )

            clip_start = frame_idx - clip_frames + 1
            if clip_start >= next_emit_start and len(buffer) == clip_frames:
                row = summarize_clip(video_path, list(buffer), fps)
                if keep_all or should_keep_row(row, min_confidence, label_thresholds):
                    rows.append(row)
                next_emit_start = clip_start + stride_frames

            frame_idx += 1
    finally:
        analyzer.close()
        cap.release()

    return rows, debug_rows


def collect_video_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    allowed_suffixes = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg"}
    return sorted(path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in allowed_suffixes)


def resolve_output_csv_path(output_arg: Path | None, video_path: Path, input_root: Path) -> Path:
    if output_arg is None:
        return video_path.with_name(f"{video_path.stem}_filtered.csv")

    if input_root.is_file():
        return output_arg

    output_arg.mkdir(parents=True, exist_ok=True)
    return output_arg / f"{video_path.stem}_filtered.csv"


def resolve_clips_dir(clips_dir_arg: Path | None, video_path: Path, input_root: Path) -> Path:
    if clips_dir_arg is None:
        return video_path.with_name(f"{video_path.stem}_clips")

    if input_root.is_file():
        return clips_dir_arg

    clips_dir_arg.mkdir(parents=True, exist_ok=True)
    return clips_dir_arg / video_path.stem


def resolve_debug_csv_path(debug_dir_arg: Path | None, video_path: Path, input_root: Path) -> Path:
    if debug_dir_arg is None:
        return video_path.with_name(f"{video_path.stem}_frame_debug.csv")

    if input_root.is_file():
        return debug_dir_arg

    debug_dir_arg.mkdir(parents=True, exist_ok=True)
    return debug_dir_arg / f"{video_path.stem}_frame_debug.csv"


def build_frame_debug_row(
    video_path: Path,
    frame_idx: int,
    fps: float,
    metrics: FaceMetrics,
) -> dict[str, float | int | str]:
    eye_closure_severity = _clamp01((CONFIG.closed_eyes_ear - metrics.ear) / 0.10)
    mouth_open_severity = _clamp01((metrics.mar - CONFIG.yawning_mar) / 0.20)
    head_down_severity = _clamp01((metrics.head_pitch_proxy - CONFIG.head_down_pitch) / 0.20)
    return {
        "video": video_path.name,
        "frame_idx": frame_idx,
        "sec": round(frame_idx / fps, 3),
        "ear": round(metrics.ear, 4),
        "mar": round(metrics.mar, 4),
        "pitch": round(metrics.head_pitch_proxy, 4),
        "eyes_closed": 1 if metrics.ear < CONFIG.closed_eyes_ear else 0,
        "perclos_closed": 1 if metrics.ear < CONFIG.perclos_ear else 0,
        "eye_closure_severity": round(eye_closure_severity, 4),
        "yawning": 1 if metrics.mar > CONFIG.yawning_mar else 0,
        "mouth_open_severity": round(mouth_open_severity, 4),
        "head_down": 1 if metrics.head_pitch_proxy > CONFIG.head_down_pitch else 0,
        "head_down_severity": round(head_down_severity, 4),
    }


def summarize_clip(video_path: Path, clip_data: list[dict[str, float | int]], fps: float) -> dict[str, float | int | str]:
    total = len(clip_data)
    eyes_closed_ratio = sum(int(item["eyes_closed"]) for item in clip_data) / total
    perclos_ratio = sum(int(item["perclos_closed"]) for item in clip_data) / total
    yawning_ratio = sum(int(item["yawning"]) for item in clip_data) / total
    head_down_ratio = sum(int(item["head_down"]) for item in clip_data) / total
    eye_closure_severity = sum(
        _clamp01((CONFIG.closed_eyes_ear - float(item["ear"])) / 0.10) for item in clip_data
    ) / total
    mouth_open_severity = sum(
        _clamp01((float(item["mar"]) - CONFIG.yawning_mar) / 0.20) for item in clip_data
    ) / total
    head_down_severity = sum(
        _clamp01((float(item["pitch"]) - CONFIG.head_down_pitch) / 0.20) for item in clip_data
    ) / total

    longest_closed_run = 0
    current_closed_run = 0
    for item in clip_data:
        if int(item["eyes_closed"]):
            current_closed_run += 1
            longest_closed_run = max(longest_closed_run, current_closed_run)
        else:
            current_closed_run = 0
    sustained_eye_closure_ratio = longest_closed_run / total

    summary = summarize_ratios(
        eyes_closed_ratio,
        perclos_ratio,
        eye_closure_severity,
        sustained_eye_closure_ratio,
        yawning_ratio,
        mouth_open_severity,
        head_down_ratio,
        head_down_severity,
    )

    start_frame = int(clip_data[0]["frame_idx"])
    end_frame = int(clip_data[-1]["frame_idx"])
    return {
        "video": video_path.name,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_sec": round(start_frame / fps, 3),
        "end_sec": round(end_frame / fps, 3),
        "state": summary.state,
        "confidence": round(summary.confidence, 4),
        "risk_score": round(summary.risk_score, 4),
        "eyes_closed_ratio": round(summary.eyes_closed_ratio, 4),
        "perclos_ratio": round(summary.perclos_ratio, 4),
        "eye_closure_severity": round(summary.eye_closure_severity, 4),
        "sustained_eye_closure_ratio": round(summary.sustained_eye_closure_ratio, 4),
        "yawning_ratio": round(summary.yawning_ratio, 4),
        "mouth_open_severity": round(summary.mouth_open_severity, 4),
        "head_down_ratio": round(summary.head_down_ratio, 4),
        "head_down_severity": round(summary.head_down_severity, 4),
        "mean_ear": round(sum(float(item["ear"]) for item in clip_data) / total, 4),
        "mean_mar": round(sum(float(item["mar"]) for item in clip_data) / total, 4),
        "mean_pitch": round(sum(float(item["pitch"]) for item in clip_data) / total, 4),
    }


def write_csv(output_path: Path, rows: list[dict[str, float | int | str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video",
        "start_frame",
        "end_frame",
        "start_sec",
        "end_sec",
        "state",
        "confidence",
        "risk_score",
        "eyes_closed_ratio",
        "perclos_ratio",
        "eye_closure_severity",
        "sustained_eye_closure_ratio",
        "yawning_ratio",
        "mouth_open_severity",
        "head_down_ratio",
        "head_down_severity",
        "mean_ear",
        "mean_mar",
        "mean_pitch",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_frame_debug_csv(output_path: Path, debug_rows: list[dict[str, float | int | str]]) -> None:
    if not debug_rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video",
        "frame_idx",
        "sec",
        "ear",
        "mar",
        "pitch",
        "eyes_closed",
        "perclos_closed",
        "eye_closure_severity",
        "yawning",
        "mouth_open_severity",
        "head_down",
        "head_down_severity",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(debug_rows)


def build_label_thresholds(args: argparse.Namespace) -> dict[str, float]:
    thresholds: dict[str, float] = {"alert": 0.0}
    if args.alert_confidence is not None:
        thresholds["alert"] = args.alert_confidence
    if args.low_vigilance_confidence is not None:
        thresholds["low_vigilance"] = args.low_vigilance_confidence
    if args.drowsy_confidence is not None:
        thresholds["drowsy"] = args.drowsy_confidence
    return thresholds


def should_keep_row(
    row: dict[str, float | int | str],
    min_confidence: float,
    label_thresholds: dict[str, float],
) -> bool:
    state = str(row["state"])
    confidence = float(row["confidence"])
    threshold = label_thresholds.get(state, min_confidence)
    return confidence >= threshold


def export_clips(video_path: Path, rows: list[dict[str, float | int | str]], clips_dir: Path) -> None:
    clips_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Log] Exporting {len(rows)} clips to {clips_dir}...")
    for idx, row in enumerate(rows):
        label_dir = clips_dir / str(row["state"])
        label_dir.mkdir(parents=True, exist_ok=True)
        clip_name = (
            f"{video_path.stem}_{idx:05d}_"
            f"{float(row['start_sec']):08.3f}_{float(row['end_sec']):08.3f}_"
            f"c{float(row['confidence']):.2f}.mp4"
        )
        export_single_clip(video_path, label_dir / clip_name, int(row["start_frame"]), int(row["end_frame"]))


def export_single_clip(video_path: Path, output_path: Path, start_frame: int, end_frame: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for clip export: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        while frame_idx <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frame_idx += 1
    finally:
        writer.release()
        cap.release()


if __name__ == "__main__":
    main()
