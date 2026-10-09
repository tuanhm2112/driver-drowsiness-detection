import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm


KEY_LANDMARKS = [
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
    61, 291, 13, 14, 82, 312, 87, 317, 78, 308, 95, 324, 0, 17,
    1, 2, 5, 4, 19, 94,
    468, 469, 470, 471, 472,
    473, 474, 475, 476, 477,
    10, 152, 234, 454, 127, 356, 162, 389,
]
LANDMARK_TO_POS = {landmark_id: position for position, landmark_id in enumerate(KEY_LANDMARKS)}
LEFT_EYE_POS = [LANDMARK_TO_POS[idx] for idx in [33, 160, 158, 133, 153, 144]]
RIGHT_EYE_POS = [LANDMARK_TO_POS[idx] for idx in [362, 385, 387, 263, 373, 380]]
MOUTH_POS = [LANDMARK_TO_POS[idx] for idx in [61, 291, 13, 14, 82, 312, 87, 317]]
LEFT_BROW_CENTER = [LANDMARK_TO_POS[idx] for idx in [70, 63, 105, 66, 107]]
RIGHT_BROW_CENTER = [LANDMARK_TO_POS[idx] for idx in [300, 293, 334, 296, 336]]
LEFT_IRIS_CENTER = [LANDMARK_TO_POS[idx] for idx in [468, 469, 470, 471, 472]]
RIGHT_IRIS_CENTER = [LANDMARK_TO_POS[idx] for idx in [473, 474, 475, 476, 477]]
NOSE_POS = LANDMARK_TO_POS[1]
CHIN_POS = LANDMARK_TO_POS[152]
FOREHEAD_POS = LANDMARK_TO_POS[10]
LEFT_FACE_POS = LANDMARK_TO_POS[234]
RIGHT_FACE_POS = LANDMARK_TO_POS[454]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sequence classifier from NPZ splits.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root folder containing train/val/test.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_npz"), help="Where to save outputs.")
    parser.add_argument("--sequence-length", type=int, default=90, help="Target sequence length after resizing.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-metric", choices=["loss", "f1", "accuracy"], default="loss")
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--optimize-threshold", action="store_true")
    parser.add_argument("--threshold-metric", choices=["f1", "accuracy"], default="f1")
    parser.add_argument("--append-ear-mar", action="store_true")
    parser.add_argument("--personalize-handcrafted", action="store_true")
    parser.add_argument("--handcrafted-only", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--scheduler-patience", type=int, default=1)
    parser.add_argument("--scheduler-factor", type=float, default=0.3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-noise-std", type=float, default=0.01)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def discover_labels(data_root: Path) -> Dict[str, int]:
    train_dir = data_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {train_dir}")
    labels = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    if not labels:
        raise RuntimeError("No class folders found under train/")
    return {label_name: index for index, label_name in enumerate(labels)}


def find_npz_files(split_dir: Path, label_map: Dict[str, int]) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for label_name, label_id in label_map.items():
        label_dir = split_dir / label_name
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.rglob("*.npz")):
            samples.append((path, label_id))
    if not samples:
        raise RuntimeError(f"No .npz files found under {split_dir}")
    return samples


def choose_feature_array(npz_file: np.lib.npyio.NpzFile) -> np.ndarray:
    preferred_keys = ["x", "X", "features", "feature", "sequence", "sequences", "data", "arr_0"]
    for key in preferred_keys:
        if key in npz_file.files:
            array = np.asarray(npz_file[key])
            if array.size > 0:
                return array

    candidates = []
    for key in npz_file.files:
        array = np.asarray(npz_file[key])
        if array.ndim >= 2 and array.size > 0:
            candidates.append(array)
    if candidates:
        candidates.sort(key=lambda arr: arr.size, reverse=True)
        return candidates[0]

    raise RuntimeError(f"Could not find a usable feature array. Keys: {npz_file.files}")


def choose_label_array(npz_file: np.lib.npyio.NpzFile) -> np.ndarray | None:
    preferred_keys = ["labels", "y", "Y", "target", "targets", "arr_1"]
    for key in preferred_keys:
        if key in npz_file.files:
            array = np.asarray(npz_file[key])
            if array.size > 0:
                return array
    return None


def coerce_sequence(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        if array.shape[0] == 1:
            return array[0]
        if array.shape[-1] == 1:
            return array.reshape(array.shape[0], array.shape[1])
        if array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
            array = np.transpose(array, (1, 0, 2))
        return array.reshape(array.shape[0], -1)
    return array.reshape(array.shape[0], -1)


def coerce_sequence_batch(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        array = normalize_landmark_batch(array)
        return array.reshape(array.shape[0], array.shape[1], -1)
    if array.ndim >= 5:
        array = normalize_landmark_batch(array)
        return array.reshape(array.shape[0], array.shape[1], -1)
    raise RuntimeError(f"Unsupported feature array shape: {array.shape}")


def normalize_landmark_batch(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32).copy()
    if array.ndim < 4:
        return array

    xy = array[..., :2]
    center_xy = xy.mean(axis=2, keepdims=True)
    centered_xy = xy - center_xy

    min_xy = xy.min(axis=2, keepdims=True)
    max_xy = xy.max(axis=2, keepdims=True)
    face_size = np.maximum(max_xy - min_xy, 1e-6)
    scale = np.max(face_size, axis=-1, keepdims=True)
    centered_xy = centered_xy / scale
    array[..., :2] = centered_xy

    if array.shape[-1] >= 3:
        z = array[..., 2:3]
        center_z = z.mean(axis=2, keepdims=True)
        array[..., 2:3] = (z - center_z) / scale

    return array


def euclidean_distance(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points_a - points_b, axis=-1)


def compute_eye_aspect_ratio_batch(array: np.ndarray, eye_positions: Sequence[int]) -> np.ndarray:
    p1, p2, p3, p4, p5, p6 = [array[:, :, position, :2] for position in eye_positions]
    vertical = euclidean_distance(p2, p6) + euclidean_distance(p3, p5)
    horizontal = 2.0 * np.maximum(euclidean_distance(p1, p4), 1e-6)
    return (vertical / horizontal).astype(np.float32)


def compute_mouth_aspect_ratio_batch(array: np.ndarray) -> np.ndarray:
    left_corner, right_corner, upper_inner, lower_inner, upper_mid, lower_mid, upper_outer, lower_outer = [
        array[:, :, position, :2] for position in MOUTH_POS
    ]
    vertical = (
        euclidean_distance(upper_inner, lower_inner)
        + euclidean_distance(upper_mid, lower_mid)
        + euclidean_distance(upper_outer, lower_outer)
    )
    horizontal = 3.0 * np.maximum(euclidean_distance(left_corner, right_corner), 1e-6)
    return (vertical / horizontal).astype(np.float32)


def build_handcrafted_features(array: np.ndarray) -> np.ndarray:
    left_ear = compute_eye_aspect_ratio_batch(array, LEFT_EYE_POS)
    right_ear = compute_eye_aspect_ratio_batch(array, RIGHT_EYE_POS)
    ear = ((left_ear + right_ear) / 2.0).astype(np.float32)
    mar = compute_mouth_aspect_ratio_batch(array)

    left_eye_center = array[:, :, LEFT_EYE_POS, :2].mean(axis=2)
    right_eye_center = array[:, :, RIGHT_EYE_POS, :2].mean(axis=2)
    left_brow_center = array[:, :, LEFT_BROW_CENTER, :2].mean(axis=2)
    right_brow_center = array[:, :, RIGHT_BROW_CENTER, :2].mean(axis=2)
    left_iris_center = array[:, :, LEFT_IRIS_CENTER, :2].mean(axis=2)
    right_iris_center = array[:, :, RIGHT_IRIS_CENTER, :2].mean(axis=2)

    left_brow_eye = euclidean_distance(left_brow_center, left_eye_center).astype(np.float32)
    right_brow_eye = euclidean_distance(right_brow_center, right_eye_center).astype(np.float32)
    brow_eye = ((left_brow_eye + right_brow_eye) / 2.0).astype(np.float32)

    left_gaze = euclidean_distance(left_iris_center, left_eye_center).astype(np.float32)
    right_gaze = euclidean_distance(right_iris_center, right_eye_center).astype(np.float32)
    gaze_offset = ((left_gaze + right_gaze) / 2.0).astype(np.float32)

    nose = array[:, :, NOSE_POS, :2]
    chin = array[:, :, CHIN_POS, :2]
    forehead = array[:, :, FOREHEAD_POS, :2]
    left_face = array[:, :, LEFT_FACE_POS, :2]
    right_face = array[:, :, RIGHT_FACE_POS, :2]

    face_height = np.maximum(euclidean_distance(forehead, chin), 1e-6).astype(np.float32)
    face_width = np.maximum(euclidean_distance(left_face, right_face), 1e-6).astype(np.float32)
    head_ratio = (face_width / face_height).astype(np.float32)
    nose_mid_y = (((forehead[:, :, 1] + chin[:, :, 1]) / 2.0) - nose[:, :, 1]).astype(np.float32)
    nose_mid_x = (nose[:, :, 0] - ((left_face[:, :, 0] + right_face[:, :, 0]) / 2.0)).astype(np.float32)

    return np.stack(
        [ear, mar, brow_eye, gaze_offset, head_ratio, nose_mid_x, nose_mid_y],
        axis=-1,
    ).astype(np.float32)


def extract_user_key(file_path: Path) -> str:
    match = re.search(r"user_(\d+)", file_path.stem)
    return match.group(1) if match else file_path.stem


def compute_handcrafted_baseline(handcrafted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = handcrafted.reshape(-1, handcrafted.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def build_user_baselines(split_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    baselines: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    alert_dir = split_dir / "alert"
    if not alert_dir.exists():
        return baselines

    for file_path in sorted(alert_dir.rglob("*.npz")):
        with np.load(file_path, allow_pickle=False) as npz_file:
            feature_array = choose_feature_array(npz_file)
        if feature_array.ndim < 4:
            continue
        normalized = normalize_landmark_batch(feature_array)
        handcrafted = build_handcrafted_features(normalized)
        baselines[extract_user_key(file_path)] = compute_handcrafted_baseline(handcrafted)
    return baselines


def personalize_handcrafted_features(
    handcrafted: np.ndarray,
    file_path: Path,
    user_baselines: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    user_key = extract_user_key(file_path)
    if user_key in user_baselines:
        mean, std = user_baselines[user_key]
    else:
        mean, std = compute_handcrafted_baseline(handcrafted)
    return ((handcrafted - mean) / std).astype(np.float32)


def resize_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    current_length, feature_dim = sequence.shape
    if current_length == target_length:
        return sequence.astype(np.float32)
    old_positions = np.linspace(0.0, 1.0, num=current_length)
    new_positions = np.linspace(0.0, 1.0, num=target_length)
    resized = np.empty((target_length, feature_dim), dtype=np.float32)
    for feature_index in range(feature_dim):
        resized[:, feature_index] = np.interp(new_positions, old_positions, sequence[:, feature_index])
    return resized


def resize_sequence_batch(sequences: np.ndarray, target_length: int) -> np.ndarray:
    resized = [resize_sequence(sequence, target_length) for sequence in sequences]
    return np.stack(resized).astype(np.float32)


def load_split(
    split_dir: Path,
    label_map: Dict[str, int],
    sequence_length: int,
    append_ear_mar: bool,
    personalize_handcrafted: bool,
    handcrafted_only: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    samples = find_npz_files(split_dir, label_map)
    features: List[np.ndarray] = []
    labels: List[int] = []
    user_baselines = build_user_baselines(split_dir) if append_ear_mar and personalize_handcrafted else {}

    for file_path, label_id in tqdm(samples, desc=f"Loading {split_dir.name}"):
        with np.load(file_path, allow_pickle=False) as npz_file:
            feature_array = choose_feature_array(npz_file)
            label_array = choose_label_array(npz_file)

        if label_array is not None and feature_array.ndim >= 3 and label_array.shape[0] == feature_array.shape[0]:
            sequence_batch = coerce_sequence_batch(feature_array)
            sequence_batch = resize_sequence_batch(sequence_batch, sequence_length)
            if append_ear_mar and feature_array.ndim >= 4:
                handcrafted = build_handcrafted_features(normalize_landmark_batch(feature_array))
                if personalize_handcrafted:
                    handcrafted = personalize_handcrafted_features(handcrafted, file_path, user_baselines)
                handcrafted = resize_sequence_batch(handcrafted, sequence_length)
                if handcrafted_only:
                    sequence_batch = handcrafted.astype(np.float32)
                else:
                    sequence_batch = np.concatenate([sequence_batch, handcrafted], axis=-1).astype(np.float32)
            features.extend(sequence_batch)
            labels.extend(label_array.astype(np.int64).tolist())
            continue

        sequence = coerce_sequence(feature_array)
        sequence = resize_sequence(sequence, sequence_length)
        features.append(sequence)
        labels.append(label_id)

    return np.stack(features).astype(np.float32), np.array(labels, dtype=np.int64)


def compute_normalization_stats(train_x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flat = train_x.reshape(-1, train_x.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_sequences(data_x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((data_x - mean) / std).astype(np.float32)


def summarize_labels(labels: np.ndarray, label_map: Dict[str, int]) -> Dict[str, int]:
    inverse_map = {index: name for name, index in label_map.items()}
    counts = np.bincount(labels, minlength=len(label_map))
    return {inverse_map[index]: int(counts[index]) for index in range(len(label_map))}


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, num_classes: int):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        pooled = outputs.mean(dim=1)
        return self.classifier(pooled)


def create_dataloader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool):
    dataset = SequenceDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_eval_criterion() -> nn.Module:
    return nn.CrossEntropyLoss()


def evaluate_model(model: nn.Module, data_loader, device: torch.device) -> Dict[str, object]:
    model.eval()
    losses = []
    all_logits = []
    all_labels = []
    criterion = build_eval_criterion()

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            losses.append(loss.item())
            all_logits.append(logits.cpu())
            all_labels.append(batch_y.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    predictions = torch.argmax(logits, dim=1).numpy()
    average = "binary" if len(np.unique(labels)) == 2 else "macro"

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average=average, zero_division=0)),
        "recall": float(recall_score(labels, predictions, average=average, zero_division=0)),
        "f1": float(f1_score(labels, predictions, average=average, zero_division=0)),
        "per_class_precision": precision_score(labels, predictions, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(labels, predictions, average=None, zero_division=0).tolist(),
        "per_class_f1": f1_score(labels, predictions, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def collect_predictions(model: nn.Module, data_loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            all_logits.append(logits.cpu())
            all_labels.append(batch_y.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = np.argmax(probabilities, axis=1)
    return labels, probabilities, predictions


def build_metrics_from_predictions(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, object]:
    average = "binary" if len(np.unique(labels)) == 2 else "macro"
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average=average, zero_division=0)),
        "recall": float(recall_score(labels, predictions, average=average, zero_division=0)),
        "f1": float(f1_score(labels, predictions, average=average, zero_division=0)),
        "per_class_precision": precision_score(labels, predictions, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(labels, predictions, average=None, zero_division=0).tolist(),
        "per_class_f1": f1_score(labels, predictions, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def optimize_binary_threshold(labels: np.ndarray, probabilities: np.ndarray) -> Tuple[float, Dict[str, object]]:
    return optimize_binary_threshold_for_metric(labels, probabilities, "f1")


def optimize_binary_threshold_for_metric(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metric_name: str,
) -> Tuple[float, Dict[str, object]]:
    best_threshold = 0.5
    best_metrics = build_metrics_from_predictions(labels, (probabilities[:, 1] >= 0.5).astype(np.int64))
    best_score = best_metrics[metric_name]

    for threshold in np.arange(0.05, 0.96, 0.01):
        predictions = (probabilities[:, 1] >= threshold).astype(np.int64)
        metrics = build_metrics_from_predictions(labels, predictions)
        if metrics[metric_name] > best_score:
            best_score = metrics[metric_name]
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def train_one_epoch(model, data_loader, optimizer, criterion, device, train_noise_std: float) -> float:
    model.train()
    epoch_losses = []
    for batch_x, batch_y in tqdm(data_loader, desc="Training", leave=False):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if train_noise_std > 0.0:
            batch_x = batch_x + (torch.randn_like(batch_x) * train_noise_std)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_losses.append(loss.item())
    return float(np.mean(epoch_losses))


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def get_monitor_value(metrics: Dict[str, object], metric_name: str) -> float:
    if metric_name == "loss":
        return float(metrics["loss"])
    if metric_name == "f1":
        return float(metrics["f1"])
    if metric_name == "accuracy":
        return float(metrics["accuracy"])
    raise ValueError(f"Unsupported monitor metric: {metric_name}")


def is_better_metric(current: float, best: float, metric_name: str) -> bool:
    if metric_name == "loss":
        return current < best
    return current > best


def get_current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def plot_training_history(history: List[Dict[str, object]], output_path: Path) -> None:
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val"]["loss"] for entry in history]
    val_accuracy = [entry["val"]["accuracy"] * 100.0 for entry in history]
    val_f1 = [entry["val"]["f1"] * 100.0 for entry in history]

    figure, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Val Loss")
    axes[0].set_title("Loss by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, val_accuracy, marker="o", label="Val Accuracy")
    axes[1].set_title("Validation Accuracy by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, val_f1, marker="o", label="Val F1")
    axes[2].set_title("Validation F1 by Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 (%)")
    axes[2].set_ylim(0.0, 100.0)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_map = discover_labels(args.data_root)
    train_x, train_y = load_split(
        args.data_root / "train",
        label_map,
        args.sequence_length,
        args.append_ear_mar,
        args.personalize_handcrafted,
        args.handcrafted_only,
    )
    val_x, val_y = load_split(
        args.data_root / "val",
        label_map,
        args.sequence_length,
        args.append_ear_mar,
        args.personalize_handcrafted,
        args.handcrafted_only,
    )
    test_x, test_y = load_split(
        args.data_root / "test",
        label_map,
        args.sequence_length,
        args.append_ear_mar,
        args.personalize_handcrafted,
        args.handcrafted_only,
    )

    train_distribution = summarize_labels(train_y, label_map)
    val_distribution = summarize_labels(val_y, label_map)
    test_distribution = summarize_labels(test_y, label_map)

    print(f"Train/Val/Test samples: {len(train_y)}/{len(val_y)}/{len(test_y)}")
    print(f"Sequence shape: {train_x.shape[1:]}")
    print(f"Label map: {label_map}")
    print(f"Train distribution: {train_distribution}")
    print(f"Val distribution: {val_distribution}")
    print(f"Test distribution: {test_distribution}")

    mean, std = compute_normalization_stats(train_x)
    np.savez(args.output_dir / "scaler_stats.npz", mean=mean, std=std)

    train_x = normalize_sequences(train_x, mean, std)
    val_x = normalize_sequences(val_x, mean, std)
    test_x = normalize_sequences(test_x, mean, std)

    train_loader = create_dataloader(train_x, train_y, args.batch_size, shuffle=True)
    val_loader = create_dataloader(val_x, val_y, args.batch_size, shuffle=False)
    test_loader = create_dataloader(test_x, test_y, args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(
        input_size=train_x.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=len(label_map),
    ).to(device)

    if args.disable_class_weights:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        class_counts = np.bincount(train_y, minlength=len(label_map))
        class_weights = len(train_y) / np.maximum(class_counts * len(label_map), 1.0)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
            label_smoothing=args.label_smoothing,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if args.early_stopping_metric == "loss" else "max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_learning_rate,
    )

    best_monitor_value = float("inf") if args.early_stopping_metric == "loss" else float("-inf")
    best_epoch = 0
    stale_epochs = 0
    best_model_path = args.output_dir / "best_model.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.train_noise_std)
        val_metrics = evaluate_model(model, val_loader, device)
        current_lr = get_current_learning_rate(optimizer)
        history.append({"epoch": epoch, "train_loss": train_loss, "learning_rate": current_lr, "val": val_metrics})
        print(
            f"Epoch {epoch:02d}/{args.epochs} | lr={current_lr:.6f} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_f1={val_metrics['f1']:.4f} | val_acc={val_metrics['accuracy']:.4f}"
        )

        monitor_value = get_monitor_value(val_metrics, args.early_stopping_metric)
        if is_better_metric(monitor_value, best_monitor_value, args.early_stopping_metric):
            best_monitor_value = monitor_value
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            stale_epochs += 1

        scheduler.step(monitor_value)

        if stale_epochs >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch:02d}. "
                f"Best epoch was {best_epoch:02d} using val_{args.early_stopping_metric}={best_monitor_value:.4f}"
            )
            break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    val_metrics = evaluate_model(model, val_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)

    threshold_summary = None
    if args.optimize_threshold and len(label_map) == 2:
        val_labels, val_probabilities, _ = collect_predictions(model, val_loader, device)
        best_threshold, threshold_val_metrics = optimize_binary_threshold_for_metric(
            val_labels,
            val_probabilities,
            args.threshold_metric,
        )
        test_labels, test_probabilities, _ = collect_predictions(model, test_loader, device)
        test_threshold_predictions = (test_probabilities[:, 1] >= best_threshold).astype(np.int64)
        threshold_test_metrics = build_metrics_from_predictions(test_labels, test_threshold_predictions)
        threshold_summary = {
            "metric": args.threshold_metric,
            "best_threshold": best_threshold,
            "validation": threshold_val_metrics,
            "test": threshold_test_metrics,
        }

    save_json(
        args.output_dir / "metrics.json",
        {
            "train_sample_count": int(len(train_y)),
            "val_sample_count": int(len(val_y)),
            "test_sample_count": int(len(test_y)),
            "feature_dim": int(train_x.shape[-1]),
            "sequence_length": int(train_x.shape[1]),
            "label_map": label_map,
            "train_distribution": train_distribution,
            "val_distribution": val_distribution,
            "test_distribution": test_distribution,
            "best_epoch": int(best_epoch),
            "early_stopping_metric": args.early_stopping_metric,
            "best_monitor_value": float(best_monitor_value),
            "validation": val_metrics,
            "test": test_metrics,
            "threshold_optimization": threshold_summary,
            "history": history,
        },
    )
    plot_training_history(history, args.output_dir / "training_curves.png")
    save_json(args.output_dir / "label_map.json", label_map)

    print("Best validation metrics:")
    print(json.dumps(val_metrics, indent=2))
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))
    if threshold_summary is not None:
        print("Threshold-optimized metrics:")
        print(json.dumps(threshold_summary, indent=2))
    print(f"Best epoch: {best_epoch}")
    print(f"Selection metric: val_{args.early_stopping_metric}")
    print(f"Saved model to: {best_model_path}")


if __name__ == "__main__":
    main()
