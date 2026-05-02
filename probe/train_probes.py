"""
Train one linear probe per (codec, RVQ layer, task) on frozen token embeddings.

Tasks:
  phoneme -> LogisticRegression  (class_weight="balanced" for skewed phoneme dist.)
  speaker -> LogisticRegression
  pitch   -> LinearRegression    (voiced frames only; unvoiced tokens are NaN)

Codec parameters are never updated - embeddings are treated as fixed features.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

NUM_LAYERS = 8
PROBE_TASKS = ("phoneme", "speaker", "pitch")
DEFAULT_CLASSIFICATION_MAX_ITER = 1000


@dataclass(frozen=True)
class ProbeTrainingBundle:
    """All probe-training inputs for one codec."""

    codec_name: str
    embeddings_by_layer: List[np.ndarray]
    phoneme_labels: np.ndarray
    speaker_labels: np.ndarray
    pitch_values: np.ndarray


@dataclass(frozen=True)
class ProbeJob:
    """One independent probe fitting job."""

    codec_name: str
    layer_num: int
    task: str
    X: np.ndarray
    y: np.ndarray
    output_path: Path
    voiced_mask: Optional[np.ndarray] = None
    classification_max_iter: int = DEFAULT_CLASSIFICATION_MAX_ITER


class ProbeTrainingError(RuntimeError):
    """Raised when one or more probe jobs fail."""


# -- Label encoder fitting -------------------------------------------------------


def fit_label_encoders(
    phoneme_labels: np.ndarray,
    speaker_labels: np.ndarray,
    probe_dir: Path,
) -> Dict[str, LabelEncoder]:
    """
    Fit LabelEncoders for phoneme and speaker on training data and save to disk.

    Must be called once before train_probes or evaluate_probes so that both
    codecs and both splits share identical integer mappings.
    """
    probe_dir.mkdir(parents=True, exist_ok=True)
    encoders: Dict[str, LabelEncoder] = {}
    for task, labels in [("phoneme", phoneme_labels), ("speaker", speaker_labels)]:
        le = LabelEncoder()
        le.fit(labels)
        encoders[task] = le
        with open(probe_dir / f"label_encoder_{task}.pkl", "wb") as f:
            pickle.dump(le, f)
    return encoders


# -- Probe fitting helpers -------------------------------------------------------


def _fit_classification_probe(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = DEFAULT_CLASSIFICATION_MAX_ITER,
) -> LogisticRegression:
    # class_weight="balanced" compensates for skewed phoneme/speaker distributions.
    clf = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        class_weight="balanced",
    )
    clf.fit(X, y)
    return clf


def _fit_regression_probe(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    reg = LinearRegression()
    reg.fit(X, y)
    return reg


def _load_label_encoders(probe_dir: Path) -> Dict[str, LabelEncoder]:
    with open(probe_dir / "label_encoder_phoneme.pkl", "rb") as f:
        phoneme = pickle.load(f)
    with open(probe_dir / "label_encoder_speaker.pkl", "rb") as f:
        speaker = pickle.load(f)
    return {"phoneme": phoneme, "speaker": speaker}


def _normalize_worker_count(max_workers: int) -> int:
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    return max_workers


def _resolve_blas_threads(max_workers: int, blas_threads: int) -> int:
    if blas_threads < 0:
        raise ValueError("blas_threads must be >= 0")
    if blas_threads > 0:
        return blas_threads
    # Auto mode: leave single-worker runs unconstrained, but cap nested BLAS
    # threads for parallel probe jobs to avoid severe oversubscription.
    return 0 if max_workers == 1 else 1


def _threadpool_limit_context(blas_threads: int):
    if blas_threads <= 0:
        return nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        print(
            "threadpoolctl unavailable; proceeding without BLAS thread limits. "
            "Install threadpoolctl for more predictable parallel speedups."
        )
        return nullcontext()

    print(f"Limiting BLAS/OpenMP threads per worker to {blas_threads}.")
    return threadpool_limits(limits=blas_threads)


def _build_probe_jobs(
    bundle: ProbeTrainingBundle,
    output_dir: Path,
    label_encoders: Dict[str, LabelEncoder],
    classification_max_iter: int,
) -> List[ProbeJob]:
    if len(bundle.embeddings_by_layer) != NUM_LAYERS:
        raise ValueError(
            f"Expected {NUM_LAYERS} embedding layers for {bundle.codec_name}, "
            f"got {len(bundle.embeddings_by_layer)}"
        )

    phoneme_enc = np.asarray(label_encoders["phoneme"].transform(bundle.phoneme_labels))
    speaker_enc = np.asarray(label_encoders["speaker"].transform(bundle.speaker_labels))
    voiced_mask = ~np.isnan(bundle.pitch_values)
    task_specs: Tuple[Tuple[str, np.ndarray, Optional[np.ndarray]], ...] = (
        ("phoneme", phoneme_enc, None),
        ("speaker", speaker_enc, None),
        ("pitch", bundle.pitch_values, voiced_mask),
    )

    jobs: List[ProbeJob] = []
    for layer_idx in range(NUM_LAYERS):
        X = bundle.embeddings_by_layer[layer_idx]
        layer_num = layer_idx + 1

        for task, target_y, task_voiced_mask in task_specs:
            jobs.append(
                ProbeJob(
                    codec_name=bundle.codec_name,
                    layer_num=layer_num,
                    task=task,
                    X=X,
                    y=target_y,
                    voiced_mask=task_voiced_mask,
                    output_path=output_dir / f"probe_{bundle.codec_name}_layer{layer_num}_{task}.pkl",
                    classification_max_iter=classification_max_iter,
                )
            )

    return jobs


def _build_jobs_for_bundles(
    bundles: Sequence[ProbeTrainingBundle],
    output_dir: Path,
    label_encoders: Dict[str, LabelEncoder],
    classification_max_iter: int,
) -> List[ProbeJob]:
    jobs: List[ProbeJob] = []
    for bundle in bundles:
        jobs.extend(
            _build_probe_jobs(
                bundle,
                output_dir,
                label_encoders,
                classification_max_iter=classification_max_iter,
            )
        )
    return jobs


def _job_label(job: ProbeJob) -> str:
    return f"[{job.codec_name}] layer {job.layer_num}/{NUM_LAYERS} {job.task}"


def _run_probe_job(job: ProbeJob) -> str:
    if job.task not in PROBE_TASKS:
        raise ValueError(f"Unsupported probe task: {job.task}")

    if job.task in ("phoneme", "speaker"):
        probe = _fit_classification_probe(job.X, job.y, max_iter=job.classification_max_iter)
    else:
        if job.voiced_mask is None:
            raise ValueError("Pitch jobs require voiced_mask")
        probe = (
            _fit_regression_probe(job.X[job.voiced_mask], job.y[job.voiced_mask])
            if job.voiced_mask.sum() > 0
            else None
        )

    with open(job.output_path, "wb") as f:
        pickle.dump(probe, f)

    return _job_label(job)


def _format_failure(job: ProbeJob, exc: BaseException) -> str:
    return f"{_job_label(job)} -> {type(exc).__name__}: {exc}"


def _iter_job_outcomes(
    jobs: Sequence[ProbeJob],
    worker_count: int,
) -> Iterator[Tuple[ProbeJob, Optional[str], Optional[BaseException]]]:
    if worker_count == 1:
        for job in jobs:
            try:
                yield job, _run_probe_job(job), None
            except Exception as exc:  # pragma: no cover - tested via failure aggregation
                yield job, None, exc
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_job = {executor.submit(_run_probe_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                yield job, future.result(), None
            except Exception as exc:
                yield job, None, exc


def _run_probe_jobs(jobs: Sequence[ProbeJob], max_workers: int) -> None:
    worker_count = _normalize_worker_count(max_workers)
    total_jobs = len(jobs)
    failures: List[Tuple[ProbeJob, BaseException]] = []

    for completed, (job, label, error) in enumerate(
        _iter_job_outcomes(jobs, worker_count),
        start=1,
    ):
        if error is None:
            print(f"  ({completed}/{total_jobs}) {label}")
            continue
        failures.append((job, error))
        print(f"  ({completed}/{total_jobs}) {_job_label(job)} failed")

    if failures:
        preview_count = 5
        preview = [_format_failure(job, exc) for job, exc in failures[:preview_count]]
        if len(failures) > preview_count:
            preview.append(f"... {len(failures) - preview_count} additional failures")
        raise ProbeTrainingError(
            f"{len(failures)} probe jobs failed out of {total_jobs}: "
            + "; ".join(preview)
        )


def _split_pending_jobs(jobs: Sequence[ProbeJob]) -> Tuple[List[ProbeJob], int]:
    pending: List[ProbeJob] = []
    skipped = 0
    for job in jobs:
        if job.output_path.exists():
            skipped += 1
        else:
            pending.append(job)
    return pending, skipped


def train_probes_for_codecs(
    bundles: Sequence[ProbeTrainingBundle],
    output_dir: str,
    label_encoders: Optional[Dict[str, LabelEncoder]] = None,
    max_workers: int = 1,
    blas_threads: int = 0,
    classification_max_iter: int = DEFAULT_CLASSIFICATION_MAX_ITER,
) -> None:
    """
    Train probes across one or more codecs with a shared job dispatcher.

    Args:
        bundles:      One ProbeTrainingBundle per codec.
        output_dir:   Directory for saved .pkl probe files.
        label_encoders:
            Pre-fitted encoders from fit_label_encoders().
            If None, loads from output_dir (backward compatibility).
        max_workers:  Maximum concurrent probe jobs. 1 = sequential.
        blas_threads:
            Maximum BLAS/OpenMP threads used per worker during probe fitting.
            0 = auto (1 when max_workers > 1, otherwise leave unconstrained).
        classification_max_iter:
            Max iterations for classification probes (phoneme/speaker).
    """
    if not bundles:
        raise ValueError("At least one ProbeTrainingBundle is required")
    if classification_max_iter < 1:
        raise ValueError("classification_max_iter must be >= 1")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    encoders = label_encoders if label_encoders is not None else _load_label_encoders(out)
    jobs = _build_jobs_for_bundles(
        bundles,
        out,
        encoders,
        classification_max_iter=classification_max_iter,
    )
    pending_jobs, skipped_jobs = _split_pending_jobs(jobs)
    total_jobs = len(jobs)

    if skipped_jobs:
        print(f"Skipping {skipped_jobs}/{total_jobs} probe jobs with existing artifacts.")
    if not pending_jobs:
        print("All probe artifacts already exist; skipping probe training.")
        return

    print(
        f"Training {len(pending_jobs)} probe jobs across {len(bundles)} codec(s) "
        f"with {max_workers} worker(s)..."
    )

    resolved_blas_threads = _resolve_blas_threads(max_workers=max_workers, blas_threads=blas_threads)
    with _threadpool_limit_context(resolved_blas_threads):
        _run_probe_jobs(pending_jobs, max_workers=max_workers)


# -- Main training function ------------------------------------------------------


def train_probes(
    embeddings_by_layer: List[np.ndarray],
    phoneme_labels: np.ndarray,
    speaker_labels: np.ndarray,
    pitch_values: np.ndarray,
    codec_name: str,
    output_dir: str,
    label_encoders: Optional[Dict[str, LabelEncoder]] = None,
    max_workers: int = 1,
    blas_threads: int = 0,
    classification_max_iter: int = DEFAULT_CLASSIFICATION_MAX_ITER,
) -> None:
    """
    Train and save 3 probes x 8 layers = 24 probes for one codec.

    Args:
        embeddings_by_layer: List of 8 arrays, shape (N_tokens, embed_dim).
        phoneme_labels:      String phoneme labels, shape (N_tokens,).
        speaker_labels:      String speaker IDs, shape (N_tokens,).
        pitch_values:        Float Hz values, shape (N_tokens,); NaN = unvoiced.
        codec_name:          "encodec" or "speechtokenizer" - used in filenames.
        output_dir:          Directory for saved .pkl probe files.
        label_encoders:      Pre-fitted encoders from fit_label_encoders().
                             If None, loads from output_dir (backward compatibility).
        max_workers:         Maximum concurrent probe jobs. 1 = sequential.
        blas_threads:        Maximum BLAS/OpenMP threads per worker (0 = auto).
        classification_max_iter:
            Max iterations for classification probes (phoneme/speaker).
    """
    bundle = ProbeTrainingBundle(
        codec_name=codec_name,
        embeddings_by_layer=embeddings_by_layer,
        phoneme_labels=phoneme_labels,
        speaker_labels=speaker_labels,
        pitch_values=pitch_values,
    )
    train_probes_for_codecs(
        bundles=[bundle],
        output_dir=output_dir,
        label_encoders=label_encoders,
        max_workers=max_workers,
        blas_threads=blas_threads,
        classification_max_iter=classification_max_iter,
    )
