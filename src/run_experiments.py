from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib

# Use headless backend for environments without display
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Supported metrics mapping to scipy.spatial.distance.cdist names
SUPPORTED = {
    "cosine": "cosine",          # distance = 1 - cosine_similarity
    "euclidean": "euclidean",    # L2
    "manhattan": "cityblock",    # L1
    "cityblock": "cityblock",    # alias
    "chebyshev": "chebyshev",
    "braycurtis": "braycurtis",
}

DEFAULT_EMBEDDINGS_PATH = "../data/embeddings/embeddings_mobilenet_v2.npy"


def load_embeddings(emb_path: str) -> Tuple[np.ndarray, List[str]]:
    if not os.path.isfile(emb_path):
        raise FileNotFoundError(f"Embeddings .npy non trovato: {emb_path}")
    X = np.load(emb_path)
    paths_txt = os.path.splitext(emb_path)[0] + ".paths.txt"
    if not os.path.isfile(paths_txt):
        raise FileNotFoundError(
            f"File paths non trovato: {paths_txt}. Riesegui embed_crop.py con --save_paths."
        )
    with open(paths_txt, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]
    if X.shape[0] != len(paths):
        raise ValueError(
            f"Mismatch tra numero di embeddings ({X.shape[0]}) e paths ({len(paths)})"
        )
    return X.astype(np.float32, copy=False), paths


def derive_labels(paths: Sequence[str], mode: str = "dirname") -> List[str]:
    labels: List[str] = []
    if mode == "dirname":
        for p in paths:
            labels.append(os.path.basename(os.path.dirname(p)))
    elif mode == "filename_prefix":
        # Esempio: starry_night_1_crop.png -> 'starry_night'
        for p in paths:
            base = os.path.basename(p)
            low = base.lower()
            lbl = low
            if "_crop" in low:
                lbl = low.split("_crop")[0]
            # Rimuovi suffisso numerico finale se presente (es. _1)
            parts = lbl.split("_")
            if parts and parts[-1].isdigit():
                lbl = "_".join(parts[:-1])
            labels.append(lbl)
    else:
        raise ValueError(f"label_mode non supportato: {mode}")
    return labels


def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n


def loo_splits(n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        ref_mask = np.ones(n, dtype=bool)
        ref_mask[i] = False
        qry_mask = ~ref_mask
        splits.append((idx[ref_mask], idx[qry_mask]))
    return splits


def top2_distances(D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (d1, d2) the smallest and second smallest distances per row.

    Ensures d1 <= d2 for each query by partitioning and sorting the two smallest.
    """
    if D.shape[1] < 2:
        raise ValueError("Servono almeno 2 elementi nel reference set per il ratio test (2-NN)")
    # Partition to get two smallest in first two positions, then sort those two
    first_two = np.partition(D, 1, axis=1)[:, :2]
    first_two.sort(axis=1)  # in-place row-wise sort
    d1 = first_two[:, 0]
    d2 = first_two[:, 1]
    return d1, d2


def evaluate_ratio_test(
    y_true: Sequence[str],
    nn_labels: Sequence[str],
    d1: np.ndarray,
    d2: np.ndarray,
    taus: np.ndarray,
) -> Dict[float, Dict[str, float]]:
    results: Dict[float, Dict[str, float]] = {}
    y_true = list(y_true)
    nn_labels = list(nn_labels)
    total = len(y_true)
    for tau in taus:
        accepted = d1 / (d2 + 1e-12) < float(tau)
        # Build predictions with possible no-match
        y_pred: List[str] = [
            (lbl if acc else "__no_match__") for acc, lbl in zip(accepted, nn_labels)
        ]
        matches_idx = [i for i, yp in enumerate(y_pred) if yp != "__no_match__"]
        correct = sum(1 for i in matches_idx if y_pred[i] == y_true[i])
        precision = correct / (len(matches_idx) + 1e-12)
        recall = correct / (total + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        accuracy = sum(1 for i in range(total) if y_pred[i] == y_true[i]) / (total + 1e-12)
        coverage = len(matches_idx) / (total + 1e-12)
        results[float(tau)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "coverage": coverage,
        }
    return results


def plot_curve(taus: Sequence[float], results: Dict[float, Dict[str, float]], out_png: str, title: str) -> None:
    order = sorted(results.keys())
    f1 = [results[t]["f1"] for t in order]
    prec = [results[t]["precision"] for t in order]
    rec = [results[t]["recall"] for t in order]
    cov = [results[t]["coverage"] for t in order]

    plt.figure(figsize=(7, 4))
    plt.plot(order, f1, label="F1")
    plt.plot(order, prec, label="Precision")
    plt.plot(order, rec, label="Recall")
    plt.plot(order, cov, label="Coverage")
    plt.xlabel("tau (d1/d2)")
    plt.ylabel("score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def save_results_csv(results: Dict[float, Dict[str, float]], out_csv: str) -> None:
    """Save per-metric results to CSV with columns: tau, precision, recall, f1, accuracy, coverage."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    order = sorted(results.keys())
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("tau,precision,recall,f1,accuracy,coverage\n")
        for t in order:
            m = results[t]
            f.write(
                f"{float(t):.6f},{m['precision']:.6f},{m['recall']:.6f},{m['f1']:.6f},{m['accuracy']:.6f},{m['coverage']:.6f}\n"
            )


def run_metric(
    X: np.ndarray,
    paths: Sequence[str],
    labels: Sequence[str],
    metric_key: str,
    normalize: bool,
    taus: np.ndarray,
) -> Tuple[Dict[float, Dict[str, float]], Dict[str, float]]:
    splits = loo_splits(len(paths))

    all_true: List[str] = []
    all_nn_labels: List[str] = []
    all_d1: List[np.ndarray] = []
    all_d2: List[np.ndarray] = []

    for ref_idx, qry_idx in splits:
        X_ref = X[ref_idx]
        X_qry = X[qry_idx]
        if normalize:
            X_ref = l2_normalize(X_ref)
            X_qry = l2_normalize(X_qry)
        D = cdist(X_qry, X_ref, metric=SUPPORTED[metric_key])  # shape (1, n_ref) in LOO

        # 1-NN labels
        nn_idx = np.argmin(D, axis=1)
        y_ref = [labels[i] for i in ref_idx]
        y_qry = [labels[i] for i in qry_idx]
        nn_labels = [y_ref[j] for j in nn_idx]

        d1, d2 = top2_distances(D)

        all_true.extend(y_qry)
        all_nn_labels.extend(nn_labels)
        all_d1.append(d1)
        all_d2.append(d2)

    all_d1_arr = np.concatenate(all_d1, axis=0)
    all_d2_arr = np.concatenate(all_d2, axis=0)

    results = evaluate_ratio_test(all_true, all_nn_labels, all_d1_arr, all_d2_arr, taus)
    # Best by F1
    best_tau, best_metrics = max(results.items(), key=lambda kv: kv[1]["f1"])  # type: ignore[index]
    best = {**best_metrics, "best_tau": float(best_tau)}
    return results, best


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sperimenta metriche + 2-NN ratio test su embeddings (MobileNetV2)")
    p.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS_PATH, help="Percorso a embeddings .npy (default: ../data/embeddings/embeddings_mobilenet_v2.npy)")
    p.add_argument("--label_mode", choices=["dirname", "filename_prefix"], default="dirname", help="Come derivare le etichette dai path")
    p.add_argument("--metrics", default="cosine,euclidean,manhattan", help="Metriche da testare separate da virgola")
    p.add_argument("--normalize", action="store_true", help="Applica L2-normalizzazione ai vettori prima della distanza")
    p.add_argument("--tau_min", type=float, default=0.6)
    p.add_argument("--tau_max", type=float, default=0.95)
    p.add_argument("--tau_steps", type=int, default=20)
    p.add_argument("--out_dir", default="../experiments_out", help="Cartella di output: verranno create le sottocartelle 'plots/' (grafici) e 'csv_files/' (CSV)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        X, paths = load_embeddings(args.embeddings)
    except Exception as e:
        print(f"[ERRORE] {e}", file=sys.stderr)
        return 2

    if X.shape[0] < 3:
        print("[ERRORE] Servono almeno 3 embeddings per eseguire il ratio test con LOO (ogni query deve avere almeno 2 riferimenti).", file=sys.stderr)
        return 2

    labels = derive_labels(paths, mode=args.label_mode)

    taus = np.linspace(float(args.tau_min), float(args.tau_max), int(args.tau_steps), dtype=np.float32)

    metrics = [m.strip().lower() for m in (args.metrics or "").split(",") if m.strip()]
    if not metrics:
        print("[ERRORE] Nessuna metrica specificata.", file=sys.stderr)
        return 2

    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
    # Define subdirectories for plots and CSV files
    plots_dir = os.path.join(args.out_dir, "plots")
    csv_dir = os.path.join(args.out_dir, "csv_files")
    os.makedirs(os.path.abspath(plots_dir), exist_ok=True)
    os.makedirs(os.path.abspath(csv_dir), exist_ok=True)

    summary: List[Tuple[str, float, Dict[str, float]]] = []

    for m in metrics:
        if m not in SUPPORTED:
            print(f"[WARN] metrica '{m}' non supportata, salto.", file=sys.stderr)
            continue
        print(f"[INFO] Eseguo metrica={m} | normalize={args.normalize}")
        results, best = run_metric(X, paths, labels, metric_key=m, normalize=bool(args.normalize), taus=taus)

        # Salva curva
        plot_path = os.path.join(plots_dir, f"curve_{m}.png")
        plot_curve(taus, results, plot_path, title=f"Ratio test curve - {m}")
        print(f"[OK] Salvato grafico: {plot_path}")

        # Salva CSV risultati completi per metrica
        results_csv = os.path.join(csv_dir, f"results_{m}.csv")
        save_results_csv(results, results_csv)
        print(f"[OK] Salvato CSV risultati: {results_csv}")

        print(
            f"[BEST] metric={m} best_tau={best['best_tau']:.3f} F1={best['f1']:.3f} "
            f"P={best['precision']:.3f} R={best['recall']:.3f} Acc={best['accuracy']:.3f} Cov={best['coverage']:.3f}"
        )
        summary.append((m, float(best["best_tau"]), best))

    # Salva riepilogo migliori per metrica
    if summary:
        best_csv = os.path.join(csv_dir, "best_summary.csv")
        with open(best_csv, "w", encoding="utf-8") as f:
            f.write("metric,best_tau,precision,recall,f1,accuracy,coverage\n")
            for m, tau, metr in summary:
                f.write(
                    f"{m},{tau:.6f},{metr['precision']:.6f},{metr['recall']:.6f},{metr['f1']:.6f},{metr['accuracy']:.6f},{metr['coverage']:.6f}\n"
                )
        print(f"[OK] Salvato best_summary: {best_csv}")

    print("\n=== RIEPILOGO ===")
    if summary:
        best_global = max(summary, key=lambda t: t[2]["f1"])  # type: ignore[index]
        m, tau, metr = best_global
        print(
            f"Metrica migliore: {m} | tau={tau:.3f} | F1={metr['f1']:.3f} "
            f"P={metr['precision']:.3f} R={metr['recall']:.3f} Acc={metr['accuracy']:.3f} Cov={metr['coverage']:.3f}"
        )
    else:
        print("Nessun risultato (metrica non valida o problemi negli input)")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
