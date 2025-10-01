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
    "cosine": "cosine",         
    "euclidean": "euclidean",    # L2
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

# ===== Helpers per esperimento Top-K =====
from collections import Counter
from typing import Dict as _Dict, List as _List

def topk_indices_from_scores(S: np.ndarray, K: int, larger_is_better: bool) -> np.ndarray:
    """Restituisce indici dei top-K per riga.
    Se larger_is_better=True (es. similarità coseno), seleziona i K più grandi; altrimenti i K più piccoli.
    Usa argpartition per efficienza; l'ordinamento all'interno dei K verrà raffinato a parte.
    """
    if K <= 0:
        raise ValueError("K deve essere >= 1")
    if S.shape[1] < K:
        # Se K supera il riferimento, limita a tutto il set
        K = S.shape[1]
    if larger_is_better:
        return np.argpartition(-S, K - 1, axis=1)[:, :K]
    else:
        return np.argpartition(S, K - 1, axis=1)[:, :K]


def refine_sorted_topk(S: np.ndarray, topk_idx: np.ndarray, larger_is_better: bool) -> np.ndarray:
    """Ordina correttamente i top-K selezionati per riga, mantenendo solo i K migliori in ordine."""
    rows = np.arange(S.shape[0])[:, None]
    vals = S[rows, topk_idx]
    order_in_k = np.argsort(-vals if larger_is_better else vals, axis=1)
    return topk_idx[rows, order_in_k]


def majority_vote(labels_topk: _List[str], weights: _List[float] | None = None, prefer_max: bool = True) -> str:
    """Voto di maggioranza sui top-K.
    - Se weights è None: puro conteggio delle etichette; pareggio rotto alfabeticamente.
    - Se weights è fornito: somma dei pesi per classe e scelta della classe con somma massima (prefer_max=True, tipico per similarità) o minima (prefer_max=False, tipico per distanze). In caso di pareggio, rottura alfabetica.
    """
    if weights is None:
        counter = Counter(labels_topk)
        max_count = max(counter.values())
        tied = [c for c, v in counter.items() if v == max_count]
        if len(tied) == 1:
            return tied[0]
        return sorted(tied)[0]
    else:
        groups: _Dict[str, float] = {}
        for lbl, w in zip(labels_topk, weights):
            groups[lbl] = groups.get(lbl, 0.0) + float(w)
        if prefer_max:
            best_val = max(groups.values())
            tied = [c for c, v in groups.items() if v == best_val]
        else:
            best_val = min(groups.values())
            tied = [c for c, v in groups.items() if v == best_val]
        if len(tied) == 1:
            return tied[0]
        return sorted(tied)[0]


def evaluate_ratio_test(
    y_true: Sequence[str],
    nn_labels: Sequence[str],
    d1: np.ndarray,
    d2: np.ndarray,
    ratios: np.ndarray,
) -> Dict[float, Dict[str, float]]:
    results: Dict[float, Dict[str, float]] = {}
    y_true = list(y_true)
    nn_labels = list(nn_labels)
    total = len(y_true)
    for ratio in ratios:
        accepted = d1 / (d2 + 1e-12) < float(ratio)
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
        results[float(ratio)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "coverage": coverage,
        }
    return results


def plot_curve(ratios: Sequence[float], results: Dict[float, Dict[str, float]], out_png: str, title: str) -> None:
    order = sorted(results.keys())
    f1 = [results[t]["f1"] for t in order]
    prec = [results[t]["precision"] for t in order]
    rec = [results[t]["recall"] for t in order]
    cov = [results[t]["coverage"] for t in order]

    # Individua il best_ratio (massimo F1)
    best_idx = int(np.argmax(f1)) if len(f1) > 0 else None
    best_tau = order[best_idx] if best_idx is not None else None
    best_f1 = f1[best_idx] if best_idx is not None else None

    plt.figure(figsize=(7, 4))
    # Plot F1 curve without per-point markers; only the best point will be shown as a dot.
    plt.plot(order, f1, label="F1", linewidth=1.5)
    # Other curves without markers to reduce clutter
    plt.plot(order, prec, label="Precision")
    plt.plot(order, rec, label="Recall")
    plt.plot(order, cov, label="Coverage")

    # Highlight and annotate only the best ratio (max F1)
    if best_tau is not None and best_f1 is not None:
        plt.scatter([best_tau], [best_f1], color="red", zorder=6)
        plt.annotate(f"{best_f1:.2f}", (best_tau, best_f1), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="red")

    plt.xlabel("ratio (d1/d2)")
    plt.ylabel("score")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def save_results_csv(results: Dict[float, Dict[str, float]], out_csv: str) -> None:
    """[DEPRECATO] Era usato per il ratio test. Lasciato per retrocompatibilità."""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    order = sorted(results.keys())
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("ratio,precision,recall,f1,accuracy,coverage\n")
        for t in order:
            m = results[t]
            f.write(
                f"{float(t):.6f},{m['precision']:.6f},{m['recall']:.6f},{m['f1']:.6f},{m['accuracy']:.6f},{m['coverage']:.6f}\n"
            )


def plot_metrics_bar(metric_names: Sequence[str], scores: Sequence[float], out_png: str, title: str, ylabel: str | None = None, bar_color: str | None = None) -> None:
    plt.figure(figsize=(7, 4))
    x = np.arange(len(metric_names))
    plt.bar(x, scores, color=(bar_color if bar_color is not None else "#69b3a2"))
    plt.xticks(x, metric_names)
    plt.ylim(0.0, 1.0)
    for xi, s in zip(x, scores):
        plt.text(xi, s + 0.01, f"{s:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel(ylabel if ylabel is not None else "accuracy")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.savefig(out_png)
    plt.close()


def save_metrics_summary_csv(summary: Dict[str, float], out_csv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("metric,accuracy\n")
        for m in summary:
            f.write(f"{m},{summary[m]:.6f}\n")


def save_scores_csv(scores: Dict[str, float], out_csv: str, score_name: str) -> None:
    """Salva un dizionario {metrica->valore} in CSV con header personalizzato.
    Esempio: score_name="p_at_3" produrrà colonne: metric,p_at_3
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(f"metric,{score_name}\n")
        for m, v in scores.items():
            f.write(f"{m},{float(v):.6f}\n")


def run_metric(
    X: np.ndarray,
    paths: Sequence[str],
    labels: Sequence[str],
    metric_key: str,
    normalize: bool,
) -> Dict[str, float]:
    """Run LOO 1-NN evaluation for a single metric (no ratio test).

    Returns a dict with aggregated metrics (currently accuracy only).
    """
    splits = loo_splits(len(paths))

    all_true: List[str] = []
    all_pred: List[str] = []

    for ref_idx, qry_idx in splits:
        X_ref = X[ref_idx]
        X_qry = X[qry_idx]
        if metric_key == "cosine":
            # Compute cosine similarity and select argmax
            if normalize:
                X_ref_n = l2_normalize(X_ref)
                X_qry_n = l2_normalize(X_qry)
            else:
                # Normalize only for cosine computation (does not affect euclidean path)
                X_ref_n = l2_normalize(X_ref)
                X_qry_n = l2_normalize(X_qry)
            S = np.dot(X_qry_n, X_ref_n.T)  # cosine similarity
            nn_idx = np.argmax(S, axis=1)
        else:
            if normalize:
                X_ref = l2_normalize(X_ref)
                X_qry = l2_normalize(X_qry)
            D = cdist(X_qry, X_ref, metric=SUPPORTED[metric_key])  # shape (1, n_ref) in LOO
            # 1-NN prediction (min distance)
            nn_idx = np.argmin(D, axis=1)
        y_ref = [labels[i] for i in ref_idx]
        y_qry = [labels[i] for i in qry_idx]
        nn_labels = [y_ref[j] for j in nn_idx]

        all_true.extend(y_qry)
        all_pred.extend(nn_labels)

    total = len(all_true)
    correct = sum(1 for i in range(total) if all_true[i] == all_pred[i])
    accuracy = correct / (total + 1e-12)
    return {"accuracy": float(accuracy)}


def collect_ratio_data(
    X: np.ndarray,
    paths: Sequence[str],
    labels: Sequence[str],
    metric_key: str,
    normalize: bool,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """Collect y_true, 1-NN labels, and top-2 distances (d1, d2) across LOO splits."""
    splits = loo_splits(len(paths))
    all_true: List[str] = []
    all_nn_labels: List[str] = []
    d1_list: List[float] = []
    d2_list: List[float] = []

    for ref_idx, qry_idx in splits:
        X_ref = X[ref_idx]
        X_qry = X[qry_idx]
        if metric_key == "cosine":
            # Cosine similarity for NN; convert to distances for ratio extraction
            if normalize:
                X_ref_n = l2_normalize(X_ref)
                X_qry_n = l2_normalize(X_qry)
            else:
                X_ref_n = l2_normalize(X_ref)
                X_qry_n = l2_normalize(X_qry)
            S = np.dot(X_qry_n, X_ref_n.T)  # (1, n_ref)
            D = 1.0 - S  # cosine distance derived from similarity, only for ratio d1/d2
            # top-2 distances
            d1, d2 = top2_distances(D)
            d1_list.extend(d1.tolist())
            d2_list.extend(d2.tolist())
            # 1-NN labels from maximum similarity
            nn_idx = np.argmax(S, axis=1)
        else:
            if normalize:
                X_ref = l2_normalize(X_ref)
                X_qry = l2_normalize(X_qry)
            D = cdist(X_qry, X_ref, metric=SUPPORTED[metric_key])  # (1, n_ref)
            # top-2 distances
            d1, d2 = top2_distances(D)
            d1_list.extend(d1.tolist())
            d2_list.extend(d2.tolist())
            # 1-NN labels (min distance)
            nn_idx = np.argmin(D, axis=1)
        y_ref = [labels[i] for i in ref_idx]
        y_qry = [labels[i] for i in qry_idx]
        nn_labels = [y_ref[j] for j in nn_idx]

        all_true.extend(y_qry)
        all_nn_labels.extend(nn_labels)

    return all_true, all_nn_labels, np.asarray(d1_list, dtype=np.float32), np.asarray(d2_list, dtype=np.float32)


def run_metric_topk(
    X: np.ndarray,
    paths: Sequence[str],
    labels: Sequence[str],
    metric_key: str,
    normalize: bool,
    K: int,
) -> Dict[str, float]:
    """Valuta P@K (classica) e Voting@K (accuracy) in LOO."""
    splits = loo_splits(len(paths))

    p_at_k_sum = 0.0
    voting_correct = 0
    total = 0

    for ref_idx, qry_idx in splits:
        X_ref = X[ref_idx]
        X_qry = X[qry_idx]
        y_ref = [labels[i] for i in ref_idx]
        y_qry = [labels[i] for i in qry_idx]

        if metric_key == "cosine":
            # Similarità coseno: più alto è meglio
            X_ref_n = l2_normalize(X_ref)
            X_qry_n = l2_normalize(X_qry)
            S = np.dot(X_qry_n, X_ref_n.T)  # shape (1, n_ref) in LOO
            larger_is_better = True
            tk_idx = topk_indices_from_scores(S, K, larger_is_better)
            tk_idx = refine_sorted_topk(S, tk_idx, larger_is_better)
            weights = S[0, tk_idx[0]].tolist()
        else:
            # Distanze: più basso è meglio
            if normalize:
                X_ref = l2_normalize(X_ref)
                X_qry = l2_normalize(X_qry)
            D = cdist(X_qry, X_ref, metric=SUPPORTED[metric_key])  # (1, n_ref)
            larger_is_better = False
            tk_idx = topk_indices_from_scores(D, K, larger_is_better)
            tk_idx = refine_sorted_topk(D, tk_idx, larger_is_better)
            weights = D[0, tk_idx[0]].tolist()

        true_lbl = y_qry[0]
        topk_labels = [y_ref[j] for j in tk_idx[0]]

        rel = sum(1 for lbl in topk_labels if lbl == true_lbl)
        p_at_k_sum += rel / max(1, min(K, len(y_ref)))

        # Voting (tie-break dipende dal tipo di punteggio)
        if metric_key == "cosine":
            pred_vote = majority_vote(topk_labels, weights=weights, prefer_max=True)
        else:
            pred_vote = majority_vote(topk_labels, weights=weights, prefer_max=False)
        if pred_vote == true_lbl:
            voting_correct += 1

        total += 1

    denom = (total + 1e-12)
    k_eff = min(K, len(paths) - 1)  # in LOO il ref ha n-1 elementi
    return {
        f"p_at_{K}": float(p_at_k_sum / denom),
        f"voting_acc_at_{K}": float(voting_correct / denom),
        "k_effective": float(k_eff),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Confronto metriche di distanza su embeddings (LOO 1-NN, senza ratio)")
    p.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS_PATH, help="Percorso a embeddings .npy (default: ../data/embeddings/embeddings_mobilenet_v2.npy)")
    p.add_argument("--label_mode", choices=["dirname", "filename_prefix"], default="dirname", help="Come derivare le etichette dai path")
    p.add_argument("--metrics", default="cosine,euclidean", help="Metriche da testare separate da virgola")
    p.add_argument("--normalize", action="store_true", help="Applica L2-normalizzazione ai vettori prima della distanza")
    p.add_argument("--out_dir", default="../experiments_out", help="Cartella di output: verranno create le sottocartelle 'plots/' (grafici) e 'csv_files/' (CSV)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        X, paths = load_embeddings(args.embeddings)
    except Exception as e:
        print(f"[ERRORE] {e}", file=sys.stderr)
        return 2

    if X.shape[0] < 2:
        print("[ERRORE] Servono almeno 2 embeddings per eseguire LOO 1-NN (almeno 1 riferimento per query).",
              file=sys.stderr)
        return 2

    labels = derive_labels(paths, mode=args.label_mode)

    metrics = [m.strip().lower() for m in (args.metrics or "").split(",") if m.strip()]
    if not metrics:
        print("[ERRORE] Nessuna metrica specificata.", file=sys.stderr)
        return 2

    os.makedirs(os.path.abspath(args.out_dir), exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots")
    csv_dir = os.path.join(args.out_dir, "csv_files")
    os.makedirs(os.path.abspath(plots_dir), exist_ok=True)
    os.makedirs(os.path.abspath(csv_dir), exist_ok=True)

    # Esegui sia SENZA normalizzazione che CON normalizzazione, producendo due grafici separati
    normalize_variants = [False, True]

    for norm in normalize_variants:
        variant_name = "normalized" if norm else "unnormalized"
        print(f"\n===== VALUTAZIONE ({variant_name.upper()}) =====")

        accuracies: Dict[str, float] = {}
        p_at3: Dict[str, float] = {}
        voting3: Dict[str, float] = {}
        for m in metrics:
            if m not in SUPPORTED:
                print(f"[WARN] metrica '{m}' non supportata, salto.", file=sys.stderr)
                continue
            print(f"[INFO] Eseguo metrica={m} | normalize={norm}")
            metr = run_metric(X, paths, labels, metric_key=m, normalize=norm)
            acc = float(metr.get("accuracy", 0.0))
            accuracies[m] = acc
            print(f"[OK] {m}: accuracy={acc:.4f}")

            # Esperimento Top-K (Precision@K, Voting@K) - SOLO COSINE e SOLO CON NORMALIZZAZIONE
            if norm and m == "cosine":
                K = 3
                tk = run_metric_topk(X, paths, labels, metric_key=m, normalize=norm, K=K)
                print(
                    f"[INFO] {m} (norm): "
                    f"Precision@{K}={tk[f'p_at_{K}']:.4f}, "
                    f"Voting accuracy@{K}={tk[f'voting_acc_at_{K}']:.4f} (K_eff={int(tk['k_effective'])})"
                )
                p_at3[m] = float(tk[f'p_at_{K}'])
                voting3[m] = float(tk[f'voting_acc_at_{K}'])
            else:
                # Non eseguire Precision@3 / Voting@3 per metriche diverse da cosine o senza normalizzazione
                pass

            # Calcola curve Precision/Recall/F1/Coverage vs tau (ratio test)
            if len(paths) >= 3:
                y_true, nn_labels, d1, d2 = collect_ratio_data(X, paths, labels, metric_key=m, normalize=norm)
                ratios = np.linspace(0.5, 1, 46, dtype=np.float32)
                results = evaluate_ratio_test(y_true, nn_labels, d1, d2, ratios)
                # Trova best tau per F1
                order = sorted(results.keys())
                f1_vals = [results[t]["f1"] for t in order]
                best_idx = int(np.argmax(f1_vals)) if len(f1_vals) > 0 else None
                best_tau = order[best_idx] if best_idx is not None else None
                best_f1 = f1_vals[best_idx] if best_idx is not None else None
                # Salva plot curva
                curve_png = os.path.join(plots_dir, f"curve_{m}_{variant_name}.png")
                curve_title = f"Curve PR/F1/Coverage vs ratio - {m} ({'norm' if norm else 'no-norm'})"
                plot_curve(order, results, curve_png, curve_title)
                if best_tau is not None and best_f1 is not None:
                    print(f"[OK] {m}: best tau={best_tau:.3f} con F1={best_f1:.4f} | curva: {curve_png}")
                else:
                    print(f"[WARN] {m}: non è stato possibile determinare il best tau | curva: {curve_png}")
            else:
                print(f"[WARN] {m}: dataset troppo piccolo per ratio test (servono >=3 esempi totali). Salto la curva.")

        if accuracies:
            # Salva CSV riepilogo per variante
            summary_csv = os.path.join(csv_dir, f"metrics_summary_{variant_name}.csv")
            save_metrics_summary_csv(accuracies, summary_csv)
            print(f"[OK] Salvato CSV riepilogo: {summary_csv}")

            # Salva grafico a barre per variante (accuracy 1-NN)
            names = list(accuracies.keys())
            scores = [accuracies[n] for n in names]
            title_suffix = "con normalizzazione" if norm else "senza normalizzazione"
            plot_path_acc = os.path.join(plots_dir, f"metrics_accuracy_{variant_name}.png")
            plot_metrics_bar(names, scores, plot_path_acc, title=f"Accuracy per metrica (1-NN LOO) - {title_suffix}",
                             ylabel="Accuracy")
            print(f"[OK] Salvato grafico: {plot_path_acc}")

            # Salva grafici e CSV per Precision@3 e Voting accuracy@3 (solo metriche presenti nei rispettivi dizionari)
            if p_at3:
                # P@3 (solo cosine)
                names_p = list(p_at3.keys())
                scores_p = [p_at3[n] for n in names_p]
                plot_path_p = os.path.join(plots_dir, f"metrics_p_at_3_{variant_name}.png")
                plot_metrics_bar(names_p, scores_p, plot_path_p, title="Precision@3", ylabel="Precision@3",
                                 bar_color="#4C78A8")
                save_scores_csv(p_at3, os.path.join(csv_dir, f"metrics_p_at_3_{variant_name}.csv"), score_name="p_at_3")
                print(f"[OK] Salvati P@3: grafico={plot_path_p}")

                # Voting accuracy@3 (solo cosine)  <-- modificato titolo, ylabel, filename e nome campo CSV
                names_v = list(voting3.keys())
                scores_v = [voting3[n] for n in names_v]
                plot_path_v = os.path.join(plots_dir, f"metrics_voting_accuracy_at_3_{variant_name}.png")
                plot_metrics_bar(names_v, scores_v, plot_path_v, title="Voting accuracy@3", ylabel="Voting accuracy@3",
                                 bar_color="#4C78A8")
                save_scores_csv(voting3, os.path.join(csv_dir, f"metrics_voting_accuracy_at_3_{variant_name}.csv"),
                                score_name="voting_accuracy_at_3")
                print(f"[OK] Salvati Voting accuracy@3: grafico={plot_path_v}")

            # Best metric per variante
            best_name = max(accuracies.items(), key=lambda kv: kv[1])[0]
            print("\n=== RIEPILOGO VARIANTE ===")
            print(f"Metrica migliore ({variant_name}): {best_name} | accuracy={accuracies[best_name]:.3f}")
        else:
            print(f"Nessun risultato per la variante {variant_name} (metrica non valida o problemi negli input)")

    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
