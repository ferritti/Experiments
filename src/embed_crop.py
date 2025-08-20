from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import cv2

# TensorFlow / Keras
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import (
        MobileNetV2,
        preprocess_input,
    )
except Exception as e:  # pragma: no cover
    print(
        "[ERRORE] TensorFlow/Keras non è installato o non è importabile. Installa tensorflow prima di usare questo script.",
        file=sys.stderr,
    )
    raise


DEFAULT_INPUT_DIR: str = "../data/crops"
DEFAULT_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")
DEFAULT_IMAGE_SIZE: int = 224
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_OUTPUT_DIR: str = "../data/embeddings"


def _ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _gather_images(
    input_dir: str,
    exts: Sequence[str],
    recursive: bool,
    only_crops: bool,
) -> List[str]:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Cartella di input non trovata: {input_dir}")

    def is_valid(fname: str) -> bool:
        low = fname.lower()
        if not any(low.endswith(exts2) for exts2 in exts):
            return False
        if only_crops:
            return "_crop" in low
        return True

    paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(input_dir):
            for f in files:
                if is_valid(f):
                    paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(input_dir):
            if is_valid(f):
                paths.append(os.path.join(input_dir, f))

    paths.sort()
    return paths


def _load_and_preprocess_batch(paths: Sequence[str], image_size: int) -> Tuple[np.ndarray, List[str]]:
    batch: List[np.ndarray] = []
    used: List[str] = []
    for p in paths:
        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Skip silently; caller can log
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
        batch.append(img_rgb.astype(np.float32))
        used.append(p)
    if not batch:
        return np.empty((0, image_size, image_size, 3), dtype=np.float32), []
    x = np.stack(batch, axis=0)
    x = preprocess_input(x)
    return x, used


def compute_embeddings(
    image_paths: Sequence[str],
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[np.ndarray, List[str]]:
    """Compute MobileNetV2 embeddings for given images.

    Returns:
        embeddings: (N, 1280) float32 array
        used_paths: list of paths actually embedded (skips unreadable)
    """
    # Build model once
    model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(image_size, image_size, 3))

    all_embeds: List[np.ndarray] = []
    used_paths: List[str] = []

    # Batch through images
    total = len(image_paths)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        chunk_paths = list(image_paths[start:end])
        x, used_chunk_paths = _load_and_preprocess_batch(chunk_paths, image_size)
        if x.shape[0] == 0:
            continue
        # Compute embeddings
        embeds = model.predict(x, verbose=0)
        if embeds.ndim != 2:
            embeds = embeds.reshape((embeds.shape[0], -1))
        all_embeds.append(embeds.astype(np.float32))
        # Accumula i percorsi effettivamente caricati in questo batch
        used_paths.extend(used_chunk_paths)

    if not all_embeds:
        return np.empty((0, 1280), dtype=np.float32), []

    embeddings = np.concatenate(all_embeds, axis=0)
    return embeddings, used_paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcola embeddings con MobileNetV2 per i crop generati e li salva in .npy. "
            "Per default considera solo file che contengono '_crop' nel nome."
        )
    )
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Cartella radice dei crop con struttura data/crops/{quadri,statue}/<opera>/*; la scansione è ricorsiva")
    parser.add_argument(
        "--output",
        default=None,
        help="Percorso file .npy di output (default: ../data/embeddings/embeddings_mobilenet_v2.npy)",
    )
    parser.add_argument(
        "--recursive", action="store_true", default=True, help="Cerca immagini ricorsivamente nella cartella (default: attivo)"
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Estensioni immagine accettate separate da virgola (default: .png,.jpg,.jpeg,.bmp)",
    )
    parser.add_argument(
        "--only_crops",
        action="store_true",
        default=True,
        help="Considera solo file con '_crop' nel nome (default: attivo)",
    )
    parser.add_argument("--image_size", type=int, default=DEFAULT_IMAGE_SIZE, help="Dimensione lato input (default: 224)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size per inferenza (default: 32)")
    parser.add_argument(
        "--save_paths",
        action="store_true",
        default=True,
        help="Salva anche un file .txt con i percorsi delle immagini nello stesso ordine (default: attivo)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    exts = [e.strip().lower() for e in (args.extensions or "").split(",") if e.strip()]
    exts = [e if e.startswith(".") else f".{e}" for e in exts]
    exts = tuple(exts) if exts else DEFAULT_EXTENSIONS

    try:
        image_paths = _gather_images(
            input_dir=args.input_dir, exts=exts, recursive=bool(args.recursive), only_crops=bool(args.only_crops)
        )
    except Exception as e:
        print(f"[ERRORE] {e}", file=sys.stderr)
        return 2

    if not image_paths:
        print("[INFO] Nessuna immagine trovata da elaborare.")
        return 2

    print(f"[INFO] Immagini trovate: {len(image_paths)}")

    embeddings, used_paths = compute_embeddings(
        image_paths=image_paths, image_size=int(args.image_size), batch_size=int(args.batch_size)
    )

    if embeddings.size == 0 or not used_paths:
        print("[INFO] Nessuna immagine valida per l'estrazione degli embeddings.")
        return 2

    # Determina percorso output
    output_path = args.output
    if not output_path:
        output_dir = DEFAULT_OUTPUT_DIR
        _ensure_dir(os.path.abspath(output_dir))
        output_path = os.path.join(output_dir, "embeddings_mobilenet_v2.npy")
    # Se è una directory, salva come file predefinito al suo interno
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "embeddings_mobilenet_v2.npy")

    _ensure_dir(os.path.dirname(os.path.abspath(output_path)))

    np.save(output_path, embeddings)
    print(f"[OK] Salvati embeddings in: {output_path} | shape={embeddings.shape}")

    if args.save_paths:
        paths_txt = os.path.splitext(output_path)[0] + ".paths.txt"
        try:
            with open(paths_txt, "w", encoding="utf-8") as f:
                for p in used_paths:
                    f.write(p + "\n")
            print(f"[OK] Salvati percorsi immagini in: {paths_txt}")
        except Exception as e:
            print(f"[ATTENZIONE] Impossibile salvare i percorsi: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
