from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


DEFAULT_IMAGE: Optional[str] = None  # es. "/percorso/immagine.jpg"
DEFAULT_IMAGE_DIR: Optional[str] = "../data/images"
DEFAULT_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_MODEL: str = "../last_model.tflite"
DEFAULT_OUTPUT_DIR: Optional[str] = "../data/crops"
DEFAULT_SCORE_THRESHOLD: float = 0.25
DEFAULT_MARGIN: float = 0.0
DEFAULT_SAVE_ALL: bool = False
DEFAULT_RECURSIVE: bool = True

@dataclass
class DetectionCrop:
    index: int
    score: float
    category: Optional[str]
    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    output_path: str


def _load_image_bgr(image_path: str) -> np.ndarray:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File immagine non trovato: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossibile leggere l'immagine: {image_path}")
    return img


def _ensure_dir(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _expand_and_clip_bbox(x: int, y: int, w: int, h: int, img_w: int, img_h: int, margin: float) -> Tuple[int, int, int, int]:
    # Margine in frazione della dimensione del box
    dx = int(round(w * float(margin)))
    dy = int(round(h * float(margin)))
    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(img_w, x + w + dx)
    y2 = min(img_h, y + h + dy)
    # Garantire coordinate valide
    if x2 <= x1:
        x2 = min(img_w, x1 + max(1, w))
    if y2 <= y1:
        y2 = min(img_h, y1 + max(1, h))
    return x1, y1, x2, y2


def detect_artwork(
    image_path: str,
    model_path: str = "last_model.tflite",
    output_dir: Optional[str] = None,
    score_threshold: float = 0.25,
    margin: float = 0.0,
    save_all: bool = False,
) -> List[DetectionCrop]:
    """Esegue detection e salva i crop. Ritorna la lista dei crop creati."""
    # Carica immagine
    bgr = _load_image_bgr(image_path)
    img_h, img_w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")

    # Prepara MediaPipe ObjectDetector
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.ObjectDetectorOptions(
        base_options=base_options,
        score_threshold=float(score_threshold),
        max_results=25,
    )
    detector = mp_vision.ObjectDetector.create_from_options(options)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    detections = getattr(result, "detections", []) or []
    if not detections:
        return []

    # Prepara directory output
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(image_path))
    _ensure_dir(output_dir)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Costruisci lista di candidates con priorità a categoria 'Artwork' se presente
    candidates = []
    for i, det in enumerate(detections):
        bbox = det.bounding_box  # x,y,width,height in pixel
        x1, y1, x2, y2 = _expand_and_clip_bbox(bbox.origin_x, bbox.origin_y, bbox.width, bbox.height, img_w, img_h, margin)
        # Prendi la categoria migliore (max score)
        best_cat = None
        best_score = 0.0
        for cat in det.categories:
            if cat.score is not None and cat.score > best_score:
                best_score = float(cat.score)
                best_cat = cat.category_name or None
        # Filtro per score minimo
        if best_score < float(score_threshold):
            continue
        # Flag preferenza se etichetta è Artwork (case-insensitive)
        is_artwork = (best_cat or "").strip().lower() == "artwork"
        candidates.append((i, best_score, best_cat, is_artwork, (x1, y1, x2, y2)))

    if not candidates:
        return []

    # Ordina: prima quelli con Artwork, poi per score decrescente
    candidates.sort(key=lambda t: (not t[3], -t[1]))

    crops: List[DetectionCrop] = []

    selected = candidates if save_all else [candidates[0]]
    for rank, (idx, score, cat_name, is_artwork, (x1, y1, x2, y2)) in enumerate(selected, start=1):
        crop_rgb = rgb[y1:y2, x1:x2]
        if crop_rgb.size == 0:
            continue
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        if save_all:
            out_name = f"{base_name}_crop_{rank}.png"
        else:
            out_name = f"{base_name}_crop.png"
        out_path = os.path.join(output_dir, out_name)
        ok = cv2.imwrite(out_path, crop_bgr)
        if not ok:
            raise IOError(f"Errore nel salvataggio del crop: {out_path}")
        crops.append(
            DetectionCrop(
                index=idx,
                score=score,
                category=cat_name,
                bbox_xyxy=(x1, y1, x2, y2),
                output_path=out_path,
            )
        )

    return crops


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rileva opere (Artwork) con MediaPipe Object Detector e salva il/i crop")
    parser.add_argument("--image", required=False, default=DEFAULT_IMAGE, help="Percorso dell'immagine di input")
    parser.add_argument("--image_dir", required=False, default=DEFAULT_IMAGE_DIR, help="Cartella radice delle immagini, con struttura: data/images/{quadri,statue}/<opera>/*; i crop saranno salvati in data/crops mantenendo la stessa struttura")
    parser.add_argument("--ext", default=",".join(DEFAULT_EXTENSIONS), help="Estensioni immagini accettate separate da virgola (es: .jpg,.png)")
    parser.add_argument("--recursive", action="store_true", default=DEFAULT_RECURSIVE, help="Cerca immagini ricorsivamente dentro la cartella (default: attivo)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Percorso al modello TFLite (default: last_model.tflite)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory radice di output per i crop (default: Data/Crops)")
    parser.add_argument("--score_threshold", type=float, default=DEFAULT_SCORE_THRESHOLD, help="Soglia di confidenza minima [0..1] (default: 0.25)")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN, help="Margine intorno al bbox come frazione (es. 0.1 = 10%)")
    parser.add_argument("--all", action="store_true", default=DEFAULT_SAVE_ALL, help="Salva tutti i crop invece del migliore")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Prepara lista estensioni
    exts = [e.strip().lower() for e in (args.ext or "").split(",") if e.strip()]
    exts = [e if e.startswith(".") else f".{e}" for e in exts]
    exts = tuple(exts) if exts else tuple(DEFAULT_EXTENSIONS)

    # Raccogli immagini da processare
    image_paths: List[str] = []
    if args.image_dir:
        if not os.path.isdir(args.image_dir):
            print(f"[ERRORE] Cartella immagini non trovata: {args.image_dir}", file=sys.stderr)
            return 1
        if args.recursive:
            for root, _, files in os.walk(args.image_dir):
                for f in files:
                    if f.lower().endswith(exts):
                        image_paths.append(os.path.join(root, f))
        else:
            for f in os.listdir(args.image_dir):
                if f.lower().endswith(exts):
                    image_paths.append(os.path.join(args.image_dir, f))
        image_paths.sort()
    elif args.image:
        image_paths = [args.image]
    else:
        print("[ERRORE] Specifica --image oppure --image_dir, oppure imposta DEFAULT_IMAGE/DEFAULT_IMAGE_DIR nello script.", file=sys.stderr)
        return 2

    if not image_paths:
        print("[INFO] Nessuna immagine trovata con le estensioni specificate.")
        return 2

    # Determina radice di output
    output_root = args.output_dir

    # Pre-crea la struttura di categorie nel folder dei crops
    if output_root:
        _ensure_dir(output_root)
        _ensure_dir(os.path.join(output_root, "quadri"))
        _ensure_dir(os.path.join(output_root, "statue"))

    total_images = len(image_paths)
    total_crops = 0
    no_detections: List[str] = []

    for img_path in image_paths:
        # Calcola la directory di output per questa immagine
        if args.image_dir:
            # Mappa la struttura di sottocartelle da image_dir a output_root
            root = output_root or DEFAULT_OUTPUT_DIR
            img_parent = os.path.dirname(img_path)
            try:
                rel_dir = os.path.relpath(img_parent, start=args.image_dir)
            except ValueError:
                # In caso di percorsi non correlati, salva direttamente sotto root
                rel_dir = ""
            per_out_dir = os.path.join(root, rel_dir) if rel_dir and rel_dir != os.curdir else root
        else:
            # Modalità singola immagine
            per_out_dir = output_root or os.path.dirname(os.path.abspath(img_path))

        _ensure_dir(per_out_dir)

        try:
            crops = detect_artwork(
                image_path=img_path,
                model_path=args.model,
                output_dir=per_out_dir,
                score_threshold=args.score_threshold,
                margin=args.margin,
                save_all=args.all,
            )
        except Exception as e:
            print(f"[ERRORE] {img_path}: {e}", file=sys.stderr)
            no_detections.append(img_path)
            continue

        if not crops:
            print(f"[INFO] Nessuna opera rilevata: {img_path}")
            no_detections.append(img_path)
            continue

        total_crops += len(crops)
        print(f"Creati {len(crops)} crop per {os.path.basename(img_path)}:")
        for c in crops:
            cat = c.category or "(sconosciuta)"
            x1, y1, x2, y2 = c.bbox_xyxy
            print(f" - {c.output_path} | score={c.score:.3f} | categoria={cat} | bbox=({x1},{y1})-({x2},{y2})")

    # Riepilogo finale
    if total_images == 1:
        return 0 if total_crops > 0 else 2

    print("\n=== RIEPILOGO BATCH ===")
    print(f"Immagini processate: {total_images}")
    print(f"Totale crop creati: {total_crops}")
    if no_detections:
        print(f"Senza rilevazioni ({len(no_detections)}):")
        for p in no_detections:
            print(f" - {p}")

    return 0 if total_crops > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())