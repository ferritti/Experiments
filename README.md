# Experiments — Guida e descrizione dell’esperimento

Questo progetto contiene uno script per generare embeddings di immagini (con MobileNetV2) e uno script per valutare, in maniera sistematica, diverse metriche di similarità tramite classificazione 1-NN in Leave-One-Out (LOO) e un “ratio test” in stile 2-NN di Lowe. Gli output includono CSV riepilogativi e grafici (bar plot delle accuracy e curve PR/Recall/F1/Coverage al variare della soglia del ratio test).

In sintesi, l’esperimento misura quanto bene semplici regole di nearest neighbour riescano a riconoscere immagini dello stesso “autore/opera/classe” a partire da vettori di embedding estratti da una rete pre-addestrata.

- Generazione embeddings: `src/embed_crop.py`
- Esecuzione esperimenti: `src/run_experiments.py`
- Output: `experiments_out/` (sottocartelle `plots/` e `csv_files/`)


## 1) Prerequisiti e installazione

- Python 3.10+ consigliato.
- Si raccomanda un ambiente virtuale (venv o conda).
- Installare i requisiti:
  
  - con venv (esempio):
    - `python -m venv .venv`
    - attiva: macOS/Linux `source .venv/bin/activate` — Windows `./.venv/Scripts/activate`
    - `python -m pip install --upgrade pip`
    - `pip install -r requirements.txt`

Note su TensorFlow:
- In `requirements.txt` sono presenti pacchetti per CPU e, su macOS Apple Silicon, anche `tensorflow-macos`. In caso di conflitti, installare solo il pacchetto appropriato alla propria piattaforma (ad esempio rimuovendo l’altro dal file o scegliendo la distribuzione supportata).


## 2) Struttura dei dati attesa

Per default, lo script di embedding cerca immagini dentro `data/crops` (scansione ricorsiva) e considera solo file che contengono la sottostringa `_crop` nel nome.

Esempio di struttura (indicativa):

- `data/crops/quadri/NomeOpera/immagine_1_crop.jpg`
- `data/crops/statue/NomeOpera/immagine_2_crop.png`

Durante la valutazione, le etichette (classi) vengono derivate automaticamente dai path delle immagini — a scelta tra:
- `--label_mode dirname`: usa il nome della cartella padre come etichetta (default).
- `--label_mode filename_prefix`: ricava l’etichetta dal prefisso del filename (es. `starry_night_1_crop.png` → `starry_night`).

Se non possiedi file con `_crop` nel nome, puoi:
- Rinominare/duplicare i file per includere `_crop`, oppure
- Modificare lo script `src/embed_crop.py` per rimuovere il filtro (vedi nota in “Limitazioni”).


## 3) Generare gli embeddings (MobileNetV2)

Script: `src/embed_crop.py`

Funzionalità principali:
- Carica immagini da una cartella (default `../data/crops` relativamente alla posizione dello script).
- Estrae embeddings con MobileNetV2 (pooling=avg, dimensione 1280).
- Salva un file `.npy` con shape `(N, 1280)` e un file `.paths.txt` con l’elenco dei percorsi processati nello stesso ordine.

Opzioni principali (estratto):
- `--input_dir`: cartella radice delle immagini (default: `../data/crops`).
- `--output`: file `.npy` di destinazione (default: `../data/embeddings/embeddings_mobilenet_v2.npy`).
- `--image_size`: lato dell’immagine di input per la rete (default: 224).
- `--batch_size`: batch per l’inferenza (default: 32).

Nota importante: nella versione attuale i flag `--recursive`, `--only_crops` e `--save_paths` sono già attivi e non disattivabili via CLI (sono implementati come `store_true` con default=True). Pertanto lo script:
- Scansiona ricorsivamente le sottocartelle.
- Considera solo file i cui nomi contengono `_crop`.
- Salva sempre il file `.paths.txt` accanto al `.npy`.

Esempio di esecuzione:
- `python src/embed_crop.py --input_dir data/crops --output data/embeddings/embeddings_mobilenet_v2.npy`

Output attesi:
- `data/embeddings/embeddings_mobilenet_v2.npy`
- `data/embeddings/embeddings_mobilenet_v2.paths.txt`


## 4) Eseguire gli esperimenti (LOO 1‑NN + Ratio Test)

Script: `src/run_experiments.py`

Cosa fa:
- Carica gli embeddings `.npy` e il corrispondente `.paths.txt`.
- Deriva le etichette dai path (vedi `--label_mode`).
- Esegue classificazione Leave‑One‑Out con 1‑NN usando metriche specificate (cosine, euclidean).
- Confronta l’accuratezza tra varianti senza normalizzazione e con normalizzazione L2 dei vettori.
- Calcola curve Precision/Recall/F1/Coverage al variare della soglia del ratio test (d1/d2) stile Lowe 2‑NN, individuando il tau che massimizza F1.

Opzioni principali:
- `--embeddings`: percorso al `.npy` (default: `../data/embeddings/embeddings_mobilenet_v2.npy`).
- `--label_mode`: `dirname` (default) oppure `filename_prefix`.
- `--metrics`: elenco separato da virgola, es. `cosine,euclidean` (default: entrambi).
- `--out_dir`: directory di output (default: `../experiments_out`).

Nota: il flag `--normalize` è attualmente ignorato; il codice esegue comunque entrambe le varianti (non normalizzato e normalizzato) e salva i risultati separatamente.

Esempio di esecuzione:
- `python src/run_experiments.py --embeddings data/embeddings/embeddings_mobilenet_v2.npy --label_mode dirname --metrics cosine,euclidean --out_dir experiments_out`

Output prodotti in `experiments_out/`:
- `plots/metrics_accuracy_unnormalized.png`
- `plots/metrics_accuracy_normalized.png`
- `plots/curve_{cosine,euclidean}_{unnormalized,normalized}.png` (se il dataset ha almeno 3 immagini totali)
- `csv_files/metrics_summary_unnormalized.csv`
- `csv_files/metrics_summary_normalized.csv`

Log a console: per ogni metrica e variante, mostra l’accuracy e, per le curve del ratio test, il miglior `tau` (soglia su d1/d2) secondo F1.

Requisiti minimi:
- Almeno 2 embeddings per eseguire LOO 1‑NN (almeno 1 riferimento per la query).
- Almeno 3 immagini totali per poter tracciare la curva del ratio test.


## 5) In cosa consiste l’esperimento

Obiettivo: valutare la capacità di embeddings generici (MobileNetV2 pre‑addestrata su ImageNet) di distinguere classi visive nel dataset, con un classificatore 1‑NN estremamente semplice.

- LOO 1‑NN: per ciascuna immagine, la si usa come query contro tutte le altre come riferimento; la classe predetta è quella della più vicina (min distanza o max similarità).
- Metriche: si confrontano almeno `cosine` e `euclidean`, con e senza normalizzazione L2 dei vettori, per capire l’impatto della normalizzazione.
- Ratio test (2‑NN): si analizza il rapporto d1/d2 tra la distanza del 1° e del 2° vicino più prossimo: più il rapporto è basso, più il match è “sicuro”. Variando la soglia `tau`, si ottengono precision/recall/F1/coverage e si individua un compromesso ottimale (massimo F1), utile per impostare una politica di “no‑match” in caso di ambiguità.

In uscita, l’esperimento fornisce:
- Accuracy di 1‑NN per ogni metrica e variante di normalizzazione.
- Curve PR/Recall/F1/Coverage vs `tau` e il valore di `tau` che massimizza F1.


## 6) Esempi pratici rapidi

1) Genera embeddings dai crop:
- `python src/embed_crop.py --input_dir data/crops --output data/embeddings/embeddings_mobilenet_v2.npy`

2) Esegui valutazione su cosine+euclidean:
- `python src/run_experiments.py --embeddings data/embeddings/embeddings_mobilenet_v2.npy --metrics cosine,euclidean --out_dir experiments_out`

Controlla `experiments_out/plots/` e `experiments_out/csv_files/` per i risultati.


## 7) Troubleshooting

- "Embeddings .npy non trovato" o "File paths non trovato": assicurati di aver eseguito `embed_crop.py` e che `embeddings_mobilenet_v2.npy` e `embeddings_mobilenet_v2.paths.txt` siano nello stesso percorso.
- Nessuna immagine trovata in `embed_crop.py`: ricorda che, in questa versione, lo script considera per default solo file con `_crop` nel nome. Adegua i nomi file o modifica lo script per rimuovere il filtro.
- Errori TensorFlow/Keras: verifica la compatibilità della versione con la tua piattaforma; su Apple Silicon può essere necessario `tensorflow-macos`.
- Dataset troppo piccolo: servono >=2 immagini per 1‑NN LOO e >=3 per il ratio test.


## 8) Limitazioni e note

- I flag `--recursive`, `--only_crops`, `--save_paths` di `embed_crop.py` sono sempre attivi (non disattivabili via CLI nella versione corrente).
- In `run_experiments.py` il flag `--normalize` è attualmente non operativo: lo script esegue entrambe le varianti (non normalizzato e normalizzato) in ogni caso.
- Gli embeddings sono generati da MobileNetV2 pre‑addestrata su ImageNet; risultati e metriche dipendono dalla natura del dataset (quadri, statue, ecc.).


## 9) Struttura del repository (riepilogo)

- `src/`
  - `embed_crop.py`: genera embeddings e salva `.npy` + `.paths.txt`.
  - `run_experiments.py`: valuta 1‑NN LOO e produce grafici/CSV, incluse le curve del ratio test.
- `data/`
  - `crops/` (attesa per default da `embed_crop.py`)
  - `embeddings/` (uscita di `embed_crop.py`)
- `experiments_out/` (uscita degli esperimenti)
- `requirements.txt`


## 10) Riproducibilità

- Fissa le versioni con `requirements.txt` (già incluso).
- Mantieni stabile la struttura dei path (lo stesso ordine del `.paths.txt` viene usato nella valutazione).
- Usa gli stessi parametri CLI per replicare i risultati (dataset, metrica, label_mode, ecc.).
