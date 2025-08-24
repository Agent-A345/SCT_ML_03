from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


IMAGE_SIZE = 64
SEED = 42
LIMIT_PER_CLASS = None  

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def extract_id_from_name(name: str) -> str:
    m = re.findall(r"\d+", name)
    return m[-1] if m else Path(name).stem

def load_and_flatten(img_path: Path, size: int) -> np.ndarray:
    img = Image.open(img_path).convert("L").resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()

def main():
    script_dir = Path(__file__).resolve().parent
    train_dir = script_dir / "train"
    test_dir = script_dir / "test1"
    outdir = script_dir / "runs" / "simple_svm"
    outdir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists():
        print(f"[ERROR] train folder not found at: {train_dir}")
        sys.exit(1)
    if not test_dir.exists():
        print(f"[ERROR] test1 folder not found at: {test_dir}")
        sys.exit(1)

    #Train files
    all_train = [p for p in train_dir.iterdir() if p.is_file() and is_image(p)]
    cats = [p for p in all_train if "cat" in p.name.lower()]
    dogs = [p for p in all_train if "dog" in p.name.lower()]
    if LIMIT_PER_CLASS is not None:
        cats = cats[:LIMIT_PER_CLASS]
        dogs = dogs[:LIMIT_PER_CLASS]
    files = cats + dogs
    labels = [0]*len(cats) + [1]*len(dogs)

    if len(files) == 0:
        print("[ERROR] No train images found. Filenames must contain 'cat' or 'dog'.")
        sys.exit(1)

    #x,y
    print(f"Loading {len(files)} train images (cats={len(cats)}, dogs={len(dogs)})...")
    X = [load_and_flatten(p, IMAGE_SIZE) for p in tqdm(files, desc="Train images")]
    X = np.vstack(X).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    #Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    #Training SVM
    print("Training Linear SVM...")
    clf = LinearSVC(C=1.0, max_iter=10000, random_state=SEED)
    clf.fit(X_tr, y_tr)

    #Evvaluating
    val_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    report = classification_report(y_val, val_pred, target_names=["cat (0)", "dog (1)"])
    cm = confusion_matrix(y_val, val_pred)

    print(f"\nValidation accuracy: {acc:.4f}")
    print("\nClassification report:\n", report)
    print("Confusion matrix:\n", cm)

    #Model & Metrics
    model_path = outdir / "model.joblib"
    joblib.dump(clf, model_path)  
    with open(outdir / "metrics.txt", "w") as f:
        f.write(f"Validation accuracy: {acc:.6f}\n\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(str(cm))
    print(f"\nSaved model -> {model_path}")
    print(f"Saved metrics -> {outdir / 'metrics.txt'}")

    #Test1
    test_files = sorted(
        [p for p in test_dir.iterdir() if p.is_file() and is_image(p)],
        key=lambda p: p.name,
    )
    if not test_files:
        print("[WARN] No images in test1/.")
        return

    print(f"\nLoading {len(test_files)} test images...")
    X_test = [load_and_flatten(p, IMAGE_SIZE) for p in tqdm(test_files, desc="Test images")]
    X_test = np.vstack(X_test).astype(np.float32)
    ids = [extract_id_from_name(p.name) for p in test_files]

    test_pred = clf.predict(X_test).astype(int)  # dog=1, cat=0
    sub = pd.DataFrame({"id": ids, "label": test_pred})
    sub.to_csv(outdir / "submission.csv", index=False)
    print(f"Saved predictions -> {outdir / 'submission.csv'}")

if __name__ == "__main__":
    main()
