"""Automated Industrial Quality Inspection Using CNN and Anomaly Detection (MVTecADsmall).

Flowchart Steps:
  1. Define the problem: Identify defective vs. non-defective products
  2. Data collection: Collect labeled images from MVTecADsmall (10 categories)
  3. Data preprocessing: Resize images (128x128), normalize pixel values, label classes & defect severity
  4. Split the dataset: Training set (70%) • Validation set (15%) • Test set (15%)
  5. Build the CNN model: Convolution → ReLU → Pooling → Dense → Output
  6. Train the model: Learn patterns using training images and labels
  7. Evaluate performance: Accuracy, precision, recall, F1-score, confusion matrix
  8. Predict on new images: Output Defective or Non-defective + Anomaly Detection
  At the end: 10 separate tables of datatypes with defect severity (no, low, medium, high).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    AE_EPOCHS,
    CATEGORIES,
    CNN_EPOCHS,
    IMG_SIZE,
    OUT_DIR,
    REPORT_DIR,
    get_data_type,
)
from src.dataset import build_index, load_rgb, populate_image_cache, supervised_splits
from src.evaluate import evaluate_cnn, predict_image
from src.extract_dataset import resolve_mvtec_root
from src.models import get_device
from src.reports import safe_save_csv, save_reports
from src.train import train_autoencoder, train_cnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated Industrial Quality Inspection using CNN & Anomaly Detection"
    )
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Build index and 10 category tables only without training",
    )
    parser.add_argument(
        "--skip-ae",
        action="store_true",
        help="Skip autoencoder anomaly detection training",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run (fewer epochs for fast smoke-test)",
    )
    parser.add_argument(
        "--cnn-epochs",
        type=int,
        default=CNN_EPOCHS,
        help=f"Number of CNN epochs (default: {CNN_EPOCHS})",
    )
    parser.add_argument(
        "--ae-epochs",
        type=int,
        default=AE_EPOCHS,
        help=f"Number of Autoencoder epochs (default: {AE_EPOCHS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cnn_epochs = 3 if args.quick else args.cnn_epochs
    ae_epochs = 2 if args.quick else args.ae_epochs

    print("=" * 75)
    print(" AUTOMATED INDUSTRIAL QUALITY INSPECTION USING CNN & ANOMALY DETECTION")
    print(" Dataset: MVTecADsmall (10 Categories)")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. Define the Problem
    # -------------------------------------------------------------
    print("\n[Step 1/8] Defining the Problem:")
    print(" - Goal: Automated visual inspection to identify Defective vs. Non-defective parts.")
    print(" - Objective: High precision and recall to prevent defective parts from passing inspection.")
    print(" - Anomaly Detection: Detect subtle defects via unsupervised reconstruction residuals.")

    # -------------------------------------------------------------
    # 2. Data Collection
    # -------------------------------------------------------------
    print("\n[Step 2/8] Data Collection:")
    mvtec_root = resolve_mvtec_root()
    print(f" - Found MVTecADsmall dataset at: {mvtec_root}")
    print(f" - Target categories (10): {', '.join(CATEGORIES)}")

    # -------------------------------------------------------------
    # 3. Data Preprocessing
    # -------------------------------------------------------------
    print("\n[Step 3/8] Data Preprocessing:")
    print(" - Scanning directories and associating ground truth defect masks...")
    df = build_index(mvtec_root)
    index_path = OUT_DIR / "mvtec_small_index.csv"
    safe_save_csv(df, OUT_DIR / "mvtec_index.csv")
    safe_save_csv(df, index_path)
    print(f" - Total images indexed: {len(df)}")
    print(f" - Image resolution: 128x128 RGB, normalized to [0, 1]")
    print(f" - Ground truth defect severity: Classified into 'no', 'low' (<1%), 'medium' (1-5%), and 'high' (>5%)")
    print(f" - Saved dataset index to: {index_path}")

    # Build memory / disk cache for fast DataLoader loading
    populate_image_cache(df)

    if args.reports_only:
        print("\nGenerating 10 category datatype & severity tables...")
        save_reports(df, verbose=True)
        return

    # -------------------------------------------------------------
    # 4. Split the Dataset
    # -------------------------------------------------------------
    print("\n[Step 4/8] Splitting the Dataset:")
    train_df, val_df, test_df = supervised_splits(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )
    print(f" - Training set   : {len(train_df):4d} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f" - Validation set : {len(val_df):4d} images ({len(val_df)/len(df)*100:.1f}%)")
    print(f" - Test set       : {len(test_df):4d} images ({len(test_df)/len(df)*100:.1f}%)")
    print(f"   (Stratified on binary label: Non-defective vs Defective)")

    # -------------------------------------------------------------
    # 5 & 6. Build the CNN Model & Train
    # -------------------------------------------------------------
    print("\n[Step 5/8] Building CNN Model Architecture:")
    print(" - Convolution → ReLU → Pooling → Dense → Output")
    print(" - 4 Conv Blocks (32, 64, 128, 256 filters) with BatchNorm and MaxPool")
    print(" - Adaptive Average Pooling + Dropout(0.4) + Dense(128) + Sigmoid Output")

    print("\n[Step 6/8] Training the CNN Model:")
    model = train_cnn(train_df, val_df, epochs=cnn_epochs)

    # -------------------------------------------------------------
    # 7. Evaluate Performance
    # -------------------------------------------------------------
    print("\n[Step 7/8] Evaluating Performance on Test Set:")
    metrics = evaluate_cnn(model, test_df)

    # -------------------------------------------------------------
    # 8. Anomaly Detection (Autoencoder) & Predict on New Images
    # -------------------------------------------------------------
    if not args.skip_ae:
        print("\n[Step 8/8] Anomaly Detection: Training Autoencoders on Defect-Free Parts:")
        ae_results = []
        for cat in CATEGORIES:
            cat_df = df[df.category == cat]
            res = train_autoencoder(cat_df, cat, epochs=ae_epochs)
            ae_results.append(res)
            print(f"   * {cat:<12} (Data Type: {get_data_type(cat):<7}) | "
                  f"AUROC: {res['auroc']} | Test Accuracy: {res['accuracy']*100:.1f}%")

        ae_summary_df = pd.DataFrame(ae_results)
        safe_save_csv(ae_summary_df, OUT_DIR / "anomaly_detection_results.csv")
        valid_aurocs = ae_summary_df["auroc"].dropna()
        if not valid_aurocs.empty:
            print(f"\nMean Autoencoder Anomaly AUROC: {valid_aurocs.mean():.4f}")

    # Sample single-image inference demo
    sample_row = test_df[test_df["label"] == 1].iloc[0]
    sample_rgb = load_rgb(sample_row.path, IMG_SIZE)
    sample_tensor = torch.from_numpy(sample_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    pred_label, pred_prob = predict_image(model, sample_tensor)
    print(f"\n[Step 8 Demo] Inference verification on sample defective image:")
    print(f" - Path: {sample_row.path}")
    print(f" - Category: {sample_row.category} ({sample_row.data_type})")
    print(f" - Actual Defect: {sample_row.defect_type} (Severity: {sample_row.severity})")
    print(f" - Model Prediction: {pred_label} (Defect probability: {pred_prob*100:.2f}%)")

    # -------------------------------------------------------------
    # Final Deliverable: 10 Separate Category Tables
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print(" FINAL STEP: GENERATING 10 CATEGORY DATATYPE & SEVERITY TABLES")
    print("=" * 75)
    save_reports(df, verbose=True)

    print("\nExecution complete! All artifacts and models saved in:", OUT_DIR)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
