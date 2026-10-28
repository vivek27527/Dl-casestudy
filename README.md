# Automated Industrial Quality Inspection Using CNN and Anomaly Detection

Industrial quality inspection system using Convolutional Neural Networks (CNN) for supervised defect classification combined with Convolutional Autoencoders for unsupervised anomaly localization on the **MVTecADsmall** dataset (10 industrial categories).

---

## 1. Project Overview & Flowchart

The project implements the complete industrial inspection lifecycle:

```
[1. Problem Definition] ➔ [2. Data Collection] ➔ [3. Data Preprocessing] ➔ [4. Dataset Splitting]
                                                                                   │
[8. Predict on New Images] ⮜ [7. Evaluate Performance] ⮜ [6. Train Model] ⮜ [5. Build CNN Model]
```

1. **Define the problem**: Identify defective vs. non-defective manufactured products with high precision and recall.
2. **Data collection**: Gather defect-free and defect-bearing images across 10 industrial product categories from `MVTecADsmall`.
3. **Data preprocessing**: Standardize images to 128×128 RGB, normalize pixel intensities to `[0.0, 1.0]`, map binary labels (0: Non-defective, 1: Defective), and calculate defect severity (`none`, `low`, `medium`, `high`) using pixel-accurate ground-truth masks.
4. **Split the dataset**: Stratified 70% Training / 15% Validation / 15% Test partitioning preserving class balance.
5. **Build the CNN model**: Layered architecture featuring `Convolution → ReLU → Pooling → Dense → Output` alongside convolutional autoencoders for anomaly heatmaps.
6. **Train the model**: End-to-end optimization using weighted Binary Cross-Entropy loss to handle industrial class imbalance.
7. **Evaluate performance**: Comprehensive assessment via Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, and category-level diagnostics.
8. **Predict on new images**: Production-ready inference CLI and interactive web app outputting defect classification and anomaly heatmaps.

---

## 2. Dataset Specification (MVTecADsmall: 10 Categories)

The dataset contains 1,836 preprocessed images across **10 industrial categories** spanning two core data types (**Object** and **Texture**):

| Category | Data Type | Train Good | Test Good | Defective | Total Images | Defect Types |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **bottle** | Object | 110 | 10 | 30 | 150 | broken_large, broken_small, contamination |
| **capsule** | Object | 100 | 10 | 50 | 160 | crack, faulty_imprint, poke, scratch, squeeze |
| **carpet** | Texture | 150 | 10 | 50 | 210 | color, cut, hole, metal_contamination, thread |
| **leather** | Texture | 130 | 25 | 50 | 205 | color, cut, fold, glue, poke |
| **metal_nut** | Object | 153 | 10 | 40 | 203 | bent, color, flip, scratch |
| **pill** | Object | 144 | 13 | 69 | 226 | color, combined, contamination, crack, faulty_imprint, pill_type, scratch |
| **screw** | Object | 216 | 9 | 50 | 275 | manipulated_front, scratch_head, scratch_neck, thread_side, thread_top |
| **tile** | Texture | 126 | 20 | 50 | 196 | crack, glue_strip, gray_stroke, oil, rough |
| **toothbrush** | Object | 27 | 10 | 10 | 47 | defective |
| **transistor** | Object | 117 | 27 | 20 | 164 | bent_lead, cut_lead, damaged_case, misplaced |
| **Total** | **7 Objects / 3 Textures** | **1,273** | **144** | **419** | **1,836** | **40 Defect Types** |

---

## 3. Defect Severity Classification

Defects are categorized based on their ground-truth mask area relative to the image:

- **No Defect (`none`)**: 0% area (defect-free / good products)
- **Low Defect (`low`)**: $< 1\%$ of total surface area (subtle scratches, minor pinholes)
- **Medium Defect (`medium`)**: $1\% \text{ to } 5\%$ surface area (localized cracks, edge chips)
- **High Defect (`high`)**: $\ge 5\%$ surface area (structural breaks, large contamination, misassembly)

Detailed separate tables for each of the 10 categories are exported in:
- `outputs/category_tables/<category>_severity_table.csv`
- `outputs/per_category_severity_tables.md`
- `outputs/per_category_severity_tables.xlsx`

---

## 4. CNN Architecture

The classifier strictly adheres to the requested flowchart:

```
Input (128x128x3)
  ├── ConvBlock 1: Conv2D(3➔32, k=3, p=1) ➔ BatchNorm ➔ ReLU ➔ MaxPool(2x2)
  ├── ConvBlock 2: Conv2D(32➔64, k=3, p=1) ➔ BatchNorm ➔ ReLU ➔ MaxPool(2x2)
  ├── ConvBlock 3: Conv2D(64➔128, k=3, p=1) ➔ BatchNorm ➔ ReLU ➔ MaxPool(2x2)
  ├── ConvBlock 4: Conv2D(128➔256, k=3, p=1) ➔ BatchNorm ➔ ReLU ➔ MaxPool(2x2)
  ├── Global Feature Pooling: AdaptiveAvgPool2d(1) ➔ Flatten
  ├── Dense 1: Linear(256 ➔ 128) ➔ ReLU ➔ Dropout(0.4)
  └── Dense 2 (Output): Linear(128 ➔ 1) ➔ Sigmoid Logit
```

---

## 5. Quickstart & Execution

### Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### Run Complete Pipeline (Train, Evaluate & Report)
```powershell
python run_all.py
```

### Generate 10 Category Severity Tables Only (Instant)
```powershell
python run_all.py --reports-only
```

### Predict on Any Image (CLI)
```powershell
python -m src.predict "mvtec_small/bottle/test/broken_large/000.png" --category bottle
```

### Launch Interactive Web Application
```powershell
streamlit run app.py
```

---

## 6. Generated Output Files

All run artifacts are stored in `outputs/`:
- `checkpoints/cnn_defect_classifier.pt`: Trained PyTorch CNN model weights
- `checkpoints/ae_<category>.pt`: Trained Autoencoder weights for anomaly localization
- `category_accuracy_graph.png`: Bar graph of test accuracy by category and data type
- `category_evaluation_comparison.png`: Dual-panel comparison of category accuracy and defect counts
- `confusion_matrix.png`: Heatmap plot of true vs. predicted classifications
- `training_curves.png`: Loss and accuracy curves over epochs
- `cnn_metrics.csv`: Summary of Accuracy, Precision, Recall, F1, ROC-AUC
- `category_summary_small.csv`: Master breakdown of all 10 categories and datatypes
- `category_tables/<category>_severity_table.csv`: 10 individual CSV tables detailing defect severity counts and average defect area percentages
- `per_category_severity_tables.md`: Consolidated Markdown presentation of all 10 category tables
