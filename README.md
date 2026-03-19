# brca-multiomics-ai

Mapping Breast Cancer Histology through Multi-Omics Intelligence

---

## Overview

brca-multiomics-ai is an interpretable, AI-driven platform designed to classify breast cancer subtypes (Ductal vs Lobular) using integrated multi-omics data, including gene expression, copy number variation (CNV), mutation profiles, and proteomics.

The platform combines machine learning with biological interpretation to identify key biomarkers, quantify omics-layer contributions, and perform pathway enrichment analysis to uncover underlying disease mechanisms.

## Live app link
Streamlit cloud : https://histomap-ai-sj2g3bvn7nw4jef4tqvdey.streamlit.app/

## Dataset: The Cancer Genome Atlas (TCGA)

Reference:
Ciriello G., Gatza M.L., Beck A.H., et al. (2015).
Comprehensive Molecular Portraits of Invasive Lobular Breast Cancer.
Cell, 163(2), 506–519.
https://doi.org/10.1016/j.cell.2015.09.033

---

## Key Features

* Multi-omics data integration across heterogeneous biological layers
* XGBoost-based classification model
* SHAP-based interpretability for biomarker discovery
* KEGG pathway enrichment analysis for biological validation
* Interactive Streamlit-based interface
* FastAPI-based inference backend

---

## System Architecture

```
Streamlit Frontend
        ↓
FastAPI Inference Layer
        ↓
XGBoost Model + SHAP
        ↓
Predictions + Biomarkers + Pathways
```

---

## Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 91.2% |
| Precision | 90.0% |
| Recall    | 89.0% |
| F1 Score  | 89.0% |

---

## Platform Modules

| Module                | Description                   |
| --------------------- | ----------------------------- |
| Model Dashboard       | Overview of model performance |
| Data Upload           | Upload multi-omics dataset    |
| AI Prediction         | Predict cancer subtype        |
| Biomarker Explanation | SHAP-based feature importance |
| Omics Contribution    | Layer-wise importance         |
| Top Biomarkers        | Ranked key genes              |
| Feature Heatmap       | Clustered visualization       |
| Pathway Analysis      | KEGG enrichment analysis      |

---

## Installation and Setup

### 1. Clone the Repository

git clone https://github.com/kirankumar88/brca-multiomics-ai.git
cd HistoMap-AI

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Run Backend (FastAPI)

uvicorn api.api --reload

### 4. Run Frontend (Streamlit)

streamlit run app/app.py

---

## Project Structure

```
HistoMap-AI/
│
├── app/
├── api/
├── models/
├── data/
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Input Requirements

* Input must be provided as a CSV file
* Columns should match model features
* Missing features are automatically handled

---

## Limitations

* Model trained on specific datasets; performance may vary on external data
* Pathway analysis limited to predefined gene sets
* Intended for research use only

---

## License

This project is licensed under the MIT License.

---

## Author

K Kiran Kumar
Bioinformatics, Artificial Intelligence, and Multi-Omics Systems
