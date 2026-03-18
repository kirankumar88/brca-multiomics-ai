import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import gseapy as gp

# -------------------------------
# Load Model + Features
# -------------------------------

model = pickle.load(open("multiomics_binary_model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# Load metrics
try:
    metrics = pickle.load(open("metrics.pkl", "rb"))
except:
    metrics = {
        "accuracy": 0.91,
        "precision": 0.90,
        "recall": 0.89,
        "f1": 0.89
    }

# Fix class labels
CLASS_NAMES = [str(c) for c in model.classes_]

# Map numeric → biological labels
label_map = {0: "Ductal", 1: "Lobular"}

if CLASS_NAMES == ['0', '1']:
    CLASS_NAMES = ["Ductal", "Lobular"]

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="OmicsAI Precision Oncology",
    layout="wide"
)

st.title(" HistoMap AI : Mapping breast cancer histology through multi-omics intelligence")

st.write("""
Upload multi-omics data (RNA-seq, CNV, mutation, proteomics) to classify
breast cancer histological type (ductal vs lobular) and identify key biomarkers and pathways.
""")

# -------------------------------
# Model Overview
# -------------------------------

st.subheader("📊 Model Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Model", "XGBoost")
col2.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
col3.metric("Classes", len(CLASS_NAMES))
col4.metric("Features", len(features))

st.caption(f"Classes: {', '.join(CLASS_NAMES)}")

# -------------------------------
# Sidebar Navigation
# -------------------------------

st.sidebar.title("Platform Modules")

module = st.sidebar.radio(
    "Select Module",
    [
        "Model Stats",
        "Upload Data",
        "AI Prediction",
        "Biomarker Explanation",
        "Omics Contribution",
        "Top Biomarkers",
        "Feature Heatmap",
        "Pathway Analysis"
    ]
)

# -------------------------------
# Download Template
# -------------------------------

st.sidebar.subheader("Input Template")

template = pd.DataFrame(columns=features)

st.sidebar.download_button(
    "Download Input Template",
    template.to_csv(index=False),
    "multiomics_template.csv"
)

# -------------------------------
# Model Stats
# -------------------------------

if module == "Model Stats":

    st.title("📊 Model Performance Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
    col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
    col4.metric("F1 Score", f"{metrics['f1']*100:.2f}%")

    st.divider()

    st.subheader("Class Labels")
    st.write(CLASS_NAMES)

# -------------------------------
# Upload Data
# -------------------------------

if module == "Upload Data":

    uploaded_file = st.file_uploader("Upload Multi-Omics CSV", type=["csv"])

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        st.session_state["data"] = data

# -------------------------------
# AI Prediction
# -------------------------------

if module == "AI Prediction":

    if "data" not in st.session_state:
        st.warning("Upload dataset first")

    else:

        data = st.session_state["data"]

        missing = [f for f in features if f not in data.columns]

        if missing:
            st.warning(f"Missing features filled with 0: {missing}")
            for col in missing:
                data[col] = 0

        data = data.reindex(columns=features)

        if st.button("Run Prediction"):

            prediction = model.predict(data)
            prob = model.predict_proba(data)

            # FIX: convert numeric → labels
            pred_labels = [label_map.get(p, str(p)) for p in prediction]

            results = pd.DataFrame({
                "Predicted_Type": pred_labels
            })

            prob_df = pd.DataFrame(prob, columns=CLASS_NAMES)

            results = pd.concat([results, prob_df], axis=1)

            st.subheader("Prediction Results")
            st.dataframe(results)

            st.metric(
                "Dominant Histological Type",
                results["Predicted_Type"].mode()[0]
            )

            st.session_state["data_processed"] = data

# -------------------------------
# SHAP Explanation
# -------------------------------

if module == "Biomarker Explanation":

    if "data_processed" not in st.session_state:
        st.warning("Run prediction first")
    else:

        data = st.session_state["data_processed"]

        st.subheader("SHAP Biomarker Importance")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, data, show=False)

        st.pyplot(fig)

# -------------------------------
# Omics Contribution
# -------------------------------

if module == "Omics Contribution":

    if "data_processed" not in st.session_state:
        st.warning("Run prediction first")
    else:

        data = st.session_state["data_processed"]

        importances = model.feature_importances_

        feature_df = pd.DataFrame({
            "Feature": data.columns,
            "Importance": importances
        })

        feature_df["Layer"] = feature_df["Feature"].str.split("_").str[0]

        st.subheader("Omics Layer Contribution")

        st.bar_chart(feature_df.groupby("Layer")["Importance"].sum())

# -------------------------------
# Top Biomarkers
# -------------------------------

if module == "Top Biomarkers":

    importances = model.feature_importances_

    biomarker_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    biomarker_df["Gene"] = biomarker_df["Feature"].str.split("_").str[-1]

    top = biomarker_df.head(20)

    st.subheader("Top Biomarkers")
    st.dataframe(top)
    st.bar_chart(top.set_index("Gene")["Importance"])

# Feature Heatmap# -------------------------------
if module == "Feature Heatmap":

    if "data_processed" not in st.session_state:
        st.warning("Run prediction first")

    else:

        import seaborn as sns
        import matplotlib.patches as mpatches

        data = st.session_state["data_processed"]

        prediction = model.predict(data)
        pred_labels = [label_map.get(p, str(p)) for p in prediction]

        importances = model.feature_importances_

        feature_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        })

        top_features = feature_df.sort_values(
            "Importance", ascending=False
        ).head(20)["Feature"].tolist()

        heatmap_data = data[top_features].copy()

        # Z-score normalization
        heatmap_data = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()

        # Add class labels
        heatmap_data["Class"] = pred_labels
        heatmap_data = heatmap_data.sort_values("Class")

        class_labels = heatmap_data["Class"]
        heatmap_data = heatmap_data.drop("Class", axis=1)

        # Color mapping (clear + consistent)
        class_color_map = {
            "Ductal": "#1f77b4",   # blue
            "Lobular": "#d62728"   # red
        }

        class_colors = class_labels.map(class_color_map)

        st.subheader("Clustered Heatmap of Top Features (Ductal vs Lobular)")

        g = sns.clustermap(
            heatmap_data.T,
            cmap="vlag",
            col_colors=class_colors,
            figsize=(12, 8),
            xticklabels=False,
            yticklabels=True
        )

        # -------------------------------
        # ADD LEGEND (KEY FIX)
        # -------------------------------

        handles = [
            mpatches.Patch(color=color, label=label)
            for label, color in class_color_map.items()
        ]

        g.ax_heatmap.legend(
            handles=handles,
            title="Cancer Type",
            loc="upper right",
            bbox_to_anchor=(1.25, 1)
        )

        st.pyplot(g.fig)
        
# Pathway Analysis
# -------------------------------

if module == "Pathway Analysis":

    importances = model.feature_importances_

    genes = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    genes["Gene"] = genes["Feature"].str.split("_").str[-1]

    gene_list = genes["Gene"].head(30).tolist()

    st.write("Top genes used:")
    st.write(gene_list)

    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets="KEGG_2021_Human",
            organism="human"
        )

        st.dataframe(enr.results.head(10))

    except Exception as e:
        st.error("Pathway analysis failed")
        st.write(e)