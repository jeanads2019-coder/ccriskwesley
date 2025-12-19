import shap
import numpy as np
import pandas as pd

def compute_shap_single(pipeline, X_client):
    """
    Compute SHAP values for a single client only.
    """

    model = pipeline.named_steps["classifier"]
    preprocess = pipeline.named_steps["preprocessor"]

    # 1️⃣ Transformar apenas o cliente
    X_transformed = preprocess.transform(X_client)

    feature_names = preprocess.get_feature_names_out()

    X_transformed_df = pd.DataFrame(
        X_transformed,
        columns=feature_names
    )

    # 2️⃣ SHAP
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        X_transformed_df,
        check_additivity=False
    )

    return shap_values, X_transformed_df


def shap_summary(feature_names, client_features, shap_values):
    df_shap = pd.DataFrame({
        'feature': feature_names,
        'value': client_features.values,
        'shap': shap_values
    })

    df_shap['abs_shap'] = df_shap['shap'].abs()
    df_shap = df_shap.sort_values('abs_shap', ascending=False).head(5)

    summary = []
    for _, row in df_shap.iterrows():
        direction = "Increase Risk" if row['shap'] > 0 else "Decrease Risk"
        summary.append({
            'feature': row['feature'],
            'value': row['value'],
            'direction': direction
        })

    return summary