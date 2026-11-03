import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector, SelectKBest, f_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score
from datetime import datetime
from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL
from audit import generate_variable_audit_table

def calculate_smoothed_weights(y_data, smoothing='log'):
    class_counts = Counter(y_data)
    majority_count = max(class_counts.values())
    
    weights = {}
    for cls, count in class_counts.items():
        ratio = majority_count / count
        if smoothing == 'sqrt':
            weights[cls] = round(float(np.sqrt(ratio)), 2)
        elif smoothing == 'log':
            weights[cls] = round(float(np.log10(ratio) + 1.0), 2)
        else:
            weights[cls] = round(float(ratio), 2)
    return weights

def get_llm_interpretation(coeff_df_string, target_etf, horizon_days, max_retries=3, delay=10):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "DEIN_API_KEY_HIER":
        return "> *No API key provided. LLM analysis skipped.*"
    
    # Roughly calculate the months for the prompt (21 trading days = 1 month)
    horizon_months = max(1, horizon_days // 21)
    
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    You are a quantitative macro-analyst at a hedge fund. My model for predicting 
    the market state (Up, Down, Flat) for {target_etf} (Horizon: {horizon_months} months) has 
    selected and weighted the following predictors based on a stepwise variable selection 
    (Suffixes at the end show the momentum window or feature type, e.g., _126D_ret, _126D_diff, _Dist_SMA200, _Level):
    
    {coeff_df_string}
    
    Provide a highly elaborate but extremely precise economic assessment of 
    why this specific set of indicators currently acts as a leading indicator. 
    
    Rules for the output:
    - Absolute ban on continuous text and rambling. 
    - Answer exclusively in crisp bullet points.
    - Use hard, institutional logic (correlations, yield curve, sector rotations).
    
    Structure your answer strictly into these three short blocks:
    **1. Macroeconomic Setup:** (Why were interest rates/currencies/commodities chosen or ignored?)
    **2. Sector & Market Dynamics:** (What do the selected equities/sectors reveal about the business cycle?)
    **3. Quant Conclusion:** (What is the overarching narrative for {target_etf} in the next {horizon_months} months?)
    """
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            error_msg = str(e).upper()
            if any(err in error_msg for err in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                if attempt < max_retries - 1:
                    print(f"Google Server-Spike (503/429). Attempt {attempt+1}/{max_retries} failed. Waiting {delay} seconds...")
                    time.sleep(delay)
                    continue
            return f"> *Error during LLM request: {e}*"

# --- NEW FUNCTION: DUAL CUTOFF ---
def apply_dual_cutoffs(probs, classes, down_cutoff, up_cutoff):
    """
    Applies strict cutoffs for downside risk and the upside hurdle.
    Priority:
    1. If prob_down >= down_cutoff -> DOWN (-1)
    2. If prob_up >= up_cutoff -> UP (1)
    3. Else -> FLAT (0)
    """
    adjusted_preds = np.zeros(len(probs), dtype=int)
    
    down_idx = list(classes).index(-1) if -1 in classes else -1
    up_idx = list(classes).index(1) if 1 in classes else -1
    
    for i in range(len(probs)):
        p_down = probs[i, down_idx] if down_idx != -1 else 0
        p_up = probs[i, up_idx] if up_idx != -1 else 0
        
        # 1. Emergency brake: Downside risk too high
        if p_down >= down_cutoff:
            adjusted_preds[i] = -1
        # 2. High hurdle for Up: Only with extreme certainty
        elif p_up >= up_cutoff:
            adjusted_preds[i] = 1
        # 3. No extremes: The market runs sideways (Flat)
        else:
            adjusted_preds[i] = 0
            
    return adjusted_preds
# ----------------------------------

# IMPORTANT: The signature has received a new parameter 'up_cutoff_value'!
def perform_feature_selection(X_scaled, y, latest_features_scaled, target_etf, horizon, final_features=12, pre_filter_k=80, timestamp=None, up_cutoff_value=0.65):
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] MODELING: Executing Feature Selection (Pre-Filter: Top {pre_filter_k} variables)...")
    
    # Class weighting
    custom_weights = calculate_smoothed_weights(y, smoothing='log')
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] MODELING: Dynamic Algorithm Weighting (Log-Smoothed): {custom_weights}")
    
    k_actual = min(pre_filter_k, X_scaled.shape[1])
    kbest = SelectKBest(score_func=f_classif, k=k_actual)
    
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] MODELING: Starting SelectKBest (ANOVA)...")
    kbest.fit(X_scaled, y)
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] MODELING: SelectKBest complete.")
    
    features_stage_1 = X_scaled.columns[kbest.get_support()]
    rejected_stage_1 = X_scaled.columns[~kbest.get_support()]
    X_stage_1 = X_scaled[features_stage_1]
    
    initial_train_size = 252 * 3 
    if initial_train_size >= len(X_stage_1):
        initial_train_size = len(X_stage_1) // 2
        
    test_size = (len(X_stage_1) - initial_train_size - (5 * horizon)) // 5
    if test_size < 1:
        test_size = 1

    tscv = TimeSeriesSplit(n_splits=5, gap=horizon, max_train_size=None, test_size=test_size)
    
    log_reg_base = LogisticRegression(solver='lbfgs', max_iter=200, class_weight=custom_weights)
    
    sfs = SequentialFeatureSelector(log_reg_base, n_features_to_select=final_features, direction='forward', cv=tscv, n_jobs=-1)
    
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] MODELING: Starting SequentialFeatureSelector (SFS). This may take a few minutes...")
    sfs.fit(X_stage_1, y)
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] MODELING: SFS complete.")
    
    selected_features = features_stage_1[sfs.get_support()]
    rejected_stage_2 = features_stage_1[~sfs.get_support()]
    X_optimal = X_scaled[selected_features]
    
    model = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight=custom_weights)
    
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] MODELING: Fitting final Logistic Regression...")
    model.fit(X_optimal, y)
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] MODELING: Final fit complete. Starting OOF Evaluation...")

    # 1. Out-of-Fold (OOF) Evaluation
    oof_preds = np.full(len(y), np.nan)
    oof_probs = np.full((len(y), len(model.classes_)), np.nan)
    
    for train_idx, test_idx in tscv.split(X_optimal):
        X_train, y_train = X_optimal.iloc[train_idx], y.iloc[train_idx]
        X_test = X_optimal.iloc[test_idx]
        
        fold_model = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight=custom_weights)
        fold_model.fit(X_train, y_train)
        
        oof_preds[test_idx] = fold_model.predict(X_test)
        oof_probs[test_idx] = fold_model.predict_proba(X_test)
        
    valid_indices = ~np.isnan(oof_preds)
    y_valid = y.iloc[valid_indices]
    oof_probs_valid = oof_probs[valid_indices]
    
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] MODELING: OOF Evaluation complete.")

    # 2. KS-Statistic (Cutoff optimization for class -1)
    y_down_true = (y_valid == -1).astype(int)
    optimal_down_threshold = 0.33 
    
    if -1 in model.classes_:
        down_idx = list(model.classes_).index(-1)
        y_down_prob = oof_probs_valid[:, down_idx]
        
        fpr, tpr, thresholds = roc_curve(y_down_true, y_down_prob)
        ks_stats = tpr - fpr
        max_ks_idx = np.argmax(ks_stats)
        optimal_down_threshold = thresholds[max_ks_idx]
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] MODELING: KS-Statistic optimized: Down-Cutoff is at {optimal_down_threshold:.2%}")

    # ==========================================
    # NEW: Apply strict hurdle for Up
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] MODELING: Strict Up-hurdle (Cutoff) set to {up_cutoff_value:.2%}")
    # ==========================================

    # 3. Matrix synchronization with the DUAL-Cutoff
    oof_preds_dual = apply_dual_cutoffs(oof_probs_valid, model.classes_, optimal_down_threshold, up_cutoff_value)
    
    train_probs = model.predict_proba(X_optimal)
    train_preds_dual = apply_dual_cutoffs(train_probs, model.classes_, optimal_down_threshold, up_cutoff_value)

    # 4. Quality Gate 
    cv_accuracy = accuracy_score(y_valid, oof_preds_dual)
    is_valid_quality = cv_accuracy >= 0.35

    def plot_advanced_cm(y_true, y_pred, classes, title, accuracy):
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm_perc = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums!=0)
        
        labels = [f"{v1}\n({v2:.1%})" for v1, v2 in zip(cm.flatten(), cm_perc.flatten())]
        labels = np.asarray(labels).reshape(cm.shape)
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes, ax=ax, cbar=False,
                    annot_kws={"size": 11, "weight": "bold"})
        
        ax.set_ylabel('Actual Market (True)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_xlabel('Model Forecast (Predicted)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title(f"{title}\nAccuracy: {accuracy:.2%}", fontsize=14, fontweight='bold', pad=15)
        
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11, rotation=0)
        fig.patch.set_alpha(0.0) 
        plt.tight_layout()
        plt.close(fig)
        return fig

    train_accuracy = accuracy_score(y, train_preds_dual)
    fig_cm_train = plot_advanced_cm(y, train_preds_dual, model.classes_, "1. In-Sample (Training)", train_accuracy)
    fig_cm_cv = plot_advanced_cm(y_valid, oof_preds_dual, model.classes_, "2. Out-of-Sample (Cross-Validation)", cv_accuracy)

    # 5. Live Predict Logic (Dual Cutoff applied)
    X_latest_optimal = latest_features_scaled[selected_features]
    probabilities = model.predict_proba(X_latest_optimal)[0]
    
    prob_dict = dict(zip(model.classes_, probabilities))
    prob_down = prob_dict.get(-1, 0)
    prob_up = prob_dict.get(1, 0)
    
    # --- Live Predict Dual Cutoff Logic ---
    if prob_down >= optimal_down_threshold:
        prediction = -1
    elif prob_up >= up_cutoff_value:
        prediction = 1
    else:
        prediction = 0 # Default zu Flat
    # --------------------------------------

    class_mapping = {-1: "Down", 0: "Flat", 1: "Up"}
    pred_label = class_mapping.get(prediction, "Unknown")
    
    predict_date = latest_features_scaled.index[0]
    predict_date_str = predict_date.strftime('%Y-%m-%d') if isinstance(predict_date, pd.Timestamp) else str(predict_date)
    
    prob_flat = prob_dict.get(0, 0)

    print("\n" + "="*35)
    print("=== Current Model Forecast ===")
    print("="*35)
    print(f"Date: {predict_date_str}")
    print(f"Forecast: {pred_label}")
    print(f"Probabilities: Down={prob_down:.2%}, Flat={prob_flat:.2%}, Up={prob_up:.2%}\n")

    # IMPORTANT: Calculation of feature weights for export
    importance = np.mean(np.abs(model.coef_), axis=0)
    feature_names = selected_features.tolist()
    feature_weights = {feat: float(weight) for feat, weight in zip(feature_names, importance)}
    
    coeff_df = pd.DataFrame({
        'Predictor': selected_features,
        'Influence (Mean Absolute)': importance
    }).sort_values(by='Influence (Mean Absolute)', ascending=False)
    
    if timestamp:
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] MODELING: Fetching economic interpretation from LLM...")
        llm_analysis = get_llm_interpretation(coeff_df.to_string(index=False), target_etf, horizon)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        output_dir = os.path.join(project_root, 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        md_path = os.path.join(output_dir, f"feature_selection_{timestamp}.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# ETF Predictor Pipeline-Report\n\n")
            f.write(f"- **Generated at:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Target ETF:** {target_etf}\n")
            f.write(f"- **Forecast Horizon:** {horizon} Trading Days\n\n")
            
            f.write("## Current Market Forecast (Predict)\n\n")
            f.write(f"Based on closing prices from **{predict_date_str}**, the model predicts:\n\n")
            f.write(f"> **Class:** {pred_label}\n>\n")
            f.write(f"> **Probabilities:** Down: {prob_down:.2%} | Flat: {prob_flat:.2%} | Up: {prob_up:.2%}\n\n")
            f.write("---\n\n")
            
            f.write("## Selected Predictors (SFS)\n\n")
            f.write("| Predictor | Influence (Mean Absolute) |\n")
            f.write("| :--- | :--- |\n")
            for _, row in coeff_df.iterrows():
                f.write(f"| {row['Predictor']} | {row['Influence (Mean Absolute)']:.6f} |\n")
            f.write("\n")
            
            f.write("## Discarded Predictors\n\n")
            f.write("### 1. Discarded in final selection (SFS Rejects)\n")
            f.write(f"`{', '.join(rejected_stage_2.tolist())}`\n\n")
            
            f.write("### 2. Discarded in base filter (ANOVA Rejects)\n")
            f.write(f"<details>\n<summary>Click to view all <b>{len(rejected_stage_1)}</b> variables discarded in stage 1</summary>\n\n")
            f.write(f"`{', '.join(rejected_stage_1.tolist())}`\n")
            f.write("\n</details>\n\n")
            f.write("---\n\n")
            
            f.write("## AI Interpretation of Predictors (Hedge Fund Analyst)\n\n")
            f.write(llm_analysis + "\n\n")
            
            f.write("## Mathematical Model Parameters\n\n")
            f.write(f"- **Intercepts:** `{model.intercept_.tolist()}`\n\n")
            f.write("- **Coefficient Matrix:**\n")
            f.write("  ```text\n")
            f.write(str(model.coef_))
            f.write("\n  ```\n")
            
        try:
            generate_variable_audit_table(
                X_columns=X_scaled.columns, 
                p_values=kbest.pvalues_, 
                selected_features=selected_features, 
                rejected_stage_2=rejected_stage_2, 
                model_coefs=model.coef_,
                timestamp=timestamp
            )
        except Exception as e:
            print(f"Error during audit generation: {e}")
            
    return {
        "model": model,
        "X_optimal": X_optimal,
        "prediction": prediction,
        "probabilities": prob_dict,
        "cm_fig_train": fig_cm_train,
        "cm_fig_cv": fig_cm_cv,
        "cv_accuracy": cv_accuracy,
        "is_valid_quality": is_valid_quality,
        "ks_cutoff": optimal_down_threshold,
        "selected_features": feature_names,
        "feature_weights": feature_weights
    }