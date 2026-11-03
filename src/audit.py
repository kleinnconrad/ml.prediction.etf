# src/audit.py

import os
import json
import time
import pandas as pd
import numpy as np
import re
from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL

def fetch_ticker_metadata(base_tickers, max_retries=3, delay=60):
    """
    Uses the Gemini API to generate plain names and economic descriptions 
    for a list of base tickers as JSON.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "DEIN_API_KEY_HIER":
        print("No API key found for the audit script.")
        return {}

    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Format the list as a string for the prompt
    tickers_str = ", ".join(base_tickers)
    
    prompt = f"""
    You are a financial data analyst. I am giving you a list of stock tickers. 
    Return exactly one JSON object for each ticker.
    
    Ticker list: [{tickers_str}]
    
    Rules for the JSON:
    The key must be the exact ticker. 
    The value is an object with 'name' (plain name of the asset/indicator) and 'desc' (1 concise sentence explaining what it economically measures in the context of a market forecast).
    
    Example structure:
    {{
        "^TNX": {{"name": "US 10-Year Treasury Yield", "desc": "Measures the yield of 10-year US Treasury bonds; a key indicator for long-term interest rate and inflation expectations."}},
        "AAPL": {{"name": "Apple Inc.", "desc": "Largest technology company in the world; represents the strength of the broad US consumer and tech sector."}}
    }}
    
    IMPORTANT: Respond EXCLUSIVELY with pure JSON code. No Markdown blocks (like ```json), no introductions, no continuous text.
    """
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            
            # Clean up the output (in case the AI still sends Markdown ticks)
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            metadata = json.loads(raw_text.strip())
            return metadata
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error during audit (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if attempt < max_retries - 1:
                    print(f"API rate limit during audit. Waiting {delay}s...")
                    time.sleep(delay)
                    continue
            print(f"API error during audit: {e}")
            return {}
            
    return {}

def generate_variable_audit_table(X_columns, p_values, selected_features, rejected_stage_2, model_coefs, timestamp):
    """
    Compiles all statistical metrics and LLM descriptions into a Markdown table.
    """
    print("Starting in-depth variable audit (LLM query running)...")
    
    # 1. Build data structure
    audit_data = []
    
    # Mean Absolute of the coefficients for the final features (impact)
    importance = np.mean(np.abs(model_coefs), axis=0)
    coef_dict = dict(zip(selected_features, importance))
    
    for i, col in enumerate(X_columns):
        # Extract base ticker by removing the quantitative feature suffixes
        base_ticker = re.sub(r'(_(21|63|126|252)D_(ret|diff)|_Dist_SMA200|_YoY_Accel_3M|_Roll_ZScore_2Y|_Level)$', '', col)
        p_val = p_values[i]
        
        # Assign status and impact
        if col in selected_features:
            status = "🟢 Active (Selected)"
            einfluss = f"{coef_dict[col]:.4f}"
        elif col in rejected_stage_2:
            status = "🟡 Rejected (Multicollinearity)"
            einfluss = "-"
        else:
            status = "🔴 Rejected (No Significance)"
            einfluss = "-"
            
        audit_data.append({
            "Variable": col,
            "Base_Ticker": base_ticker,
            "Status": status,
            "p_Value": p_val,
            "Impact": einfluss
        })
        
    df_audit = pd.DataFrame(audit_data)
    
    # 2. Fetch metadata via Gemini API (request unique tickers only)
    unique_tickers = df_audit['Base_Ticker'].unique().tolist()
    metadata = fetch_ticker_metadata(unique_tickers)
    
    # 3. Match LLM data with statistics
    df_audit['Plain Name'] = df_audit['Base_Ticker'].apply(lambda x: metadata.get(x, {}).get('name', 'Unknown'))
    df_audit['Description'] = df_audit['Base_Ticker'].apply(lambda x: metadata.get(x, {}).get('desc', 'No description available.'))
    
    # 4. Sort table (Active features first, then by p-value)
    # Custom sorting logic via an auxiliary column
    status_order = {"🟢 Active (Selected)": 0, "🟡 Rejected (Multicollinearity)": 1, "🔴 Rejected (No Significance)": 2}
    df_audit['Sort'] = df_audit['Status'].map(status_order)
    df_audit = df_audit.sort_values(by=['Sort', 'p_Value']).drop(columns=['Sort', 'Base_Ticker'])
    
    # Format p-values nicely (Scientific notation for very small numbers)
    df_audit['p_Value'] = df_audit['p_Value'].apply(lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}")
    
    # 5. Export Markdown table
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    output_dir = os.path.join(project_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    md_path = os.path.join(output_dir, f"variable_audit_{timestamp}.md")
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Variable Audit (Feature Encyclopedia)\n\n")
        f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("> This document logs all variables of the pipeline. It transparently explains "
                "which variables actively make predictions, which were ignored by the algorithm due to redundant information (multicollinearity), "
                "and which variables showed no statistical relevance (ANOVA F-test).\n\n")
        
        f.write("| Variable | Plain Name | Status | p-Value (ANOVA) | Impact (Model) | Economic Description |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for _, row in df_audit.iterrows():
            f.write(f"| **{row['Variable']}** | {row['Plain Name']} | {row['Status']} | {row['p_Value']} | {row['Impact']} | {row['Description']} |\n")
            
    print(f"Variable audit successfully saved at: {md_path}")