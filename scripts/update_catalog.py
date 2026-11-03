# scripts/update_catalog.py

import os
import sys
from google import genai

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    # File paths
    config_path = 'src/config.py'
    pipeline_path = 'src/data_pipeline.py'
    catalog_path = 'docs/var_catalog.md'

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_content = f.read()
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            pipeline_content = f.read()
        with open(catalog_path, 'r', encoding='utf-8') as f:
            catalog_content = f.read()
    except FileNotFoundError as e:
        print(f"Error reading files: {e}")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # =========================================================================
    # UPGRADED PROMPT: Institutional-Grade Requirements
    # =========================================================================
    prompt = f"""
    You are a Quantitative Financial Analyst and Technical Documenter for an institutional trading pipeline.
    Your task is to compare the variables defined in the codebase with the existing markdown table in var_catalog.md.
    
    Identify any variable missing from the markdown table by cross-referencing:
    1. Financial tickers and indicators in the lists of config.py.
    2. Engineered interaction ratios in data_pipeline.py (look for the string names passed to the `safe_ratio` function, e.g., 'ratio_copper_gold').

    Generate ONLY the missing table rows in this exact markdown format:
    | `ticker_1M_ret`, `ticker_3M_ret`, `ticker_6M_ret` | Full Name | Description | Category | Source |

    STRICT CONTENT RULES FOR COLUMNS:
    - **Full Name:** You MUST provide the actual, professional name of the asset or ratio (e.g., "Toyota Motor Corp.", "Copper / Gold Ratio", "10-Year Minus 2-Year Treasury Yield Spread"). Do NOT just repeat the ticker. Use your financial knowledge and the inline code comments to figure this out.
    - **Description:** Provide a highly specific, institutional-grade macroeconomic explanation. What does this measure? Why is it relevant for forecasting a broad market index? (e.g., "The Growth/Inflation Engine. Measures industrial expansion against safe-haven hoarding." or "Systemic Japanese proxy for global automotive manufacturing."). ABSOLUTELY DO NOT write lazy generic text like "1-month percentage returns for X".
    - **Category:** Group them logically based on the list names in config.py (e.g., "MACRO INDICATORS", "SYSTEMIC US EQUITIES", "FRED MACROECONOMIC INDICATORS"). For ratios in data_pipeline.py, use "ENGINEERED INTERACTIONS".
    - **Source:** State "Yahoo Finance", "FRED", or "Pipeline Transformation" (for engineered ratios).

    STRICT FORMATTING RULES:
    - For config.py tickers: The code abbreviation must be lowercase and stripped of special characters (e.g., ^TNX becomes tnx, CL=F becomes cl).
    - For data_pipeline.py ratios: Use the exact string name from the safe_ratio call (e.g., ratio_copper_gold) and append _1M_ret, _3M_ret, _6M_ret.
    - Do not output table headers. Do not output markdown codeblock backticks (```).
    - If there are no missing variables, output the exact string "NO_MISSING_VARIABLES".

    === config.py ===
    {config_content}
    
    === data_pipeline.py ===
    {pipeline_content}

    === var_catalog.md ===
    {catalog_content}
    """
    # =========================================================================

    print("Checking for missing catalog entries...")
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash-lite',  # same model as GEMINI_MODEL in src/config.py
            contents=prompt,
        )
        new_rows = response.text.strip()
        
        # Clean up any rogue markdown code block tags the LLM might have generated
        new_rows = new_rows.replace("```markdown", "").replace("```", "").strip()

        if new_rows and "NO_MISSING_VARIABLES" not in new_rows:
            with open(catalog_path, 'a', encoding='utf-8') as f:
                if not catalog_content.endswith('\n'):
                    f.write('\n')
                f.write(new_rows + '\n')
            print("Successfully appended new variables to var_catalog.md:")
            print(new_rows)
        else:
            print("Catalog is already up to date. No changes made.")

    except Exception as e:
        print(f"Error during LLM generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()