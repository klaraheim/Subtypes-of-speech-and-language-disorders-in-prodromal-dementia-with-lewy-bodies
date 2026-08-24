# Extrakce kognitivních dat (z akustického .csv datového souboru)

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(SCRIPT_DIR, "acoustic_residualized_filtered.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "cognitive_data.csv")

COLUMN_MAPPING = {
    'Subject': 'Subject',
    'MOCA': 'MOCA',
    'education type': 'education_length',
    'memory z-score': 'memory_zscore',
    'visuo-spatial z-score': 'visuospatial_zscore',
    'attention z-score': 'attention_zscore',
    'executive function z-score': 'executive_zscore',
    'GDS': 'GDS',
    'Category': 'Category'
}

def load_and_extract():
    
    if not os.path.exists(INPUT_FILE):
        return None
    
    df = pd.read_csv(INPUT_FILE)
    
    
    missing_cols = []
    available_cols = []
    
    for old_name, new_name in COLUMN_MAPPING.items():
        if old_name in df.columns:
            available_cols.append(old_name)
        else:
            missing_cols.append(old_name)
    
    df_cognitive = df[available_cols].copy()
    
    df_cognitive = df_cognitive.rename(columns=COLUMN_MAPPING)
    
    
    for col in df_cognitive.columns:
        if col == 'Category':
            continue
        
        n_missing = df_cognitive[col].isna().sum()
        pct_missing = n_missing / len(df_cognitive) * 100
        
        _ = n_missing
    
    return df_cognitive

def save_data(df_cognitive):
    
    df_cognitive.to_csv(OUTPUT_FILE, index=False)
    print(f" Uloženo: {OUTPUT_FILE}")
    
def main():
    df_cognitive = load_and_extract()
    
    if df_cognitive is None:
        print("\n Extrakce selhala!")
        return
    
    save_data(df_cognitive)
    
    print("HOTOVO!")

if __name__ == "__main__":
    main()
