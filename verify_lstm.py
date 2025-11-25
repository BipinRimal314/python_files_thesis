import pandas as pd
import numpy as np
import os
from ensemble_grid_search_v2 import load_and_align_data

def main():
    # Reuse the robust loading logic from your working script
    data = load_and_align_data()
    if data[0] is None:
        print("Failed to load data.")
        return

    y_test, if_scores, dc_scores, lstm_scores = data
    
    # Create a DataFrame for easy viewing
    df = pd.DataFrame({
        'Label': y_test,
        'IF_Score': if_scores,
        'DC_Score': dc_scores,
        'LSTM_Score': lstm_scores
    })
    
    # Filter for the Insiders
    insiders = df[df['Label'] == 1]
    
    print("\n" + "="*50)
    print(f"ANALYSIS OF THE {len(insiders)} INSIDERS")
    print("="*50)
    print(insiders)
    
    print("\n" + "="*50)
    print("AVERAGE SCORES (Normal vs Insider)")
    print("="*50)
    print(df.groupby('Label').mean())
    
    print("\n[INTERPRETATION]")
    print("If LSTM_Score is significantly higher for Label=1 than Label=0,")
    print("and the gap is wider than IF/DC, that explains the 0.8 weight.")

if __name__ == "__main__":
    main()