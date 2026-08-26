import pandas as pd
import os
from data_cleaning import load_and_clean


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "European_Bank.csv")
df = load_and_clean(DATA_PATH)

# continue with segmentation...

df['AgeGroup'] = pd.cut(df['Age'],
                       bins=[18,30,45,60,100],
                       labels=['<30','30-45','46-60','60+'])

# Balance Segments
def balance_seg(x):
    if x == 0: return 'Zero'
    elif x < 100000: return 'Low'
    else: return 'High'

df['BalanceSegment'] = df['Balance'].apply(balance_seg)

if __name__ == "__main__":
    print(df.head())