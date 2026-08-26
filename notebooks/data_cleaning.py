import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "European_Bank.csv")

def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)
    df.drop(['CustomerId','Surname','Year'], axis=1, inplace=True)
    df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
    return df

# Keep Geography for EDA, encode later for ML
df = load_and_clean(DATA_PATH)

if __name__ == "__main__":
    print(df.head())
