import pandas as pd

df = pd.read_csv('hyperparameter_log.csv')
print(df.head())
print(f"Total runs: {len(df)}")