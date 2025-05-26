import pandas as pd
obs = pd.read_parquet('processed/obs_table.parquet')
print(obs.shape)
print(obs.columns.tolist())
