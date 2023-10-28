import pandas as pd

df = pd.read_csv('../cs_combo_bbox.csv')

df = df[df['Power'] == 150]

df.to_csv('cs_combo_150_bbox.csv', index=False)
