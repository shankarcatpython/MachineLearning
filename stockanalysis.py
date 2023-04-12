import pandas as pd 
import matplotlib.pyplot as plt

d1 = pd.read_csv('final.csv')

d1=d1.drop(d1.columns[0], axis=1)
d1=d1.loc[d1['asOfDate'].str.contains('2019')]

d2=d1.dropna().drop_duplicates()

d2=d2[(d2["percentage"] > 65 ) & (d2["percentage"] < 85) ].sort_values(by="percentage", ascending=False)

d2.to_csv('temp.csv')

print(round(d2["percentage"]).count())

df2 = d1.groupby(round(d2["percentage"]))["percentage"].count()

df2.to_csv('temp1.csv')