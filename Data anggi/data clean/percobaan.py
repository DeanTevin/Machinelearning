import pandas as pd
df = pd.read_csv("2019_jun_des.csv",sep=",")

print(df)

df_interpolate = df.interpolate()
print(df_interpolate)

df_interpolate.to_csv('2019_jun_des_clean.csv',sep=",")