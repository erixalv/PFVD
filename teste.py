import pandas as pd

df = pd.read_csv("DADOS/resultados.csv", encoding="latin-1")

df = df.head(10)

df.to_csv("teste.csv")