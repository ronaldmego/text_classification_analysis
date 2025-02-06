import pandas as pd

# Leer el CSV
df = pd.read_csv('data/mind_news_classification.csv')

# Tomar los primeros 100 registros
df_100 = df.head(100)

# Opcional: Guardar los 100 registros en un nuevo CSV
df_100.to_csv('data/mind_news_classification_100.csv', index=False)