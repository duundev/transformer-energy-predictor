import pandas as pd
import numpy as np
import requests
import os

# Baixa o dataset (CSV)
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv'
csv_path = 'energydata_complete.csv'

if not os.path.exists(csv_path):
    r = requests.get(url)
    with open(csv_path, 'wb') as f:
        f.write(r.content)
    print('Dataset baixado.')

# Lê o CSV
df = pd.read_csv(csv_path)

# Pega a coluna Appliances (consumo em Wh)
consumo = df['Appliances'].values

# Discretiza em tokens (ex: 10W por token)
step = 10
tokens = (consumo // step).astype(int)

# Salva tokens em arquivo texto (separado por espaço)
with open('data.txt', 'w') as f:
    f.write(' '.join(map(str, tokens)))

print(f'Dados processados e salvos em data.txt ({len(tokens)} tokens)')
