import json
import random
import pandas as pd

base_path = 'd:/A_Facultate/Diseration/code/Dissertation-code/logit_lens'
df = pd.read_json(f'{base_path}/squad_curated.json')
indices = random.sample(range(len(df)), 5)

def load_preds(path):
    try:
        with open(path, encoding='utf-8') as f:
            return {int(p['id']): p['prediction_text'] for p in json.load(f)}
    except Exception as e:
        return {}

preds_xlstm = load_preds(f'{base_path}/squad_results/predictions.json')
preds_mistral = load_preds(f'{base_path}/squad_results_mistral/predictions.json')
preds_nocache = load_preds(f'{base_path}/squad_results_nocache/predictions.json')

for i in indices:
    row = df.iloc[i]
    print(f'=== Question ID: {i} ===')
    print(f'Context:\n\n{row["context"]}\n\nQuestion: {row["question"]}\nAnswer:')
    print(f'\n**xLSTM_Cache**:\n{preds_xlstm.get(i, "N/A")}')
    print(f'\n**Mistral**:\n{preds_mistral.get(i, "N/A")}')
    print(f'\n**xLSTM_NoCache**:\n{preds_nocache.get(i, "N/A")}')
    print('----------------------------------------')
