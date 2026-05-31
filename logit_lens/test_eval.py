import pandas as pd
from datasets import load_dataset
squad = load_dataset("rajpurkar/squad")
squad_df = squad["validation"].to_pandas()

max_questions = 100
grouped = squad_df.groupby(['title', 'context'])

titles_seen = set()
selected_groups = []
total_records = 0

for (title, context), group in grouped:
    if title not in titles_seen:
        if total_records + len(group) <= 100:
            selected_groups.append(group)
            total_records += len(group)
            titles_seen.add(title)
        
        if total_records == 100:
            break

if total_records < 100:
    for (title, context), group in grouped:
        if title not in titles_seen:
            if total_records + len(group) <= 100:
                selected_groups.append(group)
                total_records += len(group)
                titles_seen.add(title)
            if total_records == 100:
                break

if total_records < 100:
    needed = 100 - total_records
    for (title, context), group in grouped:
        if title not in titles_seen:
            selected_groups.append(group.head(needed))
            total_records += needed
            break

dataset = pd.concat(selected_groups)
print(f"Total records: {len(dataset)}")
print(f"Unique titles count: {dataset['title'].nunique()}")
print(f"Unique titles: {dataset['title'].unique()}")
