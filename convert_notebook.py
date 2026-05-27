"""Convert .py notebook to .ipynb format for Colab."""
import json

with open('notebooks/colab_training_all_models.py', 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
cells = []
current_cell = []
current_type = 'code'

for line in lines:
    if line.strip().startswith('# %% [markdown]'):
        if current_cell:
            cells.append((current_type, '\n'.join(current_cell)))
        current_cell = []
        current_type = 'markdown'
    elif line.strip() == '# %%':
        if current_cell:
            cells.append((current_type, '\n'.join(current_cell)))
        current_cell = []
        current_type = 'code'
    else:
        if current_type == 'markdown':
            if line.startswith('# '):
                current_cell.append(line[2:])
            elif line.startswith('#'):
                current_cell.append(line[1:].lstrip() if len(line) > 1 else '')
            else:
                current_cell.append(line)
        else:
            current_cell.append(line)

if current_cell:
    cells.append((current_type, '\n'.join(current_cell)))

nb = {
    'nbformat': 4,
    'nbformat_minor': 0,
    'metadata': {
        'colab': {'provenance': [], 'name': 'Stock_Market_AI_Training.ipynb'},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'language_info': {'name': 'python'}
    },
    'cells': []
}

for cell_type, source in cells:
    source = source.strip()
    if not source:
        continue
    cell = {
        'cell_type': cell_type,
        'metadata': {},
        'source': [line + '\n' for line in source.split('\n')]
    }
    if cell['source']:
        cell['source'][-1] = cell['source'][-1].rstrip('\n')
    if cell_type == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
    nb['cells'].append(cell)

with open('notebooks/Stock_Market_AI_Training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Created Stock_Market_AI_Training.ipynb with", len(nb['cells']), "cells")
