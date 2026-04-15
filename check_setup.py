import pandas as pd
import glob, os, torch

print("=" * 50)
print("CHECK 1 — Excel files")
print("=" * 50)
files = glob.glob('data/raw/**/*.xlsx', recursive=True)
print(f'Found {len(files)} Excel files:')
for f in files:
    size = os.path.getsize(f)
    try:
        df = pd.read_excel(f)
        print(f'  {f}')
        print(f'    size={size} bytes | shape={df.shape}')
        print(f'    cols={list(df.columns[:8])}')
    except Exception as e:
        print(f'  {f}: ERROR - {e}')

print()
print("=" * 50)
print("CHECK 2 — __init__.py files")
print("=" * 50)
folders = [
    'backend', 'backend/api', 'backend/models',
    'backend/services', 'backend/utils', 'training'
]
for folder in folders:
    init = os.path.join(folder, '__init__.py')
    exists = os.path.exists(init)
    status = "OK" if exists else "MISSING"
    print(f'  {init}: {status}')

print()
print("=" * 50)
print("CHECK 3 — Quarter folder names")
print("=" * 50)
quarters = glob.glob('data/raw/interbank_dataset/*/')
print(f'Quarters found: {len(quarters)}')
for q in sorted(quarters):
    files_in_q = os.listdir(q)
    print(f'  {q.strip(os.sep).split(os.sep)[-1]} -> {files_in_q}')

print()
print("=" * 50)
print("CHECK 4 — Existing graph_data.pt")
print("=" * 50)
try:
    data = torch.load('data/processed/graph_data.pt', weights_only=False)
    print(f'  Nodes:    {data.num_nodes}')
    print(f'  Edges:    {data.num_edges}')
    print(f'  Features: {data.x.shape[1]}')
    has_y = hasattr(data, "y") and data.y is not None
    print(f'  Labels:   {has_y}')
    if has_y:
        print(f'  y range:  [{data.y.min():.4f}, {data.y.max():.4f}]')
        print(f'  y std:    {data.y.std():.6f}')
except Exception as e:
    print(f'  ERROR: {e}')