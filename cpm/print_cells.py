import json

with open('GNN_CVD_Project_Notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total notebook cells: {len(nb['cells'])}")
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
cells_with_outputs = [c for c in code_cells if len(c.get('outputs', [])) > 0]
print(f"Code cells with outputs: {len(cells_with_outputs)} / {len(code_cells)}")

print("\nPhase 11 Cells:")
for idx, c in enumerate(nb['cells'][74:], start=75):
    c_type = c['cell_type'].upper()
    n_out = len(c.get('outputs', []))
    title = c['source'][0].strip() if c['source'] else ''
    print(f"  Cell #{idx:02d} [{c_type:8s}] Outputs: {n_out:d} | {title[:55]}")
