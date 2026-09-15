import json
from pathlib import Path

path = Path(r'c:\Users\Asus\Desktop\GNN project\GNN_CVD_Project_Notebook.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
cell = nb['cells'][3]
src = cell['source']
for i, line in enumerate(src):
    if line.strip() == 'from xgboost import XGBClassifier':
        src[i:i+1] = [
            'try\n',
            '    from xgboost import XGBClassifier\n',
            'except ModuleNotFoundError:\n',
            '    import subprocess\n',
            '    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])\n',
            '    from xgboost import XGBClassifier\n',
        ]
        break
else:
    raise SystemExit('xgboost import line not found')
cell['source'] = src
path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('Patched xgboost import cell successfully')
