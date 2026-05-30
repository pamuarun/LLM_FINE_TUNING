import nbformat
import glob

for path in glob.glob('*.ipynb'):
    with open(path, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    nb.nbformat_minor = 5
    for cell in nb.cells:
        cell['outputs'] = []
        cell['execution_count'] = None
    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f'Fixed: {path}')

print('All done!')