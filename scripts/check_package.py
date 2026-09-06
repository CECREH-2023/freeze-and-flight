"""Verify the distributed manifest without importing the research code."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
expected={}
for line in (root/'MANIFEST.sha256').read_text().splitlines():
    digest,name=line.split('  ',1);expected[name]=digest
errors=[]
for name,digest in expected.items():
    p=root/name
    if not p.is_file():errors.append({'file':name,'problem':'missing'})
    elif sha(p)!=digest:errors.append({'file':name,'problem':'hash mismatch'})
ignore={'.git','.venv','__pycache__','.pytest_cache','node_modules','.cache'}
extra=[]
for p in root.rglob('*'):
    if not p.is_file():continue
    name=p.relative_to(root).as_posix()
    if any(x in ignore for x in p.relative_to(root).parts) or name.startswith('results/generated/'):continue
    if name not in expected and name!='MANIFEST.sha256':extra.append(name)
if extra:errors.append({'problem':'unexpected files','files':extra})
result={'checksummed_files':len(expected),'errors':errors,'scope':'Distribution integrity only; no statistical estimation'}
print(json.dumps(result,indent=2));sys.exit(bool(errors))
