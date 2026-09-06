"""Check required original-analysis inputs without reading their records."""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-root',type=Path);a=p.parse_args()
package=Path(__file__).resolve().parents[1];root=a.input_root or package
spec=json.loads((package/'data/INPUTS.json').read_text());missing=[]
for item in spec['required_inputs']:
    path=root/item['path'];ok=path.exists() and (path.is_dir() or path.stat().st_size>0)
    if not ok:missing.append(item['path'])
print(json.dumps({'scope':spec['scope'],'required_inputs':len(spec['required_inputs']),'missing':missing,'note':'Presence check only; does not validate access rights, schemas, or scientific equivalence.'},indent=2))
raise SystemExit(2 if missing else 0)
