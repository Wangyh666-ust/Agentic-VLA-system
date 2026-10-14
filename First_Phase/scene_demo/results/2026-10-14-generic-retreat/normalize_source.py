"""Normalize only edited source formatting, proving the Python AST is unchanged."""
import ast,hashlib,json
from pathlib import Path
out=Path(__file__).resolve().parent
scene=out.parents[1]
proof=[]
for name in ('finish_v1/geometry.py','finish_v1/tests/test_shape_screen.py','finish_v1/tests/test_generic_retreat.py'):
    path=scene/name
    before=path.read_bytes()
    text=before.decode('utf-8').replace('\r\n','\n')
    if name=='finish_v1/geometry.py':
        marker='    for dist, direction, source in candidates:\n'
        prefix,body=text.split(marker)
        body,tail=body.rsplit('    return fail',1)
        if body.startswith('            '):
            body=''.join(line[4:] if line.startswith('            ') else line for line in body.splitlines(keepends=True))
        text=prefix+marker+body+'    return fail'+tail
    assert ast.dump(ast.parse(before))==ast.dump(ast.parse(text)),'format change altered semantics'
    # Preserve geometry.py's existing Windows line endings to avoid a whole-file
    # diff. The other edited files use LF. Neither changes executable semantics.
    after=(text.replace('\n','\r\n') if name=='finish_v1/geometry.py' else text).encode('utf-8')
    path.write_bytes(after)
    proof.append({'file':name,'before_sha256':hashlib.sha256(before).hexdigest(),
                  'after_sha256':hashlib.sha256(after).hexdigest(),'AST_identical':True})
(out/'source_format_proof.json').write_text(json.dumps(proof,indent=2))
print('Edited source normalized; AST unchanged in all three files')
