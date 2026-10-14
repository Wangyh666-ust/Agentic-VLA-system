import copy
import runpy
from scene_demo import catalog
ns = runpy.run_path('/mnt/d/FYP/First_Phase/scene_demo/mcp_server.py', run_name='software_import_probe')
f = ns['_atomic_capabilities']
for scene, cid in [('basket_two','xvla_basket_both'),('mugs_two','xvla_mugs_both'),('microwave_mug','xvla_microwave_place_close')]:
    public = catalog.scene_capabilities(scene, backend='xvla')
    assert cid in [c['capability_id'] for c in f(public)]
    native = next(c for c in public if c['capability_id'] == cid)
    forged = dict(native, capability_id='forged')
    assert f([forged]) == []
    assert f([dict(native, audit_only=True)]) == []
    print('visible:', scene, cid, '; audit/forged: rejected')
for scene in catalog.SCENES:
    for capability in catalog.scene_capabilities(scene, include_audit=True, backend='xvla'):
        if capability['audit_only']:
            assert f([capability]) == []
print('MCP_RUNTIME_IMPORT_AND_PURE_FILTER_OK; HTTP/VLA/Qwen/Hermes/MuJoCo actions: 0')
