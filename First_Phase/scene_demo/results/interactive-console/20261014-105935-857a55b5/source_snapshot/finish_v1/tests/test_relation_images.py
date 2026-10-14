import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from scene_demo.finish_v1.relation_images import prepare_relation_images
from scene_demo.finish_v1 import vision
from scene_demo.finish_v1.tests.test_joint_verifier import RelationReviewTests

class RelationImagesTests(unittest.TestCase):
    def test_pixel_provenance_and_bad_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);obs=[]
            for t in (1.,2.):
                frames=[]
                for v in ('a','b'):
                    p=root/f'{t}_{v}.png';im=Image.new('RGB',(30,20));im.putdata([(x%255,y%255,(x+y)%255) for y in range(20) for x in range(30)]);im.save(p)
                    frames.append({'view':v,'rgb_path':str(p)})
                obs.append({'timestamp':t,'frames':frames})
            raw={'boxes':{v:{'target':[.2,.2,.4,.4],'destination':[.3,.3,.5,.5],'gripper':None} for v in ('a','b')}}
            before={f['rgb_path']:hashlib.sha256(Path(f['rgb_path']).read_bytes()).hexdigest() for o in obs for f in o['frames']}
            out=prepare_relation_images(obs,raw,root/'details')
            self.assertTrue(out['ready']);self.assertEqual(len(out['image_paths']),6)
            self.assertEqual(out['views'],['a','b']);self.assertEqual(out['distinct_timestamps'],2)
            for d in out['derivative_provenance']:
                self.assertEqual(d['timestamp'],2.);self.assertIn(d['source_frame'],[2,3])
                with Image.open(d['source_path']) as source,Image.open(d['crop_path']) as crop:self.assertEqual(source.crop(tuple(d['pixel_bounds'])).tobytes(),crop.tobytes())
            self.assertEqual(before,{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before})
            turn=copy.deepcopy(raw)
            for boxes in turn['boxes'].values(): boxes['destination']=None
            turned=prepare_relation_images(obs,turn,root/'turn',operation='turn')
            self.assertTrue(turned['ready'])
            self.assertFalse(prepare_relation_images(obs,turn,root/'place_missing')['ready'])
            for mode in ('missing','bad'):
                bad=copy.deepcopy(raw)
                if mode=='missing':bad['boxes']['a']['target']=None
                else:bad['boxes']['a']['target']=[-1,0,1,1]
                output=root/mode;r=prepare_relation_images(obs,bad,output)
                self.assertFalse(r['ready']);self.assertFalse(output.exists())

    def test_prompt_mapping_rejects_wrong_view_time_and_hidden(self):
        test=RelationReviewTests();test.setUp()
        try:
            ctx,_=test.eligible();ctx['sensor_frames'] += [{'frame':4,'view':'agentview','timestamp':1.},{'frame':5,'view':'wrist','timestamp':1.}]
            d={'frame':4,'source_frame':2,'view':'agentview','timestamp':1.,'source_size_px':[100,100],'pixel_bounds':[10,20,70,80],'original_roi':[.1,.2,.7,.8]};ctx['relation_derivatives']=[d]
            self.assertIn('original_roi',vision.build_relation_prompt(ctx))
            for key,val in [('view','bad'),('timestamp',2.),('gold',True)]:
                bad=copy.deepcopy(ctx);bad['relation_derivatives'][0][key]=val
                with self.assertRaises(ValueError):vision.build_relation_prompt(bad)
        finally:test.doCleanups()

class NotReadyRuntimeTests(RelationReviewTests):
    def test_not_ready_no_extra_call(self):
        mon=self.relation_monitor()
        from unittest import mock
        with mock.patch('scene_demo.finish_v1.relation_images.prepare_relation_images',return_value={'ready':False,'reason':'missing view'}):
            mon._check(mon.context,[])
        self.assertEqual(mon.calls,1);self.assertEqual(mon._last_decision['state'],'unknown');self.assertEqual(mon.port.steps,0)

if __name__=='__main__':unittest.main()
