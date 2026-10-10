"""Pure saved-image detail preparation; no robot or model calls."""
import hashlib
import json
import math
from pathlib import Path


def prepare_relation_images(observations, initial_verdict, output_dir, operation="place"):
    empty = dict(ready=False, reason="insufficient relation inputs", image_paths=[], sensor_frames=[], derivative_provenance=[], latest_timestamp=None, original_image_count=0, crop_image_count=0, distinct_timestamps=0, views=[])
    if operation not in ('place', 'turn'):
        raise ValueError('operation must be place or turn')
    try:
        by_time = {float(o['timestamp']): o for o in observations}
        times = sorted(by_time)[-2:]
        if len(times) != 2 or not all(math.isfinite(t) for t in times): return empty
        def frames(o): return o.get('verification_frames') or o.get('frames') or []
        latest = frames(by_time[times[-1]])
        views = [f['view'] for f in latest[:2]]
        if len(views) != 2 or len(set(views)) != 2: return empty
        originals = []
        boxes = initial_verdict['boxes']
        rois = []
        for view in views:
            pair = [boxes[view][key] for key in (('target', 'destination') if operation == 'place' else ('target',))]
            for box in pair:
                if not isinstance(box, list) or len(box) != 4 or any(isinstance(x, bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in box): return empty
                if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1): return empty
            x1,y1=min(b[0] for b in pair),min(b[1] for b in pair)
            x2,y2=max(b[2] for b in pair),max(b[3] for b in pair)
            px,py=max(.10,.25*(x2-x1)),max(.10,.25*(y2-y1))
            rois.append([max(0,x1-px),max(0,y1-py),min(1,x2+px),min(1,y2+py)])
        for t in times:
            fs = frames(by_time[t]); mapping = {f['view']: f for f in fs}
            if any(v not in mapping for v in views): return empty
            originals.extend((t,v,Path(mapping[v]['rgb_path'])) for v in views)
        from PIL import Image
        loaded = [Image.open(p).copy() for _,_,p in originals]
        output_dir = Path(output_dir)
        paths = [output_dir / ('detail_%d.png' % i) for i in range(2)]
        if any(p.exists() for p in paths): raise FileExistsError('Refusing to overwrite relation detail')
        provenance = []
        candidate_sha = hashlib.sha256(json.dumps(boxes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        output_dir.mkdir(parents=True,exist_ok=True)
        for i,(roi,dest) in enumerate(zip(rois,paths)):
            source=originals[2+i][2]; image=loaded[2+i]; w,h=image.size
            bounds=[math.floor(roi[0]*w),math.floor(roi[1]*h),math.ceil(roi[2]*w),math.ceil(roi[3]*h)]
            crop=image.crop(tuple(bounds));crop.save(dest,format='PNG')
            provenance.append(dict(frame=4+i,source_frame=2+i,view=views[i],timestamp=times[-1],source_path=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_size_px=[w,h],pixel_bounds=bounds,original_roi=[bounds[0]/w,bounds[1]/h,bounds[2]/w,bounds[3]/h],crop_path=str(dest),crop_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),crop_size_px=list(crop.size),candidate_boxes_source_sha256=candidate_sha))
        image_paths=[str(p) for _,_,p in originals]+[str(p) for p in paths]
        sensor_frames=[dict(frame=i,view=v,timestamp=t) for i,(t,v,_) in enumerate(originals)]+[dict(frame=4+i,view=views[i],timestamp=times[-1]) for i in range(2)]
        return dict(ready=True,reason='four original frames plus two latest-view crops',image_paths=image_paths,sensor_frames=sensor_frames,derivative_provenance=provenance,latest_timestamp=times[-1],original_image_count=4,crop_image_count=2,distinct_timestamps=2,views=views)
    except (KeyError,ValueError,TypeError,IndexError):
        return empty
