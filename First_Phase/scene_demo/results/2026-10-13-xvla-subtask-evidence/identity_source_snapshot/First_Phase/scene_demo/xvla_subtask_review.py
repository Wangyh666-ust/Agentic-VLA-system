"""Whole-task image comparison and retrospective release review; no control actions."""
import copy
import json
import math
from pathlib import Path
from scene_demo.xvla_finish import XvlaQwenClient
from scene_demo.xvla_evidence_memory import is_placement,STATES
from scene_demo.xvla_process_evidence import capture_frames
from scene_demo.finish_v1.simple_checker import SimpleQwenClient
from scene_demo.finish_v1 import vision
from scene_demo.xvla_subtask_evidence import SubtaskEvidence,PROTOCOL,MAX_IMAGES,digest,goal_key,overlap,build_prompt

def unknown(gid,why='not observed'):
    return {'id':gid,'state':'unknown','same_target':None,'evidence':[],
            'identity_refs':[],'supported':None,'change':'none','comparison':why}

class XvlaSubtaskReviewClient(XvlaQwenClient):
    request_options={'enable_thinking':True,'thinking_budget':1024,'max_tokens':4096}

    def __init__(self,*args,evidence_store=None,event_namespace='',**kwargs):
        super().__init__(*args,**kwargs)
        self.store=evidence_store or SubtaskEvidence();self._explicit_store=evidence_store is not None
        self.event_namespace=str(event_namespace)
        self._reference=None

    def _event_key(self,event_id):
        return self.event_namespace+'/'+event_id if self.event_namespace else event_id

    def _request(self,context,image_paths,prompt):
        try:return super()._request(context,image_paths,prompt)
        except Exception as exc:
            directory=getattr(self,'audit_dir',None)
            if directory is not None:
                directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
                record={'prompt':prompt,'context':context,'request_options':self.request_options,
                    'images':[{'path':str(p),'sha256':digest(p)} for p in image_paths],
                    'call_index':self._calls,'error_type':type(exc).__name__,'error':str(exc)}
                (directory/('call_%02d_failure.json'%self._calls)).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            raise

    def set_check_contract(self,context):
        previous=getattr(self,'_contract_signature',None)
        super().set_check_contract(context)
        if previous is not None and previous!=self._contract_signature and not self._explicit_store:
            self.store=SubtaskEvidence();self._reference=None
        self._task=context['criteria']['skill_description']

    def remember_identity_images(self,paths,views):
        if self.store.initial_id is not None or tuple(views)!=self.required_views:raise ValueError('initial reference already set or wrong views')
        ctx={'sensor_frames':[{'frame':i,'view':v,'timestamp':-1} for i,v in enumerate(views)]}
        self._reference=self.store.observe(ctx,paths,'initial',unknown_time=True)

    def remember_observation(self,capture,context):
        paths=[str(f['rgb_path']) for f in capture_frames(capture)]
        self._validate_source(capture,context,paths);self.set_check_contract(context)
        item=self.store.observe(context,paths)
        self._reference=self.store.observations[self.store.initial_id]
        return item

    def remember_release(self,capture,context,event_id):
        item=self.remember_observation(capture,context)
        self.store.release(self._event_key(event_id),item)

    def note_release_status(self,event_id,status):
        event_id=self._event_key(event_id)
        if event_id in self.store.events:
            self.store.events[event_id]['status']=status
            self.store.append('events',self.store.events[event_id])

    def _load_images(self,paths):
        if not isinstance(paths,list) or not 1<=len(paths)<=MAX_IMAGES:raise ValueError('evidence image budget exceeded')
        urls=[]
        for i in range(0,len(paths),6):urls.extend(SimpleQwenClient._load_images(self,paths[i:i+6]))
        return urls

    def _pages(self,current,mode):
        now=current['captured_at'];store=self.store
        def eligible(cid):
            stamp=store.observations[cid]['captured_at']
            return stamp is None or (now is not None and stamp<=now)
        def unique(values):return list(dict.fromkeys(values))
        fixed=unique([store.initial_id,current['capture_id']])
        bindings={k:v for k,v in store.bindings.items() if v['reviewed_at'] is None or (now is not None and v['reviewed_at']<=now)}
        for binding in bindings.values():fixed+= [a['capture_id'] for a in binding['anchors'] if eligible(a['capture_id'])]
        for proof in store.proofs.values():
            if now is not None and proof['reviewed_at']<=now:fixed.append(proof['capture_id'])
        fixed=unique(fixed)
        events=[e for e in store.events.values() if eligible(e['capture_id'])]
        events.sort(key=lambda e:store.observations[e['capture_id']]['captured_at'])
        pool=[]
        for event in events:
            cid=event['capture_id'];stamp=store.observations[cid]['captured_at'];pool.append(cid)
            before=[r for r in store.observations.values() if r['captured_at'] is not None and 0<stamp-r['captured_at']<=1.]
            after=[r for r in store.observations.values() if r['captured_at'] is not None and .2<=r['captured_at']-stamp<=1. and eligible(r['capture_id'])]
            if before:pool.append(max(before,key=lambda r:r['captured_at'])['capture_id'])
            if after:pool.append(min(after,key=lambda r:r['captured_at'])['capture_id'])
        pool=[cid for cid in unique(pool) if cid not in fixed]
        capacity=MAX_IMAGES//3-len(fixed)
        if capacity<0 or (pool and capacity==0):raise ValueError('fixed evidence exceeds image budget')
        chunks=[pool[i:i+capacity] for i in range(0,len(pool),capacity)] if pool else [[]]
        if mode=='initial':chunks=[[]];fixed=[store.initial_id];events=[]
        packets=[]
        for page,chunk in enumerate(chunks):
            ids=unique(fixed+chunk)
            ids.sort(key=lambda cid:(store.observations[cid]['captured_at'] if store.observations[cid]['captured_at'] is not None else float('-inf'),store.observations[cid]['sequence_index']))
            frames=[];paths=[];observations=[];lookup={}
            for cid in ids:
                item=store.observations[cid]
                if not eligible(cid):raise ValueError('future evidence')
                indices=[]
                for old,path,sha in zip(item['context']['sensor_frames'],item['paths'],item['sha256']):
                    if digest(path)!=sha:raise ValueError('original evidence bytes changed')
                    index=len(frames);lookup[(cid,old['view'])]=index;indices.append(index)
                    frames.append({'frame':index,'view':old['view'],'timestamp':item['captured_at'] if item['captured_at'] is not None else -1});paths.append(path)
                observations.append({'capture_id':cid,'captured_at':item['captured_at'],'sequence_index':item['sequence_index'],
                    'role':'initial' if cid==store.initial_id else 'current' if cid==current['capture_id'] else 'process',
                    'frames':indices,'encoder':item['encoder']})
            packet={'protocol':PROTOCOL,'scope':store.scope,'task':self._task,'goals':list(self._check_definitions.values()),
                'mode':mode if page==0 else 'history_page','reviewed_at':now,'initial_capture_id':store.initial_id,
                'current_capture_id':current['capture_id'],'observations':observations,'bindings':[],
                'review_requests':[],'previous_reviews':[],
                'coverage':{'page':page+1,'pages':len(chunks),'all_event_ids':[e['event_id'] for e in events],
                            'included_event_ids':[e['event_id'] for e in events if e['capture_id'] in ids]}}
            for target,binding in bindings.items():
                anchors=[{'frame':lookup[(a['capture_id'],a['view'])],'box':a['box']} for a in binding['anchors'] if (a['capture_id'],a['view']) in lookup]
                distractors=[{**a,'frame':lookup[(a['capture_id'],a['view'])]} for a in binding['distractors'] if (a['capture_id'],a['view']) in lookup]
                if anchors:packet['bindings'].append({'target_id':target,'anchors':anchors,'features':binding['features'],'distractors':distractors})
            if mode!='initial':
                for e in events:
                    if e['capture_id'] in ids and (page==0 or e['capture_id'] not in fixed):
                        for goal in packet['goals']:
                            if is_placement(goal):packet['review_requests'].append({'id':goal['id'],'capture_id':e['capture_id'],'event_id':e['event_id']})
                packet['previous_reviews']=[{'review_id':r['review_id'],'reviewed_at':r['reviewed_at'],'opinions':r['opinions']}
                    for r in store.reviews if r['reviewed_at'] is None or (now is not None and r['reviewed_at']<=now)][-8:]
            packets.append((packet,{**current['context'],'sensor_frames':frames},paths))
        return packets

    def _refs(self,refs,allowed):
        if not isinstance(refs,list):raise ValueError('evidence refs must be a list')
        refs=copy.deepcopy(refs)
        for ref in refs:
            if isinstance(ref,dict) and 'frame' not in ref and 'capture_id' in ref and 'view' in ref:
                matches=[o for o in self._active_packet['observations'] if o['capture_id']==ref['capture_id']]
                if len(matches)==1 and ref['view'] in self.required_views:
                    ref['frame']=matches[0]['frames'][self.required_views.index(ref['view'])]
            box=ref.get('box') if isinstance(ref,dict) else None
            if not isinstance(ref,dict) or type(ref.get('frame')) is not int or ref['frame'] not in allowed:raise ValueError('invalid frame reference')
            if not isinstance(box,list) or len(box)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1000 for x in box):raise ValueError('invalid image box')
            if not (box[0]<box[2] and box[1]<box[3]):raise ValueError('empty image box')
        return copy.deepcopy(refs)

    @staticmethod
    def _anchors_overlap(left,right):
        return any(a['capture_id']==b['capture_id'] and a['view']==b['view'] and overlap(a['box'],b['box'])>.1 for a in left for b in right)

    def _identity_anchors(self,target,bindings,packet):
        previous=self.store.bindings.get(target);anchors=copy.deepcopy(previous['anchors']) if previous else []
        for binding in bindings:
            if binding['target_id']!=target:continue
            incoming=self._stable_refs(binding['anchors'],packet)
            correcting=binding.get('corrects_previous') is True and isinstance(binding.get('correction_reason'),str) and binding['correction_reason'].strip()
            if correcting:anchors=incoming
            elif not previous or self._anchors_overlap(incoming,previous['anchors']):anchors+=incoming
        return anchors

    def parse_verdict(self,content):
        p=self._active_packet;all_frames={f for o in p['observations'] for f in o['frames']}
        captures={o['capture_id']:set(o['frames']) for o in p['observations']}
        issues=[];normalized={'bindings':[],'current':[],'history':[]}
        try:raw=vision._parse_verdict_content(content)
        except (ValueError,TypeError,json.JSONDecodeError):raw={};issues.append({'section':'document','error':'invalid JSON object'})
        if not isinstance(raw,dict):raw={};issues.append({'section':'document','error':'JSON object required'})
        self.last_raw_review=copy.deepcopy(raw)
        bindings=raw.get('bindings',[])
        if not isinstance(bindings,list):bindings=[];issues.append({'section':'bindings','error':'list required'})
        targets={g['target_id'] for g in p['goals']}
        if p['mode']=='initial':
            missing=targets-{b.get('target_id') for b in bindings if isinstance(b,dict)}
            if missing:issues.append({'section':'bindings','error':'missing initial target identities','targets':sorted(missing)})
        for binding in bindings:
            try:
                if not isinstance(binding,dict) or binding.get('target_id') not in targets or type(binding.get('matched')) is not bool:raise ValueError('invalid target binding')
                if not binding['matched']:continue
                anchors=self._refs(binding.get('anchors'),all_frames)
                if not anchors or not isinstance(binding.get('features'),str) or not binding['features'].strip():raise ValueError('identity anchor/features missing')
                distractors=self._refs(binding.get('distractors',[]),all_frames)
                normalized['bindings'].append({**binding,'anchors':anchors,'distractors':copy.deepcopy(distractors)})
            except (ValueError,TypeError) as exc:issues.append({'section':'bindings','error':str(exc)})
        for section in ('current','history'):
            expected=([] if p['mode'] in ('initial','history_page') else [{'id':g['id'],'capture_id':p['current_capture_id']} for g in p['goals']]) if section=='current' else p['review_requests']
            candidates=raw.get(section,[])
            if not isinstance(candidates,list):candidates=[]
            for request in expected:
                gid=request['id'];goal=next(g for g in p['goals'] if g['id']==gid);cid=request['capture_id']
                matches=[r for r in candidates if isinstance(r,dict) and r.get('id')==gid and (section=='current' or r.get('capture_id')==cid)]
                try:
                    if len(matches)!=1:raise ValueError('one observation per requested goal/capture required')
                    row=copy.deepcopy(matches[0]);domain=('open','closed','unknown') if goal['relation'] in ('open','close') else ('on','off','unknown') if goal['relation'] in STATES else ('at_destination','outside','unknown')
                    if row.get('state') not in domain:raise ValueError('wrong state domain')
                    if 'same_target' not in row or (row['same_target'] is not None and type(row['same_target']) is not bool):raise ValueError('same_target must be bool/null')
                    row['evidence']=self._refs(row.get('evidence'),captures[cid]);row['identity_refs']=self._refs(row.get('identity_refs'),all_frames)
                    if row['same_target'] is not True and row['state']!='unknown':raise ValueError('known target state requires same_target=true; otherwise state must be unknown')
                    if not isinstance(row.get('comparison'),str) or not row['comparison'].strip():raise ValueError('comparison missing')
                    if row['same_target'] is True and row['state']!='unknown' and (not row['evidence'] or not row['identity_refs']):raise ValueError('fact/identity reference missing')
                    if row['same_target'] is True and row['state']!='unknown':
                        anchors=self._identity_anchors(goal['target_id'],normalized['bindings'],p)
                        if not self._anchors_overlap(self._stable_refs(row['identity_refs'],p),anchors):
                            raise ValueError('identity_refs not linked to existing target anchor; to add a view include an existing anchor AND the new anchor in bindings, or explicitly correct the old identity with corrects_previous and visual correction_reason; never guess')
                    if 'supported' not in row or (row['supported'] is not None and type(row['supported']) is not bool):raise ValueError('supported must be bool/null')
                    if row.get('change','none') not in ('none','removed','moved','dropped'):raise ValueError('invalid change')
                    row.update(capture_id=cid,change=row.get('change','none'));normalized[section].append(row)
                except (ValueError,TypeError) as exc:
                    row=unknown(gid,str(exc));row.update(capture_id=cid,protocol_invalid=True)
                    normalized[section].append(row);issues.append({'section':section,'id':gid,'capture_id':cid,'error':str(exc)})
        self.last_review={'normalized':normalized,'issues':issues,'raw':copy.deepcopy(raw)}
        return SimpleQwenClient.parse_verdict(self,json.dumps({'goal_achieved':None,'reason':'observations awaiting evidence aggregation'}))

    def _stable_refs(self,refs,packet):
        mapping={f:(o['capture_id'],i) for o in packet['observations'] for i,f in enumerate(o['frames'])}
        return [{'capture_id':mapping[r['frame']][0],'view':self.required_views[mapping[r['frame']][1]],'box':copy.deepcopy(r['box'])} for r in refs]

    def _apply(self,review,packet,meta):
        store=self.store;now=packet['reviewed_at'];rid=store.scope_id+'/review_%05d'%(len(store.reviews)+1)
        normalized=review['normalized'];issues=copy.deepcopy(review['issues']);new_proofs=[];opinions=[]
        for binding in normalized['bindings']:
            target=binding['target_id'];anchors=self._stable_refs(binding['anchors'],packet)
            conflict=any(a['capture_id']==b['capture_id'] and a['view']==b['view'] and overlap(a['box'],b['box'])>.8
                for other in normalized['bindings'] if other['target_id']!=target
                for a in anchors for b in self._stable_refs(other['anchors'],packet))
            if conflict:issues.append({'section':'bindings','target_id':target,'error':'same region assigned to distinct targets'});continue
            correcting=(target in store.bindings and binding.get('corrects_previous') is True
                        and isinstance(binding.get('correction_reason'),str) and binding['correction_reason'].strip())
            if target in store.bindings and not correcting and self._anchors_overlap(anchors,store.bindings[target]['anchors']):
                # New views are supplemental only when the answer explicitly links them to an old image anchor.
                anchors=list({json.dumps(a,sort_keys=True):a for a in store.bindings[target]['anchors']+anchors}.values())
                store.bindings[target]={**store.bindings[target],'anchors':anchors,'reviewed_at':now,'review_id':rid}
            if target not in store.bindings or correcting:
                distractors=[{**a,'features':b.get('features','')} for a,b in zip(self._stable_refs(binding['distractors'],packet),binding['distractors'])]
                store.bindings[target]={'target_id':target,'anchors':anchors,'features':binding['features'],
                    'distractors':distractors,'reviewed_at':now,'review_id':rid}
                if correcting:
                    for key,proof in list(store.proofs.items()):
                        if proof['target_id']==target:
                            store.proofs[key]={**proof,'status':'identity_recheck_required','identity_revision':rid}
        accepted={'current':[],'history':[]}
        for section in ('history','current'):
            for original in normalized[section]:
                row=copy.deepcopy(original);goal=self._check_definitions[row['id']];key=goal_key(goal);target=goal['target_id']
                identity=self._stable_refs(row['identity_refs'],packet)
                binding=store.bindings.get(target)
                bound=binding is not None and any(a['capture_id']==b['capture_id'] and a['view']==b['view'] and overlap(a['box'],b['box'])>.1 for a in identity for b in binding['anchors'])
                if row['same_target'] is True and row['state']!='unknown' and not bound:
                    row.update(same_target=None,state='unknown',grounding_invalid=True)
                    issues.append({'section':section,'id':row['id'],'error':'target not linked to its image anchor'})
                evidence=self._stable_refs(row['evidence'],packet);item=store.observations[row['capture_id']]
                collision=is_placement(goal) and any(other['id']!=row['id'] and is_placement(self._check_definitions[other['id']])
                    and other['capture_id']==row['capture_id'] and other['same_target'] is True
                    and any(a['frame']==b['frame'] and overlap(a['box'],b['box'])>.8 for a in row['evidence'] for b in other['evidence'])
                    for other in normalized[section])
                if collision:
                    row.update(same_target=None,state='unknown',grounding_invalid=True)
                    issues.append({'section':section,'id':row['id'],'error':'same image region assigned to distinct placement targets'})
                if section=='current' and is_placement(goal) and row['same_target'] is True and row['state']=='outside' and row['change']!='none':
                    proof=store.proofs.get(key)
                    if proof and item['captured_at']>proof['observed_at']:
                        proof=copy.deepcopy(proof);proof.update(status='revoked',revoked_at=now,revocation_review_id=rid)
                        store.proofs[key]=proof
                if (is_placement(goal) and row['same_target'] is True and row['state']=='at_destination' and row['supported'] is True
                        and item['encoder'] is not None and item['encoder']['at_standard_open']):
                    existing=store.proofs.get(key)
                    # A re-review cannot resurrect evidence older than a known removal.
                    usable=existing is None or (existing['status']!='confirmed' and item['captured_at']>existing.get('revoked_at',float('-inf')))
                    if usable:
                        prior=[r['review_id'] for r in store.reviews if any(o['goal_key']==key and o['capture_id']==row['capture_id'] for o in r['opinions'])]
                        event=next((e['event_id'] for e in store.events.values() if e['capture_id']==row['capture_id']),None)
                        proof={'proof_id':rid+'/'+row['id'],'goal_key':key,'target_id':target,'destination_id':goal.get('destination_id'),
                            'capture_id':row['capture_id'],'observed_at':item['captured_at'],'reviewed_at':now,'review_id':rid,
                            'event_id':event,'status':'confirmed','kind':'historical_review' if section=='history' else 'current_observation',
                            'supersedes_review_id':prior[-1] if prior else None,'evidence':evidence,'identity_refs':identity,
                            'comparison':row['comparison'],'source_sha256':item['sha256'],'encoder':item['encoder'],
                            'request_sha256':row.get('_provenance',meta).get('request_sha256'),
                            'response_sha256':row.get('_provenance',meta).get('response_sha256')}
                        store.proofs[key]=proof;new_proofs.append(copy.deepcopy(proof))
                opinions.append({'goal_key':key,'capture_id':row['capture_id'],'state':row['state'],'same_target':row['same_target'],
                    'section':section,'comparison':row['comparison'],'evidence':evidence,
                    'provenance':{k:row.get('_provenance',meta).get(k) for k in ('request_sha256','response_sha256')}})
                accepted[section].append(row)
        record={'review_id':rid,'reviewed_at':now,'mode':packet['mode'],'opinions':opinions,'new_proofs':new_proofs,
            'issues':issues,'raw':review['raw'],'normalized':accepted,'request_sha256':meta.get('request_sha256'),
            'response_sha256':meta.get('response_sha256')}
        store.review(record)
        return record

    def _call_page(self,packet,context,paths):
        self._active_packet=packet;prompt=build_prompt(packet)
        meta=self._request(context,paths,prompt);review=copy.deepcopy(self.last_review)
        original=copy.deepcopy(review);repair=None
        if review['issues'] and not self.store.repair_used and self._calls<self.max_calls:
            self.store.repair_used=True
            correction='\n上次回答有协议缺失，只修正列出的问题，保持其余事实。不能补造图片引用；确实看不清应返回unknown。错误：'+json.dumps(review['issues'],ensure_ascii=False)+'\n上次回答：'+json.dumps(review['raw'],ensure_ascii=False)
            # Keep packet as the last JSON line for reproducible request audit.
            retry_prompt=prompt.rsplit('\n',1)[0]+correction+'\n'+json.dumps(packet,ensure_ascii=False,separators=(',',':'))
            retry_meta=self._request(context,paths,retry_prompt);fixed=copy.deepcopy(self.last_review)
            for section in ('current','history'):
                for row in fixed['normalized'][section]:
                    row['_provenance']={k:retry_meta.get(k) for k in ('request_sha256','response_sha256')}
                lookup={(r['id'],r['capture_id']):r for r in fixed['normalized'][section]}
                review['normalized'][section]=[lookup.get((r['id'],r['capture_id']),r) if r.get('protocol_invalid') else r for r in review['normalized'][section]]
            repaired_bindings={b['target_id']:b for b in review['normalized']['bindings']}
            for b in fixed['normalized']['bindings']:
                target=b['target_id'];previous=self.store.bindings.get(target)
                # Accept a repair's added identity link; keep an earlier valid binding if the repair omits it.
                linked=previous is None or self._anchors_overlap(self._stable_refs(b['anchors'],packet),previous['anchors'])
                correcting=b.get('corrects_previous') is True and bool(b.get('correction_reason'))
                if target not in repaired_bindings or linked or correcting:repaired_bindings[target]=b
            review['normalized']['bindings']=list(repaired_bindings.values())
            review['issues']=[i for i in fixed['issues'] if i.get('section')=='bindings' or any(r.get('protocol_invalid') and r['id']==i.get('id') and r['capture_id']==i.get('capture_id') for s in ('current','history') for r in review['normalized'][s])]
            repair={'metadata':retry_meta,'prompt_text':retry_prompt,'review':fixed}
        record=self._apply(review,packet,meta)
        return {'metadata':meta,'packet':packet,'prompt_text':prompt,'request_context':context,'image_paths':paths,
                'original_review':original,'repair':repair,'applied_review':record}

    def analyze_initial(self):
        if self.store.initial_review_done:return None
        if self.store.initial_id is None:raise ValueError('initial observation missing')
        initial=self.store.observations[self.store.initial_id]
        packet,ctx,paths=self._pages(initial,'initial')[0]
        result=self._call_page(packet,ctx,paths);self.store.initial_review_done=True
        return result

    def _effective(self,rows,blocked):
        result=[]
        for goal,row in zip(self._check_definitions.values(),rows):
            proof=self.store.proofs.get(goal_key(goal));known=proof and proof['status']=='confirmed';same=row['same_target'] is True
            history=[];state=row['state'];direct=same and state!='unknown'
            if goal['id'] in blocked:achieved=None
            elif goal['relation'] in STATES:achieved=(state==STATES[goal['relation']]) if direct else None
            elif is_placement(goal) and known:
                achieved=None if same and state=='outside' else True
                if achieved:history=[proof['proof_id']]
            else:achieved=(state=='at_destination') if direct else None
            item={'id':goal['id'],'achieved':achieved,'target_matches_definition':True if history else (True if same else None),
                'current_evidence':direct,'evidence_event_ids':history,'reason':row['comparison']}
            if history:item.update(history_event_id=history[0],history_event_ids=history,
                history_capture_id=proof['capture_id'],history_observed_at=proof['observed_at'],history_reviewed_at=proof['reviewed_at'])
            result.append(item)
        return result

    def _inspect(self,context,paths):
        current=self.store.observe(context,paths)
        times=[r['reviewed_at'] for r in self.store.reviews if r['reviewed_at'] is not None]
        if times and current['captured_at']<max(times):raise ValueError('review time went backwards; use history review with current capture')
        bootstrap=self.analyze_initial()
        key=(current['capture_id'],self.store.revision,tuple(goal_key(g) for g in self._check_definitions.values()))
        if key in self.store.cache:
            result=copy.deepcopy(self.store.cache[key]);result['reused_same_capture']=True;return result
        pages=[];current_rows=None;blocked=set()
        for packet,ctx,images in self._pages(current,'current'):
            page=self._call_page(packet,ctx,images);pages.append(page)
            if current_rows is None:
                current_rows=page['applied_review']['normalized']['current']
                blocked={r['id'] for r in current_rows if r.get('protocol_invalid') or r.get('grounding_invalid')}
        effective=self._effective(current_rows,blocked);aggregate=self._aggregate(effective)
        result={'checker':PROTOCOL,'verdict':SimpleQwenClient.parse_verdict(self,json.dumps({'goal_achieved':aggregate,'reason':'initial/process/current evidence review'})),
            'visual_goal_verdict':{'checks':effective,'goal_achieved':aggregate},'effective_checks':effective,
            'process_evidence':self.process_evidence_snapshot(),'pages':pages,'initial_review':bootstrap,
            'observations':self.observation_snapshot(),'release_archive':copy.deepcopy(list(self.store.events.values())),
            'actual_Qwen_calls':self._calls,'current_capture_id':current['capture_id'],'scope':copy.deepcopy(self.store.scope)}
        self.last_effective_checks=copy.deepcopy(effective)
        final_key=(current['capture_id'],self.store.revision,tuple(goal_key(g) for g in self._check_definitions.values()))
        self.store.cache={final_key:copy.deepcopy(result)}
        return result

    def analyze(self,context,image_paths):
        self._validate_capture(context,image_paths);self.set_check_contract(context)
        return self._inspect(context,image_paths)

    def analyze_release(self,context,image_paths,event_id):
        self._validate_capture(context,image_paths);self.remember_release(self._capture,context,event_id)
        event=self.store.observations[self.store.events[self._event_key(event_id)]['capture_id']]
        after=[r for r in self.store.observations.values() if r['captured_at'] is not None and .2<=r['captured_at']-event['captured_at']<=1.]
        current=min(after,key=lambda r:r['captured_at']) if after else event
        result=self._inspect(current['context'],current['paths']);self.note_release_status(event_id,'reviewed')
        result.update(event_id=event_id,observation_role='release_snapshot');return result

    def process_evidence_snapshot(self):
        return copy.deepcopy(list(self.store.proofs.values()))

    def observation_snapshot(self):
        return copy.deepcopy(list(self.store.observations.values()))
