from pathlib import Path
p=Path('First_Phase/scene_demo/xvla_console.html')
s=p.read_text(encoding='utf-8')
s=s.replace("mugs_two:'两个杯子与盘子',", "mugs_two:'两个杯子与盘子',mugs_left_occupied:'双杯：左盘已有物品',mugs_right_occupied:'双杯：右盘已有物品',mugs_both_occupied:'双杯：两盘已有物品',")
s=s.replace("const phases={", "const phases={visual_check:'视觉检查',movement:'退离',blocked:'检查阻断',executing:'X-VLA 动作',")
s=s.replace("const f=j.finish_status||{};return", "const f=j.finish_status||{},phase=f.phase||j.phase;return")
s=s.replace("phases[j.phase]||j.phase||stageNames[j.state]", "phases[phase]||phase||stageNames[j.state]")
s=s.replace("+(j.ended_reason?", "+(f.last_decision?'<details><summary>最近检查决定</summary><pre>'+esc(JSON.stringify(f.last_decision,null,2))+'</pre></details>':'')+(j.ended_reason?")
s=s.replace("['finish','release','retreat','home'].includes(j.phase)", "['finish','release','retreat','movement','home','visual_check'].includes(j.finish_status?.phase||j.phase)")
# Stable HTML preserves expanded disclosures across unchanged polls.
s=s.replace("const media=path=>", "const htmlCache=new Map();function stableHTML(id,html){if(htmlCache.get(id)!==html){const opened=[...$(id).querySelectorAll('details')].map(d=>d.open);$(id).innerHTML=html;$(id).querySelectorAll('details').forEach((d,i)=>{d.open=opened[i]||false;});htmlCache.set(id,html);}}\nconst media=path=>")
s=s.replace("$('jobs').innerHTML=jobs.length?", "stableHTML('jobs',jobs.length?")
s=s.replace("状态。')+'</div>';\nlet result=", "状态。')+'</div>');\nlet result=")
p.write_text(s,encoding='utf-8')
