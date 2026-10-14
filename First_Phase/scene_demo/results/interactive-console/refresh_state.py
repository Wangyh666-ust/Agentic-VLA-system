from pathlib import Path
p=Path('First_Phase/scene_demo/xvla_console.html')
s=p.read_text(encoding='utf-8').replace("lastVideos='', examples=[];", "lastVideos='', lastLive='', examples=[];")
s=s.replace("function render(){const live=data.mode==='live';", "function render(){const live=data.mode==='live';if(live&&data.request_id!==lastLive){lastLive=data.request_id;$('scene').value=data.scene;$('request').value=data.request||'';}")
p.write_text(s,encoding='utf-8')
