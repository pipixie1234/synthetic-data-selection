"""Sequential GPU pipeline with resumable stages and separate logs."""
import json
import subprocess
import sys
import time
from pathlib import Path

BASE=Path(__file__).resolve().parent
OUT=BASE/'outputs'
OUT.mkdir(exist_ok=True)
steps=[('score_qwen',['run_cross_verifier.py','score','--model','qwen']),('score_llama',['run_cross_verifier.py','score','--model','llama']),('selection',['run_cross_verifier.py','select']),('diagnostics',['analyze_cross_verifier.py','diagnostics']),('downstream',['run_cross_verifier.py','train']),('final_report',['analyze_cross_verifier.py','report']),('validation',['verify_results.py'])]
for name,arguments in steps:
    state=dict(stage=name,status='running',started_at_unix=time.time(),command=arguments)
    (OUT/'pipeline_status.json').write_text(json.dumps(state,indent=2)+'\n')
    print('Starting '+name,flush=True)
    with (OUT/(name+'.log')).open('a') as log:
        result=subprocess.run([sys.executable,str(BASE/arguments[0]),*arguments[1:]],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        state.update(status='failed',returncode=result.returncode)
        (OUT/'pipeline_status.json').write_text(json.dumps(state,indent=2)+'\n')
        raise SystemExit(f'{name} failed; see {OUT/(name+".log")}')
    print('Completed '+name,flush=True)
(OUT/'pipeline_status.json').write_text(json.dumps(dict(status='complete',completed_at_unix=time.time()),indent=2)+'\n')
print('Experiment and final artifacts complete',flush=True)
