#!/usr/bin/env python3
"""Frozen cross-verifier downstream experiment. No synthetic generation or holdout access."""
from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'SyntheticData/redundancy_repair'))
import run_repair as repair
old=repair.old
OUT=Path(__file__).resolve().parent/'outputs'
POOL=ROOT/'cloud/model/extra_data/candidate_pool/outputs/expanded_candidate_pool.csv'
DEV=ROOT/'SyntheticData/exp/outputs/curation_validation/dev2000.csv'
BERT=os.environ.get('BERT_MODEL','bert-base-uncased')
MODEL_SEEDS=[42,43,44,45,46]
SELECTION_SEEDS=[41,42,43,44,45]
MAIN_METHODS=['Random320','TopK320','SelfVerifier320','CrossVerifier320']
SOURCE_METHODS=['SelfQwen160','CrossQwen160','SelfLlama160','CrossLlama160']
METRICS=['Dev_Accuracy','Dev_Macro_F1']+[f'Dev_{x}_F1' for x in repair.LABELS]
HYPER=dict(epochs=3,batch_size=16,eval_batch_size=64,learning_rate=2e-5,weight_decay=.01,max_length=256,warmup_steps=0,lr_scheduler_type='linear',optim='adamw_torch',gradient_accumulation_steps=1,max_grad_norm=1.0,fp16=True,tf32=False,loss='ordinary_ce',eval_strategy='no',save_strategy='no')
SIGNAL_POLICY=dict(strong_min_macro_f1_delta=.01,weak_min_macro_f1_delta=.005,strong_min_positive_model_seeds=4,cross_required_comparators=['SelfVerifier320','TopK320'],cross_random_gate='positive mean versus RandomMean',selection_utility_required_comparators=['TopK320','RandomMean'],signal_is_pilot_heuristic=True)


def pool_rows():
    import pandas as pd
    rows=pd.read_csv(POOL).to_dict('records')
    for x in rows:
        x['label']=int(x['label'])
        x['item_id']=x['sample_id']
    return sorted(rows,key=lambda x:(x['source'],x['label'],x['sample_id']))


def stable_log_probabilities(values):
    """Avoid exp/normalization rounding high label probabilities to exactly 1."""
    names=list(values)
    winner=max(names,key=lambda k:values[k])
    maximum=values[winner]
    tail=sum(math.exp(values[k]-maximum) for k in names if k!=winner)
    remainder=math.log1p(tail)
    return {k:(values[k]-maximum)-remainder for k in names}


def finite_log(value):
    """Preserve non-finite AMP diagnostic values as explicit strings in JSON."""
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,dict):return {k:finite_log(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_log(v) for v in value]
    return value


def check_frozen():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert old.digest(POOL)==manifest['pool_sha256'],'Candidate pool changed'
    assert old.digest(OUT/'real_anchors.jsonl')==manifest['anchor_sha256']
    assert old.digest(OUT/'label_prompts.jsonl')==manifest['prompt_sha256']
    assert HYPER==manifest['downstream_hyperparameters']
    assert SIGNAL_POLICY==manifest['signal_policy']
    assert old.digest(repair.OUT/'definitions.txt')==manifest['definitions_sha256']
    return manifest


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():
        check_frozen()
        print('Frozen experiment reused',flush=True)
        return
    repair.prepare()
    rows=pool_rows()
    assert len(rows)==4000 and len({x['sample_id'] for x in rows})==4000
    assert collections.Counter((x['source'],x['label']) for x in rows)==collections.Counter({(s,l):500 for s in ['qwen','llama'] for l in range(4)})
    assert len({repair.norm(x['text']) for x in rows})==4000
    anchors=old.read_jsonl(repair.OUT/'real_anchors.jsonl')
    assert not {repair.norm(x['text']) for x in anchors}&{repair.norm(x['text']) for x in rows}
    old.write_jsonl(OUT/'real_anchors.jsonl',anchors)
    old.write_jsonl(OUT/'label_prompts.jsonl',[dict(item_id=x['sample_id'],prompt=repair.label_prompt(x['text'],anchors)) for x in rows])
    old.write_json(OUT/'manifest.json',dict(experiment='cross_verifier_downstream',created_date='2026-10-06',pool_path=str(POOL),pool_sha256=old.digest(POOL),pool_n=4000,group_size=500,selected_per_source_class=40,anchors_path=str(repair.OUT/'real_anchors.jsonl'),anchor_sha256=old.digest(OUT/'real_anchors.jsonl'),definitions_sha256=old.digest(repair.OUT/'definitions.txt'),prompt_sha256=old.digest(OUT/'label_prompts.jsonl'),repaired_protocol='exact frozen real-anchored prompt; no presented label; native chat prefix predicted_label:; four complete candidate strings including leading space/newline; no extra BOS',score_policy='LabelScore from frozen four-candidate full-sequence likelihoods; rank descending stable log-normalized LabelScore within source/class; no cross-model averaging; no hard label/quality filter',stable_log_policy='math.log1p of sum of nonmax candidate masses preserves near-1 ranking; mathematically monotonic transform of existing LabelScore, not an extra selection criterion',score_tiebreak='ascending sample_id if exact stable log-score tie',topk_policy='frozen existing proximity_score DESC, sample_id ASC; no encoder/scorer retuning',random_selection_seeds=SELECTION_SEEDS,model_seeds=MODEL_SEEDS,downstream_model=str(BERT),downstream_hyperparameters=HYPER,primary_methods=MAIN_METHODS,source_only_auxiliary_methods=SOURCE_METHODS,primary_runs=40,source_only_auxiliary_runs=20,task_b_policy='diagnostic only; reference existing repaired pilot Task B; not used to score/filter/select; not rerun over full pool',phi_branch='Skipped: no Phi-3 Mini checkpoint installed; optional branch does not block main experiment',all4000_branch='Not run: primary comparison keeps all methods at identical 320-sample budget',dev_policy='Dev2000 first loaded only after all selections frozen; one final evaluation per trained model; no tuning/threshold/model selection/early stopping',final_holdout_accessed=False,new_synthetic_generation=False,selection_frozen=False))
    manifest=json.loads((OUT/'manifest.json').read_text());manifest['signal_policy']=SIGNAL_POLICY
    old.write_json(OUT/'manifest.json',manifest)
    print('Frozen pool4000, repaired prompt/anchors, 40 main +20 source-only runs; Dev not loaded',flush=True)


def score(args):
    import torch
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    prepare()
    prompts=old.read_jsonl(OUT/'label_prompts.jsonl')
    dest=OUT/f'{args.model}_scores.jsonl'
    prior=repair.cached(dest)
    pilot_samples={x['item_id']:x for x in old.read_jsonl(repair.OUT/'samples.jsonl') if x['source']!='real'}
    pilot_scores=old.read_jsonl(repair.OUT/f'{args.model}_real_anchored_scores.jsonl')
    prompt_map={x['item_id']:x['prompt'] for x in prompts}
    for row in pilot_scores:
        if row['item_id'] not in pilot_samples: continue
        sample=pilot_samples[row['item_id']]
        candidate_id=sample['original_id']
        if candidate_id in prior: continue
        if candidate_id not in prompt_map: raise RuntimeError('Pilot synthetic ID missing from full pool')
        assert row['prompt_sha256']==hashlib.sha256(prompt_map[candidate_id].encode()).hexdigest()
        adapted=dict(row,item_id=candidate_id,reused_from_repaired_pilot=True)
        repair.append(dest,adapted)
    prior=repair.cached(dest)
    if len(prior)==4000:
        print(f'{args.model}: all4000 scores cached',flush=True)
        return
    torch.set_num_threads(8);torch.manual_seed(repair.SEED)
    path=ROOT/'cloud/model'/{'qwen':'Qwen3-4B-Instruct-2507','llama':'Llama-3.2-3B-Instruct'}[args.model]
    print(f'Loading {path.name}; cached {len(prior)}/4000',flush=True)
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left')
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').to(args.device).eval()
    runtime=dict(model=str(path),config_sha256=old.digest(path/'config.json'),torch=torch.__version__,transformers=transformers.__version__,dtype=str(model.dtype),device=args.device,score_batch_size=args.score_batch_size,prompt_sha256=old.digest(OUT/'label_prompts.jsonl'),seed=repair.SEED,reused_pilot_scores=160,reused_pilot_score_batch_size=2,native_tokenization_add_special_tokens=False,scoring_implementation='shared prefix KV cache with per-candidate continuations',retained_preexisting_scores=len(prior))
    if (OUT/f'{args.model}_runtime.json').exists():repair.append(OUT/f'{args.model}_runtime_history.jsonl',json.loads((OUT/f'{args.model}_runtime.json').read_text()))
    old.write_json(OUT/f'{args.model}_runtime.json',runtime)
    started=time.monotonic()
    score_cached(model,tokenizer,prompts,dest,args)
    elapsed=time.monotonic()-started
    print(f'{args.model}: all4000 candidate scores completed in {elapsed:.1f}s',flush=True)
    del model;gc.collect();torch.cuda.empty_cache()


def score_cached(model,tokenizer,prompts,dest,args):
    import torch
    from cached_scoring import candidate_values
    prior=repair.cached(dest)
    pending=[x for x in prompts if x['item_id'] not in prior]
    suffixes=[tokenizer.encode(' '+label+'\n',add_special_tokens=False) for label in repair.LABELS]
    for start in range(0,len(pending),args.score_batch_size):
        batch=pending[start:start+args.score_batch_size]
        rendered=[tokenizer.apply_chat_template([dict(role='user',content=x['prompt'])],tokenize=False,add_generation_prompt=True)+repair.OUTPUT_PREFIX for x in batch]
        prefixes=[tokenizer.encode(x,add_special_tokens=False) for x in rendered]
        for prompt,prefix in zip(rendered,prefixes):
            for label,suffix in zip(repair.LABELS,suffixes):assert tokenizer.encode(prompt+' '+label+'\n',add_special_tokens=False)==prefix+suffix
        values=candidate_values(model,prefixes,suffixes,tokenizer.pad_token_id,args.device)
        if start==0:
            # Check the new cache path against the previous full-sequence teacher forcing.
            reference=[]
            for suffix in suffixes:
                sequence=prefixes[0]+suffix
                with torch.inference_mode():
                    output=model(input_ids=torch.tensor([sequence],device=args.device),use_cache=False,logits_to_keep=len(suffix)+1)
                    reference.append(repair.suffix_log_probabilities(output.logits,[suffix])[0])
            difference=max(abs(a-b) for a,b in zip(values[:4],reference))
            old.write_json(OUT/f'{args.model}_cache_scoring_audit.json',dict(first_item_id=batch[0]['item_id'],cached_candidate_logprobs=values[:4],full_sequence_reference=reference,max_absolute_logprob_difference=difference,allowed_fp16_tolerance=.15))
            assert difference<.15,f'Cached/full-sequence scoring mismatch: {difference}'
        for i,x in enumerate(batch):
            logits=values[i*4:(i+1)*4];maximum=max(logits)
            masses=[math.exp(v-maximum) for v in logits]
            repair.append(dest,dict(item_id=x['item_id'],prompt_sha256=hashlib.sha256(x['prompt'].encode()).hexdigest(),candidate_logprob=dict(zip(repair.LABELS,logits)),candidate_probability=dict(zip(repair.LABELS,[v/sum(masses) for v in masses])),candidate_token_count=dict(zip(repair.LABELS,map(len,suffixes))),candidate_argmax=repair.LABELS[logits.index(maximum)],scoring_implementation='shared_prefix_cache'))
        if start%80==0 or start+len(batch)==len(pending):print(f'{args.model}: {len(prior)+min(start+len(batch),len(pending))}/4000',flush=True)


def selected_dataset(rows,method,selection_seed,score_map=None):
    selected=[]
    rng=random.Random(selection_seed) if method=='Random320' else None
    source_only='qwen' if method.endswith('Qwen160') else 'llama' if method.endswith('Llama160') else None
    for source in ['qwen','llama']:
        if source_only and source!=source_only:continue
        for label in range(4):
            group=sorted([x for x in rows if x['source']==source and x['label']==label],key=lambda x:x['sample_id'])
            if method=='Random320':chosen=rng.sample(group,40)
            elif method=='TopK320':chosen=sorted(group,key=lambda x:(-x['proximity_score'],x['sample_id']))[:40]
            else:
                verifier=source if method.startswith('Self') else {'qwen':'llama','llama':'qwen'}[source]
                chosen=sorted(group,key=lambda x:(-score_map[verifier][x['sample_id']]['stable_log_probability'][repair.LABELS[label]],x['sample_id']))[:40]
            selected+=chosen
    return sorted(selected,key=lambda x:(x['source'],x['label'],x['sample_id']))


def select():
    prepare();manifest=check_frozen()
    if manifest['selection_frozen']:
        for path,h in manifest['selected_dataset_hashes'].items():assert old.digest(OUT/path)==h
        print('All selected datasets already frozen',flush=True);return
    rows=pool_rows();scores={}
    flattened=[]
    for model in ['qwen','llama']:
        values=repair.cached(OUT/f'{model}_scores.jsonl')
        assert set(values)=={x['sample_id'] for x in rows},f'{model} missing full4000 scores'
        scores[model]={k:dict(v,stable_log_probability=stable_log_probabilities(v['candidate_logprob'])) for k,v in values.items()}
    prompts={x['item_id']:x['prompt'] for x in old.read_jsonl(OUT/'label_prompts.jsonl')}
    for row in rows:
        entry=dict(sample_id=row['sample_id'],source=row['source'],label=row['label'],class_name=repair.LABELS[row['label']],proximity_score=row['proximity_score'])
        for model in ['qwen','llama']:
            score=scores[model][row['sample_id']]
            assert score['prompt_sha256']==hashlib.sha256(prompts[row['sample_id']].encode()).hexdigest()
            entry[f'{model}_LabelScore']=score['candidate_probability'][repair.LABELS[row['label']]]
            entry[f'{model}_LogLabelScore']=score['stable_log_probability'][repair.LABELS[row['label']]]
            entry[f'{model}_predicted_label']=score['candidate_argmax']
            for label in repair.LABELS:
                entry[f'{model}_logprob_{label}']=score['candidate_logprob'][label]
        flattened.append(entry)
    old.write_csv(OUT/'candidate_verifier_scores.csv',flattened)
    directory=OUT/'selected';directory.mkdir(exist_ok=True)
    specifications=[('Random320',seed) for seed in SELECTION_SEEDS]+[(method,None) for method in MAIN_METHODS[1:]+SOURCE_METHODS]
    hashes={};counts=[]
    for method,seed in specifications:
        dataset=selected_dataset(rows,method,seed,scores)
        expected=160 if method in SOURCE_METHODS else 320
        assert len(dataset)==expected and len({x['sample_id'] for x in dataset})==expected
        assert set(collections.Counter((x['source'],x['label']) for x in dataset).values())=={40}
        name=f'{method}_sel{seed}.jsonl' if seed is not None else f'{method}.jsonl'
        path=directory/name;old.write_jsonl(path,dataset)
        hashes[str(path.relative_to(OUT))]=old.digest(path)
        counts.append(dict(Method=method,Selection_Seed=seed if seed is not None else 'fixed',N=len(dataset),dataset_sha256=old.digest(path)))
    old.write_csv(OUT/'selection_manifest.csv',counts)
    manifest['selected_dataset_hashes']=hashes;manifest['selection_frozen']=True
    manifest['scores_sha256']={model:old.digest(OUT/f'{model}_scores.jsonl') for model in ['qwen','llama']}
    old.write_json(OUT/'manifest.json',manifest)
    print('Frozen all12 datasets: 5 Random320 +3 deterministic320 +4 source-only160. Dev not loaded.',flush=True)


def training_specs():
    for seed in MODEL_SEEDS:
        for method in MAIN_METHODS:
            selections=SELECTION_SEEDS if method=='Random320' else [None]
            for selection in selections:
                yield method,selection,seed
        for method in SOURCE_METHODS:yield method,None,seed


def train():
    import numpy as np
    import pandas as pd
    import torch
    import transformers
    from datasets import Dataset
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments,DataCollatorWithPadding,set_seed
    from sklearn.metrics import accuracy_score,f1_score
    manifest=check_frozen();assert manifest['selection_frozen']
    for path,h in manifest['selected_dataset_hashes'].items():assert old.digest(OUT/path)==h
    # This is deliberately the first Dev access in this experiment.
    dev=pd.read_csv(DEV)
    assert len(dev)==2000 and dev.label.value_counts().to_dict()=={0:500,1:500,2:500,3:500}
    ids=json.loads((DEV.parent/'dev2000_ids.json').read_text())
    assert dev.original_dataset_index.tolist()==ids
    pool=pool_rows();anchors=old.read_jsonl(OUT/'real_anchors.jsonl')
    dev_text={repair.norm(x) for x in dev.text}
    assert not dev_text & {repair.norm(x['text']) for x in pool},'Leakage: candidate text in Dev'
    assert not dev_text & {repair.norm(x['text']) for x in anchors},'Leakage: demonstration text in Dev'
    audit=dict(dev_path=str(DEV),dev_sha256=old.digest(DEV),n=2000,class_counts=dev.label.value_counts().to_dict(),candidate_exact_text_overlap=0,demo_exact_text_overlap=0,all_selections_frozen_before_dev_access=True,final_holdout_accessed=False)
    old.write_json(OUT/'evaluation_audit.json',audit)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    torch.set_num_threads(8)
    tokenizer=AutoTokenizer.from_pretrained(BERT,local_files_only=True)
    tokenize=lambda x:tokenizer(x['text'],truncation=True,max_length=HYPER['max_length'])
    eval_ds=Dataset.from_list([dict(text=x['text'],labels=int(x['label'])) for x in dev.to_dict('records')]).map(tokenize,batched=True,remove_columns=['text'])
    all_results=[]
    start_all=time.monotonic()
    for run_index,(method,selection,seed) in enumerate(training_specs(),1):
        run_name=f'{method}_sel{selection}_model{seed}' if selection is not None else f'{method}_model{seed}'
        directory=OUT/'training'/run_name;directory.mkdir(parents=True,exist_ok=True)
        dataset_path=OUT/'selected'/(f'{method}_sel{selection}.jsonl' if selection is not None else f'{method}.jsonl')
        training_hash=old.digest(dataset_path)
        result_path=directory/'results.json'
        if result_path.exists():
            result=json.loads(result_path.read_text());assert result['dataset_sha256']==training_hash and result['hyperparameters']==HYPER and result['Dev_SHA256']==audit['dev_sha256']
            all_results.append(result);continue
        started=time.monotonic();set_seed(seed)
        records=old.read_jsonl(dataset_path)
        train_ds=Dataset.from_list([dict(text=x['text'],labels=int(x['label'])) for x in records]).map(tokenize,batched=True,remove_columns=['text'])
        # Seed BEFORE loading: classifier initialization must match across paired methods.
        set_seed(seed)
        model=AutoModelForSequenceClassification.from_pretrained(BERT,num_labels=4,local_files_only=True,attn_implementation='sdpa')
        head_hash=hashlib.sha256(model.classifier.weight.detach().cpu().numpy().tobytes()+model.classifier.bias.detach().cpu().numpy().tobytes()).hexdigest()
        args=TrainingArguments(output_dir=str(directory),num_train_epochs=HYPER['epochs'],per_device_train_batch_size=16,per_device_eval_batch_size=64,learning_rate=2e-5,weight_decay=.01,warmup_steps=0,lr_scheduler_type='linear',optim='adamw_torch',max_grad_norm=1.0,gradient_accumulation_steps=1,fp16=True,tf32=False,seed=seed,data_seed=seed,save_strategy='no',eval_strategy='no',logging_steps=20,report_to='none',disable_tqdm=True,dataloader_num_workers=0)
        trainer=Trainer(model=model,args=args,train_dataset=train_ds,processing_class=tokenizer,data_collator=DataCollatorWithPadding(tokenizer))
        trainer.train()
        prediction=trainer.predict(eval_ds)
        logits=prediction.predictions;predicted=logits.argmax(-1);truth=prediction.label_ids
        f1=f1_score(truth,predicted,labels=[0,1,2,3],average=None,zero_division=0)
        result=dict(Method=method,Selection_Seed=selection if selection is not None else 'fixed',Model_Seed=seed,Train_Count=len(records),Dev_Accuracy=float(accuracy_score(truth,predicted)),Dev_Macro_F1=float(f1.mean()),**{f'Dev_{label}_F1':float(value) for label,value in zip(repair.LABELS,f1)},dataset_sha256=training_hash,Dev_SHA256=audit['dev_sha256'],classifier_init_sha256=head_hash,hyperparameters=HYPER,torch=torch.__version__,transformers=transformers.__version__,training_steps=trainer.state.global_step,training_seconds=time.monotonic()-started,Run_Dir=str(directory))
        old.write_json(result_path,result)
        pd.DataFrame(dict(sample_id=dev.sample_id,true_label=truth,predicted_label=predicted)).to_csv(directory/'dev_predictions.csv',index=False)
        old.write_json(directory/'training_log.json',finite_log(trainer.state.log_history))
        all_results.append(result)
        pd.DataFrame([{k:v for k,v in x.items() if k!='hyperparameters'} for x in all_results]).to_csv(OUT/'cross_verifier_run_results.csv',index=False)
        print(f'Completed {run_index}/60: {run_name}; Dev Acc={result["Dev_Accuracy"]:.4f}; MacroF1={result["Dev_Macro_F1"]:.4f}; elapsed={time.monotonic()-start_all:.1f}s',flush=True)
        del prediction,trainer,model;gc.collect();torch.cuda.empty_cache()
    assert len(all_results)==60
    pd.DataFrame([{k:v for k,v in x.items() if k!='hyperparameters'} for x in all_results]).to_csv(OUT/'cross_verifier_run_results.csv',index=False)
    print('All 60 downstream runs completed; no checkpoint or hyperparameter selected using Dev',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','score','select','train'])
    p.add_argument('--model',choices=['qwen','llama'])
    p.add_argument('--device',default='cuda',choices=['cuda'])
    p.add_argument('--score-batch-size',type=int,default=8)
    args=p.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='select':select()
    elif args.stage=='train':train()
    else:
        if not args.model:p.error('score requires --model')
        score(args)


if __name__=='__main__':main()
