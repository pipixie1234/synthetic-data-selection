"""Selection-only diagnostics and final downstream analysis; never tunes selections."""
from __future__ import annotations
import argparse
import collections
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import run_cross_verifier as ex

OUT=ex.OUT
CLASSES=ex.repair.LABELS


def lexical(texts):
    tokens=[re.findall(r'[a-z0-9]+',str(x).lower()) for x in texts]
    unigram=[x for text in tokens for x in text]
    bigram=[(a,b) for text in tokens for a,b in zip(text,text[1:])]
    return dict(distinct1=len(set(unigram))/len(unigram) if unigram else 0,distinct2=len(set(bigram))/len(bigram) if bigram else 0,unique_unigrams=len(set(unigram)),token_n=len(unigram),word_mean=float(np.mean([len(t) for t in tokens])))


def entropy(values):
    frequencies=np.array(list(collections.Counter(values).values()),float)
    probabilities=frequencies/frequencies.sum()
    return float(-(probabilities*np.log(probabilities)).sum())


def datasets():
    result={}
    for path in sorted((OUT/'selected').glob('*.jsonl')):
        result[path.stem]=pd.DataFrame(ex.old.read_jsonl(path))
    return result


def diagnostics():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    manifest=ex.check_frozen();assert manifest['selection_frozen']
    pool=pd.DataFrame(ex.pool_rows())
    scores=pd.read_csv(OUT/'candidate_verifier_scores.csv')
    features=TfidfVectorizer(max_features=8000,min_df=2,stop_words='english',ngram_range=(1,2)).fit_transform(pool.text)
    latent=normalize(TruncatedSVD(n_components=64,n_iter=5,random_state=42).fit_transform(features))
    cluster=np.empty(len(pool),dtype=int)
    for (source,label),indices in pool.groupby(['source','label']).groups.items():
        ix=np.array(list(indices));cluster[ix]=KMeans(n_clusters=10,n_init=10,random_state=42).fit_predict(latent[ix])
    pool['topic_proxy_cluster']=cluster
    pd.DataFrame(dict(sample_id=pool.sample_id,source=pool.source,label=pool.label,topic_proxy_cluster=cluster)).to_csv(OUT/'pool_topic_proxy.csv',index=False)
    pool=pool.merge(scores.drop(columns=['source','label','proximity_score']),on='sample_id',validate='one_to_one')
    sets=datasets();rows=[]
    for name,dataset in sets.items():
        selected=pool[pool.sample_id.isin(dataset.sample_id)]
        for group_name,group in [('all',selected)]+[(f'{s}/{CLASSES[l]}',g) for (s,l),g in selected.groupby(['source','label'])]:
            n=len(group);clusters=list(group[['source','label','topic_proxy_cluster']].itertuples(index=False,name=None))
            row=dict(Dataset=name,Group=group_name,N=n,Mean_RealProximity=float(group.proximity_score.mean()),Mean_NearestSyntheticCosine=float(group.nearest_synthetic_cosine.mean()),NearDuplicate_gt099=float((group.nearest_synthetic_cosine>.99).mean()),TopicProxyEntropy=entropy(clusters),TopicProxyOccupiedClusters=len(set(clusters)),TopicProxyMaxShare=max(collections.Counter(clusters).values())/n,**lexical(group.text))
            for model in ['qwen','llama']:
                row[f'{model}_mean_LabelScore']=float(group[f'{model}_LabelScore'].mean())
                row[f'{model}_mean_LogLabelScore']=float(group[f'{model}_LogLabelScore'].mean())
                row[f'{model}_label_mismatch_rate']=float((group[f'{model}_predicted_label']!=group.class_name).mean())
            row['LegacyCandidate_N']=int(group.is_legacy_candidate.sum())
            row['LegacyCandidate_Fraction']=float(group.is_legacy_candidate.mean())
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'selection_diagnostics.csv',index=False)
    distribution=[]
    patterns=[]
    for (source,label),g in pool.groupby(['source','label']):
        for model in ['qwen','llama']:
            p=g[f'{model}_LabelScore'];lp=g[f'{model}_LogLabelScore']
            distribution.append(dict(Source=source,Class=CLASSES[label],Verifier=model,N=len(g),ScoreMean=p.mean(),ScoreStd=p.std(ddof=1),ScoreMin=p.min(),ScoreP10=p.quantile(.1),ScoreMedian=p.median(),ScoreP90=p.quantile(.9),ScoreMax=p.max(),UniqueProbabilityScores=p.nunique(),SaturatedProbabilityOne=int((p==1).sum()),UniqueLogScores=lp.nunique(),LargestLogScoreTie=int(lp.value_counts().iloc[0]),LogScoreMin=lp.min(),LogScoreMedian=lp.median(),LogScoreMax=lp.max(),LabelMismatch_N=int((g[f'{model}_predicted_label']!=g.class_name).sum()),LabelMismatch_Rate=float((g[f'{model}_predicted_label']!=g.class_name).mean())))
            for predicted,count in g[f'{model}_predicted_label'].value_counts().items():patterns.append(dict(Source=source,IntendedClass=CLASSES[label],Verifier=model,PredictedClass=predicted,N=int(count),Rate=count/500))
    pd.DataFrame(distribution).to_csv(OUT/'score_distributions.csv',index=False)
    pd.DataFrame(patterns).to_csv(OUT/'label_rejection_patterns.csv',index=False)
    overlap=[]
    for left,right in [('SelfVerifier320','CrossVerifier320'),('SelfVerifier320','TopK320'),('CrossVerifier320','TopK320')]:
        for source,label in [(None,None)]+[(s,l) for s in ['qwen','llama'] for l in range(4)]:
            a,b=sets[left],sets[right]
            if source is not None:a=a[(a.source==source)&(a.label==label)];b=b[(b.source==source)&(b.label==label)]
            sa,sb=set(a.sample_id),set(b.sample_id)
            overlap.append(dict(Left=left,Right=right,Source=source or 'all',Class=CLASSES[label] if label is not None else 'all',N_Left=len(sa),N_Right=len(sb),Intersection=len(sa&sb),Union=len(sa|sb),Jaccard=len(sa&sb)/len(sa|sb)))
    pd.DataFrame(overlap).to_csv(OUT/'selected_set_overlap.csv',index=False)
    # Existing Task B pilot is an explicitly limited diagnostic, never a full-pool filter.
    pilot=pd.read_csv(ex.repair.OUT/'paired_results.csv')
    pilot=pilot[(pilot['mode']=='real_anchored')&(pilot.source!='real')]
    pilot.to_csv(OUT/'task_b_frozen_pilot_diagnostic.csv',index=False)
    ex.old.write_json(OUT/'diagnostic_protocol.json',dict(selection_frozen=True,dev_accessed=False,tfidf_max_features=8000,tfidf_ngrams=[1,2],svd_dim=64,clusters_per_source_class=10,cluster_seed=42,topic_proxy_caveat='Lexical TFIDF/SVD clusters are a topic proxy, not annotated topics; joint(source,class,cluster) IDs avoid collisions; all-group entropy compares only identical source/class allocations',lexical_caveat='distinct-n depends on document length; equal sample counts do not equalize token budget',task_b_scope='160 synthetic examples from previous repaired pilot; no full-pool quality ground truth or new quality filtering'))
    status=json.loads((OUT/'diagnostic_protocol.json').read_text())
    status['initial_diagnostics_before_first_dev_access']=not (OUT/'evaluation_audit.json').exists()
    status['this_diagnostic_recompute_used_dev_data']=False
    status['selection_changed_after_dev']=False
    ex.old.write_json(OUT/'diagnostic_protocol.json',status)
    print('Selection diagnostics completed without Dev access or changes to frozen selection',flush=True)


def paired_frames(results):
    random=results[results.Method=='Random320']
    random_mean=random.groupby('Model_Seed')[ex.METRICS].mean()
    arms={method:results[results.Method==method].set_index('Model_Seed')[ex.METRICS] for method in ex.MAIN_METHODS[1:]+ex.SOURCE_METHODS}
    arms['RandomMean']=random_mean
    comparisons=[('CrossVerifier320','SelfVerifier320'),('CrossVerifier320','TopK320'),('CrossVerifier320','RandomMean'),('SelfVerifier320','TopK320'),('SelfVerifier320','RandomMean'),('CrossQwen160','SelfQwen160'),('CrossLlama160','SelfLlama160')]
    paired=[];summary=[]
    for a,b in comparisons:
        delta=arms[a]-arms[b]
        for seed in ex.MODEL_SEEDS:
            row=dict(Comparison=f'{a} - {b}',Left=a,Right=b,Model_Seed=seed)
            for metric in ex.METRICS:
                row[metric+'_Delta']=float(delta.loc[seed,metric])
                row[metric+'_Left']=float(arms[a].loc[seed,metric])
                row[metric+'_Right']=float(arms[b].loc[seed,metric])
            paired.append(row)
        for metric in ex.METRICS:
            d=delta[metric]
            std=d.std(ddof=1);margin=2.7764451051977987*std/math.sqrt(5)
            summary.append(dict(Comparison=f'{a} - {b}',Metric=metric,DeltaMean=float(d.mean()),DeltaStd=float(std),CI95Low=float(d.mean()-margin),CI95High=float(d.mean()+margin),PositiveSeeds=int((d>0).sum()),NegativeSeeds=int((d<0).sum()),N_ModelSeeds=5))
    method_rows=[]
    for method,frame in arms.items():
        for metric in ex.METRICS:
            method_rows.append(dict(Method=method,Metric=metric,Mean=float(frame[metric].mean()),Std=float(frame[metric].std(ddof=1)),N_ModelSeeds=5,Aggregation='mean within model seed across five selections' if method=='RandomMean' else 'one frozen selection, five model seeds'))
    for metric in ex.METRICS:method_rows.append(dict(Method='Random320_All25Runs',Metric=metric,Mean=float(random[metric].mean()),Std=float(random[metric].std(ddof=1)),N_ModelSeeds=5,Aggregation='descriptive std over25 crossed runs; not25 independent model seeds'))
    return pd.DataFrame(paired),pd.DataFrame(summary),pd.DataFrame(method_rows),random_mean.reset_index()


def signal_class(summary):
    core={x:summary[(summary.Comparison==f'CrossVerifier320 - {x}')&(summary.Metric=='Dev_Macro_F1')].iloc[0] for x in ['SelfVerifier320','TopK320','RandomMean']}
    a,b,c=core['SelfVerifier320'],core['TopK320'],core['RandomMean']
    if a.DeltaMean>=.01 and b.DeltaMean>=.01 and c.DeltaMean>0 and a.PositiveSeeds>=4 and b.PositiveSeeds>=4:return 'Strong signal'
    if a.DeltaMean>=.005 and b.DeltaMean>=.005 and c.DeltaMean>0:return 'Weak signal'
    return 'No useful signal'


def fmt(x):return f'{100*x:.2f}'


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=pd.read_csv(OUT/'cross_verifier_run_results.csv')
    assert len(results)==60
    assert not results.duplicated(['Method','Selection_Seed','Model_Seed']).any()
    for seed,g in results.groupby('Model_Seed'):
        assert g.classifier_init_sha256.nunique()==1,'Paired classifier initialization differs'
    paired,delta_summary,method_summary,random_mean=paired_frames(results)
    paired.to_csv(OUT/'paired_model_seed_deltas.csv',index=False)
    delta_summary.to_csv(OUT/'paired_delta_summary.csv',index=False)
    method_summary.to_csv(OUT/'method_summary.csv',index=False)
    random_mean.to_csv(OUT/'random_mean_by_model_seed.csv',index=False)
    signal=signal_class(delta_summary)
    utility={}
    for method in ['SelfVerifier320','CrossVerifier320']:
        contrasts=[delta_summary[(delta_summary.Comparison==f'{method} - {baseline}')&(delta_summary.Metric=='Dev_Macro_F1')].iloc[0] for baseline in ['TopK320','RandomMean']]
        utility[method]='Strong signal' if all(x.DeltaMean>=.01 and x.PositiveSeeds>=4 for x in contrasts) else 'Weak signal' if all(x.DeltaMean>=.005 for x in contrasts) else 'No useful signal'
    ex.old.write_json(OUT/'conclusion.json',dict(signal=signal,selection_utility_by_fixed_method=utility,interpretation='Strong requires Cross-Self and Cross-TopK >=1pp MacroF1, >=4/5 positive paired model seeds for both, and Cross-RandomMean >0; Weak requires both contrasts >=0.5pp with Cross-RandomMean >0; otherwise No useful signal. Standalone selection utility separately compares each fixed verifier with both TopK and RandomMean.',method_tuning_after_dev=False))
    diagnostics_frame=pd.read_csv(OUT/'selection_diagnostics.csv')
    overlap=pd.read_csv(OUT/'selected_set_overlap.csv')
    distributions=pd.read_csv(OUT/'score_distributions.csv')
    text=['# Synthetic-data cross-verifier downstream 验证','','## 最终结论','','本轮固定研究问题：在同一 4,000 条 Qwen/Llama synthetic candidate pool 上，LLM verifier 排序是否带来比 Random 和既有 Real-Anchored semantic TopK 更好的 downstream utility，以及 cross 是否优于 self。全部选集先冻结，再在 Dev2000 上评估；没有结果驱动的方法调整。','','主比较每组严格 320 条（每 source×class 40）；Random 使用 selection seeds 41–45 × model seeds 42–46，共25次；其余三个主方法各5次。额外 source-only self/cross 对照共20次，每次160条，作为不同 budget 的辅助实验。共60次 BERT ordinary CE 训练。','','| 方法 | Dev Accuracy mean±std (%) | Dev Macro-F1 mean±std (%) |','|---|---:|---:|']
    for method in ['RandomMean','TopK320','SelfVerifier320','CrossVerifier320']:
        a=method_summary[(method_summary.Method==method)&(method_summary.Metric=='Dev_Accuracy')].iloc[0]
        f=method_summary[(method_summary.Method==method)&(method_summary.Metric=='Dev_Macro_F1')].iloc[0]
        text.append(f'| {method} | {fmt(a.Mean)} ± {fmt(a.Std)} | {fmt(f.Mean)} ± {fmt(f.Std)} |')
    text+=['','RandomMean 的 std 是先在每个 model seed 内对5个 selection seeds取均值，再对5个 model seeds求std，便于与 deterministic arms配对。全部25次Random的描述性std以及每个selection/model seed的原始结果均另行保留，未把25次视为25个独立model seeds。','','## 主比较的逐 model-seed Macro-F1 delta','','所有 delta 单位为百分点；正值代表左侧方法更好。','','| 比较 | Seed42 | Seed43 | Seed44 | Seed45 | Seed46 | Mean±std | 正向 seeds |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for comparison in paired.Comparison.unique()[:5]:
        g=paired[paired.Comparison==comparison].set_index('Model_Seed')
        x=delta_summary[(delta_summary.Comparison==comparison)&(delta_summary.Metric=='Dev_Macro_F1')].iloc[0]
        values=[f"{g.loc[s,'Dev_Macro_F1_Delta']*100:+.2f}" for s in ex.MODEL_SEEDS]
        text.append('| '+comparison+' | '+' | '.join(values)+f' | {x.DeltaMean*100:+.2f} ± {x.DeltaStd*100:.2f} | {int(x.PositiveSeeds)}/5 |')
    text+=['','Accuracy 和四类 F1 的全部配对 delta 与 t(df=4) 95% 区间见 Excel/CSV。区间是5个model seeds的探索性描述；单次池/固定deterministic selection缺少跨数据池重复，不能据此宣称统计定律。','','## 每类 downstream F1','','| 方法 | World F1 (%) | Sports F1 (%) | Business F1 (%) | SciTech F1 (%) |','|---|---:|---:|---:|---:|']
    for method in ['RandomMean','TopK320','SelfVerifier320','CrossVerifier320']:
        values=[method_summary[(method_summary.Method==method)&(method_summary.Metric==f'Dev_{c}_F1')].iloc[0].Mean for c in CLASSES]
        text.append('| '+method+' | '+' | '.join(fmt(x) for x in values)+' |')
    text+=['','## Source-only 辅助对照','','这部分 source×class 每组仍选40条，但仅保留一个source，共160条，预算小于主比较。只在同source的 self/cross之间配对，不拿其绝对性能与320方法公平比较。','','| 比较 | Seed42 | Seed43 | Seed44 | Seed45 | Seed46 | Macro-F1 mean delta (pp) | 正向 seeds |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for comparison in paired.Comparison.unique()[5:]:
        g=paired[paired.Comparison==comparison].set_index('Model_Seed')
        x=delta_summary[(delta_summary.Comparison==comparison)&(delta_summary.Metric=='Dev_Macro_F1')].iloc[0]
        text.append('| '+comparison+' | '+' | '.join(f"{g.loc[s,'Dev_Macro_F1_Delta']*100:+.2f}" for s in ex.MODEL_SEEDS)+f' | {x.DeltaMean*100:+.2f} | {int(x.PositiveSeeds)}/5 |')
    text+=['','source-only差异是在不同的160条训练预算下的辅助证据。主比较混合两source，不能把主比较的每类变化直接因果归到某一source。','','## Selection diagnostics','','| 比较 | Selected交集/并集 | Jaccard |','|---|---:|---:|']
    for _,x in overlap[overlap.Source=='all'].iterrows():text.append(f'| {x.Left} vs {x.Right} | {x.Intersection}/{x.Union} | {x.Jaccard:.4f} |')
    text+=['','各source×class的40条选集 overlap、score分布、selected mean Real-Proximity、distinct1/2、词汇topic-proxy entropy/占簇数、near-duplicate和标签拒绝模式见Excel及独立CSV。所有diagnostics在第一次Dev访问前计算；没有因diagnostic好坏调整选择规则。','','| Dataset | selected mean Real-Proximity | Distinct1 | Distinct2 | Mean nearest-synthetic cosine |','|---|---:|---:|---:|---:|']
    for name in ['TopK320','SelfVerifier320','CrossVerifier320']+[f'Random320_sel{s}' for s in ex.SELECTION_SEEDS]:
        x=diagnostics_frame[(diagnostics_frame.Dataset==name)&(diagnostics_frame.Group=='all')].iloc[0]
        text.append(f'| {name} | {x.Mean_RealProximity:.6f} | {x.distinct1:.4f} | {x.distinct2:.4f} | {x.Mean_NearestSyntheticCosine:.4f} |')
    text+=['','Topic proxy：只在冻结candidate文本上fit TFIDF(1–2gram, max8000)、SVD64和每source×class的10簇KMeans(seed42)。这是词汇主题代理，未经人工topic验证。簇数字在组间重复，因此topic entropy只在同source×class内解释。Distinct-n受文本长度影响；equal samples不等于equal tokens。','','## 协议、工程处理与边界','','- Verifier：本地Qwen3-4B-Instruct-2507与Llama-3.2-3B-Instruct，沿用已冻结real-anchored prompt与每类2条真实demonstrations；没有presented label/source/id；native chat中不重复添加BOS。','- 每条计算四个完整候选标签序列（包括前置空格和结束换行）的log likelihood，主排序是intended label的归一化LabelScore。没有平均两个模型的score scale；仅source×class内排序。','- 为防止近1的概率舍入成1而丢失排序，实际使用其数学上单调等价的stable log(LabelScore)：对非最大候选概率质量使用log1p。没有添加额外selection feature或阈值；精确同分按sample_id升序。原始likelihood、probability和稳定log score均保存。','- 复用每模型160条pilot synthetic评分，逐条核对text-derived prompt hash；其余3840条批量评分。复用batch2，新增batch4；同一FP16 checkpoint，不做量化。不同batch的微小数值差异是工程限制，完整cache和runtime留存。','- full-pool标签拒绝模式由four-candidate full-sequence argmax计算；不是额外的free-generation label判断。排序没有用LabelPass作硬过滤，也没有用Task B过滤。','- Task B只保留之前修复pilot的160条synthetic诊断，不重新对4000条做质量判断，不将其视为选集质量的无偏估计。','- TopK直接使用4000候选CSV里已冻结的proximity_score，source×class Top40；没有重调encoder、reference或scorer。','- 下游：本地bert-base-uncased，3 epochs，batch16，lr2e-5，weight_decay0.01，max_length256，linear scheduler，warmup0，FP16，AdamW torch，gradient clip1，ordinary CE；所有主方法完全相同。','- model seed在模型加载前设置，确保同seed的classification head初始化在所有方法之间完全相同（hash已核对）。历史训练helper是在model初始化之后才通过Trainer设置seed，本轮使用专门wrapper并把所有主方法重新训练，没有混用历史Dev性能。','- Dev2000只在全部选集冻结之后加载，未在训练中按epoch评价、未选best checkpoint、未早停。每次固定训练后仅一次Dev预测；验证candidate/demo与Dev无规范化exact-text重叠，记录Dev hash/IDs/class counts。','- Final Holdout未读取或评估；未生成synthetic。Phi-3本地不可用，跳过可选分支；All4000未运行，保持主比较预算一致。','','## 必须回答的六个问题','']
    def contrast(a,b):return delta_summary[(delta_summary.Comparison==f'{a} - {b}')&(delta_summary.Metric=='Dev_Macro_F1')].iloc[0]
    cr,ct,cs=contrast('CrossVerifier320','RandomMean'),contrast('CrossVerifier320','TopK320'),contrast('CrossVerifier320','SelfVerifier320')
    sr,st=contrast('SelfVerifier320','RandomMean'),contrast('SelfVerifier320','TopK320')
    sq,sl=contrast('CrossQwen160','SelfQwen160'),contrast('CrossLlama160','SelfLlama160')
    class_effects='，'.join(f"{c} {delta_summary[(delta_summary.Comparison=='CrossVerifier320 - SelfVerifier320')&(delta_summary.Metric==f'Dev_{c}_F1')].iloc[0].DeltaMean*100:+.2f}pp" for c in CLASSES)
    proceed=signal=='Strong signal' or any(x=='Strong signal' for x in utility.values())
    text += [f"1. Verifier是否优于Random？Cross−RandomMean={cr.DeltaMean*100:+.2f}pp（{int(cr.PositiveSeeds)}/5正向）；Self−RandomMean={sr.DeltaMean*100:+.2f}pp（{int(sr.PositiveSeeds)}/5正向）。按该固定数据与训练协议报告，不能把小幅单次差异称为稳定优势。",f"2. 是否优于旧TopK？Cross−TopK={ct.DeltaMean*100:+.2f}pp；Self−TopK={st.DeltaMean*100:+.2f}pp。各固定方法同时相对Random/TopK的selection utility归类：Self={utility['SelfVerifier320']}，Cross={utility['CrossVerifier320']}。",f"3. Cross是否优于Self？平均={cs.DeltaMean*100:+.2f}pp，{int(cs.PositiveSeeds)}/5正向。",f"4. 是否有source/class-specific effect？辅助Qwen-source Cross−Self={sq.DeltaMean*100:+.2f}pp（{int(sq.PositiveSeeds)}/5正向）；Llama-source={sl.DeltaMean*100:+.2f}pp（{int(sl.PositiveSeeds)}/5正向）。主混合320的Cross−Self逐类F1变化：{class_effects}。辅助budget为160，不能直接外推混合320的因果source贡献。",f'5. 总体cross-verification归类：**{signal}**。Strong要求Cross相对Self和TopK都≥1pp Macro-F1、两项都至少4/5正向且Cross优于RandomMean；Weak要求两项都≥0.5pp且Cross优于RandomMean；其余为No useful signal。这是对用户pilot heuristic的预先固定保守解释。选择效用与cross优势为分别预先定义的研究问题，不按结果挑选新方法。',f"6. 是否值得进入iterative verifier-overoptimization？{'已有固定方法的选择效用强信号，可进入小规模受控试验，但应分别确认cross优势是否成立，并保留独立审计集与固定基线。' if proceed else '目前不建议直接进入。先在独立数据池/人工核验样本上确认选择效用；本轮不根据Dev反馈继续改prompt或selection救结果。'}",'','全部运行/选集/评分hash、60次run-level结果、selection diagnostics、配对delta和原始Dev预测保留。']
    text=[line.replace('簇数字在组间重复，因此topic entropy只在同source×class内解释。','使用(source,class,cluster)联合ID避免簇编号碰撞；全选集entropy只在source/class allocation相同的主方法之间比较。') for line in text]
    implementation_counts={}
    for verifier in ['qwen','llama']:
        raw=ex.old.read_jsonl(OUT/f'{verifier}_scores.jsonl')
        implementation_counts[verifier]=dict(pilot=sum(bool(x.get('reused_from_repaired_pilot')) for x in raw),shared_cache=sum(x.get('scoring_implementation')=='shared_prefix_cache' for x in raw))
        implementation_counts[verifier]['earlier_full_sequence']=4000-implementation_counts[verifier]['pilot']-implementation_counts[verifier]['shared_cache']
    marker=text.index('## 协议、工程处理与边界')
    text[marker+1:marker+1]=['',f'评分实现审计：最初新增batch4的full-sequence teacher forcing，随后复用prefix KV cache并使用batch8计算各候选的完整序列概率。它减少重复prefill，不改变prompt、候选序列或分数公式；两种模型结构的CPU逐token单元对照及实际FP16 checkpoint对照已通过。保留的有效评分分布：{json.dumps(implementation_counts,ensure_ascii=False)}。原始runtime历史与cache评分审计均保存。']
    text[marker+3:marker+3]=['','生成 provenance 限制：4000池中640条为legacy candidates，精确generation checkpoint未记录；本轮Self/Cross按既有Qwen/Llama source-family身份定义，不能保证legacy部分为同一精确checkpoint的self-verification。未为解决这个限制筛除任何候选；各选集legacy数量/比例列入selection diagnostics。']
    text=[line.replace('其余3840条批量评分。复用batch2，新增batch4；','其余3840条批量评分。复用batch2，新增batch4/8（含shared prefix cache）；') for line in text]
    text=[line.replace('所有diagnostics在第一次Dev访问前计算；没有因diagnostic好坏调整选择规则。','初始selection diagnostics在第一次Dev访问前计算；后续数据provenance审计及簇ID碰撞修正只重算diagnostics，没有读取Dev数据或改变任何选集/规则。') for line in text]
    text[marker+5:marker+5]=['','运行恢复：FP16 AMP的grad_norm诊断出现inf时，严格JSON日志序列化中断；改为将非有限诊断值保存成明确字符串，未修改训练精度或hyperparameters。受影响的Random320 sel41/model43完整重跑一次，首轮attempt归档；其余已完成且hash一致的运行复用。最终metrics均为有限值。']
    for i,line in enumerate(text):
        if line.startswith('1. Verifier是否优于Random？'):text[i]=line.replace('？Cross','？本轮是：Cross')
        elif line.startswith('2. 是否优于旧TopK？'):text[i]=line.replace('？Cross','？否：Cross')
        elif line.startswith('3. Cross是否优于Self？'):text[i]=line.replace('？平均','？本轮是：平均')
        elif line.startswith('4. 是否有source/class-specific effect？'):text[i]=line.replace('？辅助','？存在不同响应；辅助')
    if (OUT/'final_validation.json').exists():
        text[marker+7:marker+7]=['','独立最终验证已通过：核对12个选集hash、8000组模型评分、60次完整运行与同seed分类头初始化；从保存的120000条Dev预测重新计算Accuracy、Macro-F1和四类F1，核对35行paired model-seed deltas。final_validation.json保留证据。']
    (OUT/'cross_verifier_report.md').write_text('\n'.join(text)+'\n')
    with pd.ExcelWriter(OUT/'cross_verifier_summary.xlsx',engine='openpyxl') as writer:
        for name,frame in [('MethodSummary',method_summary),('RunResults',results),('PairedSeedDeltas',paired),('PairedDeltaSummary',delta_summary),('RandomMeanByModel',random_mean),('SelectionDiagnostics',diagnostics_frame),('SetOverlap',overlap),('ScoreDistributions',distributions),('LabelPatterns',pd.read_csv(OUT/'label_rejection_patterns.csv')),('SelectionManifest',pd.read_csv(OUT/'selection_manifest.csv'))]:
            frame.to_excel(writer,sheet_name=name,index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
            for column in sheet.columns:sheet.column_dimensions[column[0].column_letter].width=min(50,max(15,len(str(column[0].value))+2))
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    methods=['RandomMean','TopK320','SelfVerifier320','CrossVerifier320']
    means=[method_summary[(method_summary.Method==x)&(method_summary.Metric=='Dev_Macro_F1')].iloc[0].Mean for x in methods]
    stds=[method_summary[(method_summary.Method==x)&(method_summary.Metric=='Dev_Macro_F1')].iloc[0].Std for x in methods]
    axes[0].bar(range(4),np.array(means)*100,yerr=np.array(stds)*100,color=['#a4a7ad','#e7a048','#4d86bc','#5c9d79'],capsize=4)
    axes[0].set(xticks=range(4),xticklabels=['Random','TopK','Self','Cross'],ylabel='Dev Macro-F1 (%)',title='Main methods: 320 training samples')
    for i,v in enumerate(means):axes[0].text(i,v*100+stds[i]*100+.5,f'{v:.2%}',ha='center',fontsize=9)
    for comparison,color in [('CrossVerifier320 - SelfVerifier320','#4d86bc'),('CrossVerifier320 - TopK320','#e7a048'),('CrossVerifier320 - RandomMean','#5c9d79')]:
        g=paired[paired.Comparison==comparison].sort_values('Model_Seed')
        axes[1].plot(g.Model_Seed,g.Dev_Macro_F1_Delta*100,marker='o',color=color,label=comparison.replace('Verifier320','').replace('320',''))
    axes[1].axhline(0,color='gray',linewidth=1);axes[1].set(xlabel='Paired model seed',ylabel='Macro-F1 delta (pp)',title='Cross relative to fixed baselines');axes[1].legend(fontsize=8)
    axes[1].set_xticks(ex.MODEL_SEEDS)
    for comparison,color in [('CrossQwen160 - SelfQwen160','#4d86bc'),('CrossLlama160 - SelfLlama160','#e7a048')]:
        g=paired[paired.Comparison==comparison].sort_values('Model_Seed')
        axes[2].plot(g.Model_Seed,g.Dev_Macro_F1_Delta*100,marker='o',color=color,label=comparison.replace('160',''))
    axes[2].axhline(0,color='gray',linewidth=1);axes[2].set(xlabel='Paired model seed',ylabel='Macro-F1 delta (pp)',title='Source-only auxiliary (160 samples)');axes[2].legend(fontsize=8)
    axes[2].set_xticks(ex.MODEL_SEEDS)
    fig.savefig(OUT/'cross_verifier_summary.png',dpi=180);plt.close(fig)
    print('Final signal: '+signal,flush=True)
    print(method_summary[(method_summary.Metric=='Dev_Macro_F1')].to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['diagnostics','report']);args=parser.parse_args()
    diagnostics() if args.stage=='diagnostics' else report()
