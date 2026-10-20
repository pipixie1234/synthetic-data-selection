"""Chinese experiment report, Excel workbook and exportable static figure."""
from pathlib import Path
import json

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent/'outputs'
LABELS=['World','Sports','Business','SciTech']


def pct(x): return f'{100*x:.1f}%'
def num(x): return '未定义' if x is None else f'{x:.4f}'


def main():
    m=json.loads((OUT/'metrics.json').read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    pairs=pd.read_csv(OUT/'paired_results.csv')
    classes=pd.read_csv(OUT/'class_label_results.csv')
    baseline=pd.read_csv(OUT.parent.parent/'redundancy/outputs/paired_results.csv')
    baseline=baseline[(baseline.subset=='main')&(baseline.source=='real')]
    real=pairs[pairs.source=='real']
    modes=m['modes']
    rows=[dict(condition='previous_combined_task',**m['baseline_combined_task_real'])]+[dict(condition=mode,**modes[mode]['real']) for mode in ['zero_shot','real_anchored']]
    pd.DataFrame(rows).to_csv(OUT/'real_summary.csv',index=False,encoding='utf-8-sig')
    text=['# Verifier Repair：独立标签/质量任务与真实示例对照','','实验日期：2026-10-06。复用上一轮完全相同的 200 条主样本（160 synthetic + 40 RealReference），未重新抽样、未训练 BERT、未生成新闻文本，也未读取最终 holdout。','','## 核心结果','','| 条件 | Qwen Real LabelAccuracy | Llama Real LabelAccuracy | 共同错误 / 错误并集 | Real SharedErrorJaccard |','|---|---:|---:|---:|---:|']
    for x in rows:
        text.append(f"| {x['condition']} | {40-x['qwen_errors']}/40（{pct(x['qwen_accuracy'])}） | {40-x['llama_errors']}/40（{pct(x['llama_accuracy'])}） | {x['shared_errors']}/{x['error_union']} | {num(x['shared_error_jaccard'])} |")
    text+=['','SharedErrorJaccard 的错误集合定义为：模型在 40 条真实样本上的 predicted_label 与 AG News 原始标签不同。此处与上一轮的错标签 challenge rejection 错误是不同指标，不能直接比较那一轮 challenge 的 0.200。上表的 previous_combined_task 重新取上一轮这同一批 40 条真实文本的 label prediction 作基线。','','## Real 样本逐类表现','','| 类别（各 n=10） | 旧 Qwen | Zero Qwen | Anchored Qwen | 旧 Llama | Zero Llama | Anchored Llama |','|---|---:|---:|---:|---:|---:|---:|']
    for label in LABELS:
        values=[]
        for model in ['qwen','llama']:
            g=baseline[baseline.true_label==label]
            values.append(f'{sum(g[f"{model}_predicted_label"]==label)}/10')
            for mode in ['zero_shot','real_anchored']:
                g=classes[(classes['mode']==mode)&(classes.model==model)&(classes.source=='real')&(classes.label==label)]
                values.append(f'{int(g.iloc[0].correct_labels)}/10')
        text.append('| '+label+' | '+' | '.join(values)+' |')
    text+=['','## 固定协议','','Task A 只预测 label。zero-shot 给四类定义；real-anchored 在同一份定义上增加每类 2 条、共 8 条 RealReference demonstrations。示例从冻结 reference training IDs 中按固定 seed 抽取，按 original ID 和规范化文本双重排除本轮 200 条评测文本；两个模型共用示例与示例顺序。没有用评测正确率筛选示例，也没有给模型目标文本的 presented label 或 source。','','Task B 在独立上下文中仅评价文本，输出 news_like、coherent、plausible、template_like。每个模型在这 200 条上只跑一次 Task B，两种 Task A 设置共用它，避免将标签示例对质量判断的影响混入对照。Task B 无真实质量 ground truth，因此输出仅为描述性诊断。','','PASS 全部由代码计算：','','```text','LabelPass = (predicted_label == presented_label)','OverallPass = LabelPass AND news_like AND plausible','```','','coherent 与 template_like 单独保留，不悄悄加入 OverallPass 门槛。没有任何模型输出的 PASS 字段；因此 predicted_label 与 presented_label 不同却 PASS 的逻辑矛盾为 0。','','两模型使用相同 user prompts、相同样本与顺序，但用各自 native chat wrapper。Task A assistant 预填 `predicted_label:` 后进行 greedy continuation；返回一行，Task B 返回四行。每条独立上下文，FP16、do_sample=False、label max_new_tokens=16、quality max_new_tokens=64。','','demonstrations 保留 AG News 原始标签，未按模型判断重新清洗；部分例子存在 Business/SciTech 类别边界（如科技公司并购报道被原数据标为 SciTech）。因此本次结果仅针对固定的这 8 个示例，不能等同于示例选择的最优结果。','','## Real-anchored 相对 Zero-shot 的配对变化','','| 模型 | 错→对 | 对→错 | Accuracy 差值 | Exact McNemar two-sided p |','|---|---:|---:|---:|---:|']
    for x in m['real_paired_transitions']:
        text.append(f"| {x['model']} | {x['wrong_to_correct']} | {x['correct_to_wrong']} | {x['accuracy_delta']*100:+.1f} 个百分点 | {x['mcnemar_exact_two_sided_p']:.4f} |")
    text+=['','每个模型的准确率 Wilson 95% 区间：','']
    for mode in ['zero_shot','real_anchored']:
        for model in ['qwen','llama']:
            lo,hi=modes[mode]['real'][f'{model}_accuracy_wilson95']
            text.append(f'- {mode}/{model}: [{pct(lo)}, {pct(hi)}]。')
    text+=['','这些是 n=40 的探索性配对对照，逐类仅 10 条。p 值未作多重比较校正；指标改善不自动代表稳定的泛化效果。旧任务到新任务同时改变了任务拆分、类别定义、隐藏 presented label 和输出格式，不能把旧到新的全部改善单独归因于其中某一项。','','## 连续分数：候选 label log probability','','不用 self-reported confidence，也不需要 K 次随机采样。对于每条 Task A 输入，模型在 `predicted_label:` assistant 前缀后分别计算 ` World\n`、` Sports\n`、` Business\n`、` SciTech\n` 的完整 autoregressive log probability，再对四个候选做 softmax：','','```text','s(y) = sum_t log P(token_t | prompt, prefix, previous candidate tokens)','LabelScore = exp(s(presented_label)) / sum_{y in four labels} exp(s(y))','```','','所有标签 token 和结束换行都计入；SciTech 多 token 时不只取首 token，也不进行长度平均。这个值是四个限定候选中的归一化序列概率，未做概率校准，受候选字符串、tokenization 和 prompt 影响，不能直接叫做真实 P(correct)。测试独立核对了批量左 padding 的评分与逐 token 的 next-token reference。','','Task A 的主预测仍是实际 greedy label 输出；候选 full-sequence argmax 作为单独诊断保留，未替换原始预测。','','| 条件 | Qwen unique scores / 200 | Llama unique scores / 200 | Bottom20% 交集 / 并集 | Bottom20% Jaccard |','|---|---:|---:|---:|---:|']
    for mode in ['zero_shot','real_anchored']:
        x=modes[mode]
        d=x['score_diagnostics'];b=x['matched_budget']
        text.append(f"| {mode} | {d['qwen']['unique_scores']} | {d['llama']['unique_scores']} | {b['intersection']}/{b['union']} | {num(b['jaccard'])} |")
    text+=['','每模型固定拒绝 40/200 条，按 LabelScore 升序；精确同分时按 source-blind SHA256(seed:item_id) 排序。这是 label score 的排序，不加入 Task B 的布尔质量判断。','']
    for mode in ['zero_shot','real_anchored']:
        for model in ['qwen','llama']:
            d=modes[mode]['score_diagnostics'][model]
            text.append(f"- {mode}/{model}：截止分数={d['cutoff']:.12g}，截止同分={d['tied_at_cutoff']} 条；greedy 与 full-sequence argmax 不同={d['greedy_argmax_disagreements']} 条；Real argmax accuracy={pct(d['real_candidate_argmax_accuracy'])}。")
    text+=['','## 200 条主样本的代码 PASS 一致性','','| 条件 | LabelPass agreement | LabelPass κ | Label reject Jaccard | OverallPass agreement | OverallPass κ | Overall reject Jaccard |','|---|---:|---:|---:|---:|---:|---:|']
    for mode in ['zero_shot','real_anchored']:
        a,b=modes[mode]['label_pass'],modes[mode]['overall_pass']
        text.append(f"| {mode} | {pct(a['agreement'])} | {num(a['kappa'])} | {num(a['reject_jaccard'])} | {pct(b['agreement'])} | {num(b['kappa'])} | {num(b['reject_jaccard'])} |")
    text+=['','Synthetic 的标签来源于既有生成标签，没有人工逐条真实 ground truth，因此 synthetic 的 LabelPass rate 只代表与 presented label 一致率。核心 accuracy 与 shared errors 只以 40 条 RealReference 为依据。','','## 独立文本质量结果','','| Model | Source | n | News-like yes | Coherent yes | Plausible yes | Template-like yes |','|---|---|---:|---:|---:|---:|---:|']
    for x in m['quality']:
        text.append(f"| {x['model']} | {x['source']} | {x['n']} | {pct(x['news_like_yes_rate'])} | {pct(x['coherent_yes_rate'])} | {pct(x['plausible_yes_rate'])} | {pct(x['template_like_yes_rate'])} |")
    text+=['','## 解读与下一步','','这轮首先回答 verifier 修复是否改善同一批真实文本的类别判断。若拆分后准确率提高，应优先采用代码 LabelPass，继续在新的人工核验真实样本上确认；本轮不训练下游 BERT，也不据此断言 cross-verification 提升下游性能。','']
    gains=[x['accuracy_delta'] for x in m['real_paired_transitions']]
    if all(g>0 for g in gains):
        text.append('本轮 real-anchored 相对 zero-shot 两模型都提高，但固定少量 examples 与 n=40 限制仍在；是否有可靠增益应结合配对变化和区间。')
    elif any(g>0 for g in gains):
        text.append('Real-anchored 的收益随模型而异；不能把少量真实示例视为统一有效的修复措施。应分别选择配置并用新样本确认。')
    else:
        text.append('本轮固定 real anchors 未带来两模型真实样本准确率提高，不能按这个配置认定 anchoring 有益。独立标签任务的改善与 demos 的增量作用应分开判断。')
    x=modes['real_anchored']['real']
    if x['error_union']==0:
        text.append('Anchored 下双方没有 Real label errors，SharedErrorJaccard 为 0/0 未定义；不能视为错误不共享。')
    else:
        text.append(f"Anchored 的 Real shared errors 为 {x['shared_errors']} 条，错误并集 {x['error_union']} 条。重叠率应同时结合错误集合大小，低重叠不自动代表有效互补。")
    error_sets=pd.read_csv(OUT/'real_error_membership.csv')
    shared_review=error_sets[error_sets.real_anchored_shared_error].copy()
    shared_review.to_csv(OUT/'remaining_shared_errors.csv',index=False,encoding='utf-8-sig')
    text+=['','从 zero-shot 到 real-anchored，共同错误的绝对数量下降，但错误并集也下降，所以 Jaccard 可以不下降；不能把这一点单独解释成冗余增加。','','剩余共同错误见 remaining_shared_errors.csv。至少部分样本涉及类别边界或疑似原始标注问题，例如 “Boom times ahead for smartphones” 在原数据中标为 World，“FDA Researcher Says Supervisors Delayed Vioxx Study” 标为 Business。当前指标严格按原始标签计算，未重新标注；这些案例建议以后人工核验，不能未经核验就把所有 shared errors 视为模型共同犯了确定的错误。']
    text+=['','## 交付文件与复现','','- verifier_repair_results.xlsx：Real 汇总、逐类结果、配对变化、质量输出、200×2 条判断、分数与固定预算集合。','- paired_results.csv：各条件的预测、四个质量属性、代码 PASS 与四类概率。','- *_labels.jsonl / *_quality.jsonl：模型原始输出与解析；*_scores.jsonl：完整候选 logprob 与 probability。','- real_anchors.csv / *_prompts.jsonl：冻结真实示例与模型输入。','- manifest.json / *_runtime.json：相同数据、示例排除、prompt hashes 与运行设置。','- verifier_repair_summary.png：可导出的真实准确率、逐类结果和 score 分布图。','','```bash','/opt/anaconda/bin/python SyntheticData/redundancy_repair/run_repair.py prepare','env LD_LIBRARY_PATH=/opt/anaconda/lib CUDA_VISIBLE_DEVICES=0 /opt/anaconda/bin/python SyntheticData/redundancy_repair/run_repair.py verify --model qwen','env LD_LIBRARY_PATH=/opt/anaconda/lib CUDA_VISIBLE_DEVICES=0 /opt/anaconda/bin/python SyntheticData/redundancy_repair/run_repair.py verify --model llama','env LD_LIBRARY_PATH=/opt/anaconda/lib /opt/anaconda/bin/python SyntheticData/redundancy_repair/run_repair.py analyze','env LD_LIBRARY_PATH=/opt/anaconda/lib /opt/anaconda/bin/python SyntheticData/redundancy_repair/report_repair.py','env LD_LIBRARY_PATH=/opt/anaconda/lib /opt/anaconda/bin/python -m unittest discover -s SyntheticData/redundancy_repair -p "test_*.py" -v','```','','已有结果可断点复用；数据、prompt 与 runtime hash 不一致时拒绝混用。']
    marker=text.index('## 固定协议')
    text[marker+1:marker+1]=['','Tokenization 审计：初次 Llama generation 因 tokenizer 在 native chat template 之外额外加入 BOS，与候选 logprob 的输入不一致，已完整归档排除（attempt1_llama_double_bos）；最终 Llama 的 600 条独立判断全部使用 add_special_tokens=False 重跑，原先正确的候选评分复用。Qwen 默认和显式 tokenization 的 token IDs 完全一致，其判断可保留。新旧任务比较也包含该 tokenization 修正，不能把所有增益单独归因于任务拆分。']
    (OUT/'verifier_repair_report.md').write_text('\n'.join(text)+'\n')
    with pd.ExcelWriter(OUT/'verifier_repair_results.xlsx',engine='openpyxl') as writer:
        pd.DataFrame(rows).to_excel(writer,sheet_name='RealSummary',index=False)
        classes.to_excel(writer,sheet_name='ClassResults',index=False)
        pd.DataFrame(m['real_paired_transitions']).to_excel(writer,sheet_name='PairedTransitions',index=False)
        pd.DataFrame(m['quality']).to_excel(writer,sheet_name='QualitySummary',index=False)
        real.to_excel(writer,sheet_name='RealPredictions',index=False)
        pairs.to_excel(writer,sheet_name='AllPredictions',index=False)
        pd.read_csv(OUT/'matched_budget_membership.csv').to_excel(writer,sheet_name='MatchedBudget',index=False)
        pd.read_csv(OUT/'real_anchors.csv').to_excel(writer,sheet_name='RealAnchors',index=False)
        pd.read_csv(OUT/'real_error_membership.csv').to_excel(writer,sheet_name='RealErrorSets',index=False)
        pd.read_csv(OUT/'real_confusion_matrices.csv').to_excel(writer,sheet_name='RealConfusion',index=False)
        shared_review.to_excel(writer,sheet_name='RemainingSharedErrors',index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width=min(60,max(14,len(str(col[0].value))+2))
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    xpos=np.arange(3)
    for model,offset,color in [('qwen',-.17,'#3575ad'),('llama',.17,'#e69b40')]:
        values=[x[f'{model}_accuracy'] for x in rows]
        axes[0].bar(xpos+offset,values,width=.34,color=color,label=model)
        for i,v in enumerate(values): axes[0].text(i+offset,v+.025,f'{v:.1%}',ha='center',fontsize=9)
    axes[0].set(xticks=xpos,xticklabels=['Previous','Zero-shot','Real-anchored'],ylim=(0,1.14),ylabel='Real label accuracy (n=40)',title='Same 40 real texts');axes[0].legend()
    matrix=np.array([[classes[(classes['mode']==mode)&(classes.model==model)&(classes.source=='real')&(classes.label==label)].iloc[0].label_match_rate for label in LABELS] for mode in ['zero_shot','real_anchored'] for model in ['qwen','llama']])
    axes[1].imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    axes[1].set_xticks(range(4),LABELS);axes[1].set_yticks(range(4),['Zero / Qwen','Zero / Llama','Anchor / Qwen','Anchor / Llama']);axes[1].set_title('Real per-class accuracy (n=10 each)')
    for i in range(4):
        for j in range(4): axes[1].text(j,i,f'{matrix[i,j]:.0%}',ha='center',va='center',color='white' if matrix[i,j]>.6 else 'black')
    for mode,color in [('zero_shot','#3575ad'),('real_anchored','#e69b40')]:
        g=pairs[pairs['mode']==mode]
        axes[2].hist(g.llama_label_score,bins=np.linspace(0,1,21),alpha=.45,color=color,label=mode)
    axes[2].set(xlabel='Llama candidate probability of presented label',ylabel='Samples (n=200 per condition)',title='Logprob score distribution');axes[2].legend(fontsize=8)
    fig.savefig(OUT/'verifier_repair_summary.png',dpi=180)
    plt.close(fig)
    print('Saved report, Excel and summary figure',flush=True)


if __name__=='__main__': main()
