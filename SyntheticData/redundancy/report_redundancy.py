"""Export the completed pilot as Chinese Markdown, Excel and a static figure."""
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "outputs"


def fmt(value, percent=False):
    if value is None:
        return "未定义（分母为 0）"
    return f"{100*value:.1f}%" if percent else f"{value:.4f}"


def main():
    metrics = json.loads((OUT / "metrics.json").read_text())
    manifest = json.loads((OUT / "manifest.json").read_text())
    pairs = pd.read_csv(OUT / "paired_results.csv")
    main_rows = pairs[pairs.subset == "main"]
    agreement = []
    fields = [("label_agreement", "Label prediction agreement",True), ("pass_fail_agreement", "PASS/FAIL agreement",True), ("label_kappa", "Label Cohen's κ",False), ("quality_kappa", "PASS/FAIL Cohen's κ",False), ("reject_jaccard", "Reject-set Jaccard",False), ("bottom20_jaccard", "Matched-budget bottom20% Jaccard",False), ("qwen_pass_rate", "Qwen PASS rate",True), ("llama_pass_rate", "Llama PASS rate",True)]
    for key, name, pct in fields:
        agreement.append(dict(metric=name, main200=metrics["main200"][key],synthetic160=metrics["synthetic160"][key]))
    pd.DataFrame(agreement).to_csv(OUT / "agreement_metrics.csv",index=False,encoding="utf-8-sig")
    text = ["# Qwen / Llama Verifier Redundancy Pilot", "", "实验日期：2026-10-06；仅使用已有文本进行本地推理，未训练 BERT，未生成新数据。", "", "## 数据与协议", "", f"从已有 4,000 条 synthetic pool（2 source × 4 class × 500）固定随机抽取 20/source/class，共 160 条；从既有 RealReference 的冻结训练参考集合抽取 10/class，共 40 条。主实验为 200 条。另复制这 40 条 RealReference 文本，把标签改为其他类别，得到 40 条 challenge，合计 240 条。seed={manifest['seed']}；主样本中 {manifest['selected_legacy_count']} 条来自 legacy pool，legacy 的精确生成 checkpoint 未记录。", "", "Verifier 为本地 Qwen3-4B-Instruct-2507 与 Llama-3.2-3B-Instruct。两边逐条收到完全相同的 user prompt、相同样本顺序，每条为独立上下文；只用模型各自的 native chat wrapper。输入不含 source、item ID、challenge 标志或 true_label。解码为 greedy / do_sample=False，max_new_tokens=96，FP16，batch_size=8。", "", "输出严格为 predicted_label、quality、confidence、reason 四行。confidence 定义为训练样本的 acceptability score，错标签必须低分；这类自报分数未做概率校准。challenge 正确的拒绝输出是 quality=FAIL、reason=LABEL_MISMATCH（不是额外新增 FAIL_LABEL 枚举）。", "", "Qwen 的首轮 prompt 试跑曾把 confidence 当成判断确信度，故完整保留在 attempt1_confidence_semantics_failed，但不纳入最终比较。随后加入评分规则和两个示例；最终两边都使用同一份 v2 prompt，同一份 240 条样本。prompt 修改由 Qwen 首轮行为触发，因此这是开发性 pilot，正式确认应在新样本上冻结 v2 协议再测。", "", "## Verifier agreement", "", "| 指标 | 主实验 n=200 | Synthetic n=160 |", "|---|---:|---:|"]
    for key,name,pct in fields:
        text.append(f"| {name} | {fmt(metrics['main200'][key],pct)} | {fmt(metrics['synthetic160'][key],pct)} |")
    text += ["", "Cohen's κ 分别按 label 与 PASS/FAIL 计算，challenge 不混入主样本指标。Reject Jaccard 的空并集记为未定义，不赋值为 0 或 1。", "", "输出均保留原始模型判断，不把逻辑矛盾自动修正成 FAIL。对 240 条输出的一致性检查：", ""]
    for model,x in metrics['semantic_consistency'].items():
        text.append(f"- {model}：PASS 但 predicted_label 与 presented_label 不同 {x['pass_with_label_mismatch']} 条；PASS 却不是 OK {x['pass_with_non_OK_reason']} 条；FAIL 却是 OK {x['fail_with_OK_reason']} 条；FAIL 却给 ≥0.5 的 acceptability score {x['fail_with_high_acceptability_score']} 条；LABEL_MISMATCH score 超出 prompt 指定 0–0.1 区间 {x['label_mismatch_score_above_prompt_band']} 条。")
        text.append(f"- {model}：40 条未改标签 RealReference 的 predicted-label accuracy={fmt(metrics['real_reference_label_accuracy'][model],True)}。")
    text += ["", "这些不一致会限制 verifier 的可靠性。低 agreement 可能来自较弱的标签判断或指令遵循，不能直接当作有用 diversity。", "", "## 固定预算 overlap 与分数并列", ""]
    for scope in ["main200","synthetic160"]:
        m=metrics[scope]
        d=m['bottom20_diagnostics']
        text.append(f"- {scope}：各拒绝 {d['qwen']['reject_n']} 条，交集 {m['bottom20_intersection']}，并集 {m['bottom20_union']}，Jaccard={fmt(m['bottom20_jaccard'])}。")
        for model in ['qwen','llama']:
            x=d[model]
            text.append(f"- {scope}/{model}：截止分数 {x['cutoff']:.3f}，该分数有 {x['tied_at_cutoff']} 条，是否跨越预算边界：{'是' if x['ties_straddle_cutoff'] else '否'}。")
        t=metrics['matched_budget_tie_sensitivity'][scope]
        text.append(f"- {scope}：仅在同分条目间对两模型独立随机打破并列 1,000 次，Jaccard 均值 {t['mean']:.4f}，2.5%–97.5% 范围 [{t['p025']:.4f}, {t['p975']:.4f}]。这是并列处理敏感性范围，不是样本统计置信区间。")
    text += ["", "主结果按 score 升序排序，同分时使用共同的、source-blind 的 SHA256(seed:item_id)。共同打破并列可能增加交集，故同时报告独立处理并列的敏感性。原始 confidence 与完整 bottom20 集合见 matched_budget_membership.csv。", "", "## Self-preference", "", "| Verifier | Source | 接受数 / 样本数 | PASS rate | Wilson 95% CI | 平均 acceptability score |", "|---|---|---:|---:|---|---:|"]
    for x in metrics['self_bias']:
        text.append(f"| {x['verifier']} | {x['source']} | {x['accepts']}/{x['n']} | {fmt(x['accept_rate'],True)} | [{fmt(x['wilson95_low'],True)}, {fmt(x['wilson95_high'],True)}] | {x['mean_confidence']:.3f} |")
    text += ["", "差值定义为 P(accept own source) − P(accept other source)，每 source 的四类数量相同；按 source/class 分层 bootstrap 5,000 次：", ""]
    for model,x in metrics['self_preference_differences'].items():
        lo,hi=x['class_stratified_bootstrap95']
        text.append(f"- {model}：{x['own_minus_other_accept_rate']*100:+.1f} 个百分点，95% bootstrap 区间 [{lo*100:+.1f}, {hi*100:+.1f}] 个百分点。")
    text += ["", "source 盲评能排除显式 source 提示，但 source 的真实质量、写作风格与类别难度仍可能影响接受率。own-source 差值只能作为 self-preference 信号，不能单凭此证明因果 bias。逐类分解见 self_bias_by_class.csv。", "", "## Challenge set 与共享错误", "", "标签 ground truth 取自既有 AG News RealReference；这是可复现的 label-corruption challenge，未重新人工核验所有真实标签。每类 10 条改为其他三类，错误目标每类分配为 3/3/4。challenge 专门检验错标签识别，不能据此估计所有低质量类型的泛化表现。", "", "| 指标 | 严格：未返回 FAIL + LABEL_MISMATCH | 宽松：未返回 FAIL |", "|---|---:|---:|"]
    cm=metrics['challenge40']
    a,b=cm['strict_label_rejection'],cm['failure_to_reject']
    for key,name,pct in [("qwen_accuracy","Qwen challenge accuracy",True),("llama_accuracy","Llama challenge accuracy",True),("qwen_error_rate","Qwen error rate",True),("llama_error_rate","Llama error rate",True),("shared_error_rate","共同出错样本 / 40",True),("shared_error_jaccard","Shared-error Jaccard",False)]:
        text.append(f"| {name} | {fmt(a[key],pct)} | {fmt(b[key],pct)} |")
    text += ["", f"严格错误集合：Qwen={a['qwen_errors']}，Llama={a['llama_errors']}，交集={a['shared_errors']}，并集={a['error_union']}。用户提出的共享错误重叠指标为交集/并集；共同出错样本/40 是另一个指标，不能混用。", ""]
    for model in ['qwen','llama']:
        lo,hi=a[f'{model}_accuracy_wilson95']
        text.append(f"- {model} 严格 challenge accuracy 的 Wilson 95% CI：[{fmt(lo,True)}, {fmt(hi,True)}]；true-label prediction accuracy={fmt(cm['true_label_prediction_accuracy'][model],True)}。")
    text += ["", "## Pilot 解读", ""]
    m=metrics['main200']
    if m['quality_kappa'] is not None and m['quality_kappa'] < .4 and abs(a['qwen_accuracy']-a['llama_accuracy']) > .2:
        text.append(f"这轮最接近用户决策表中的『κ 很低，但一个模型 challenge accuracy 明显差』：PASS/FAIL κ={m['quality_kappa']:.4f}，Qwen challenge accuracy={fmt(a['qwen_accuracy'],True)}，Llama={fmt(a['llama_accuracy'],True)}。优先修正或替换当前较差的 verifier 配置；不能将结果直接解释为两模型的有效互补，也不能泛化为整个模型 family 的优劣。")
    if any(x['tied_at_cutoff'] > m['n']/2 for x in m['bottom20_diagnostics'].values()):
        text.append("至少一个 verifier 在主样本的 bottom20 截止分数上有超过一半样本并列，连续排序信号明显不足。当前 matched-budget Jaccard 应视为退化 score 下的描述性结果，不能独立支持 TopB 互补结论。")
    if a['error_union'] == 0:
        text.append("双方在这 40 条错标签 challenge 上都未出错，shared-error Jaccard 是 0/0、未定义。因此本实验无法判断两模型的错误是否高度重叠，更不能把它解释成错误冗余为 0。challenge 偏简单，后续需更难且经人工确认的类别边界错误。")
    elif a['shared_error_jaccard'] > .8:
        text.append("这组错标签 challenge 的共享错误重叠超过 0.8，但应同时查看错误集合大小：少数错误不足以稳定确认模型 family 的冗余。")
    else:
        text.append("这组错标签 challenge 的共享错误重叠未超过 0.8；这一结论仅适用于本组 challenge，尤其应结合错误集合大小评估稳定性。")
    if m['quality_kappa'] is not None and m['quality_kappa'] > .8 and m['reject_jaccard'] is not None and m['reject_jaccard'] > .75:
        text.append("主样本满足 κ>0.8 且 Reject Jaccard>0.75，判断有较高冗余；是否换 family 仍需共享错误的有效证据。")
    else:
        text.append("主样本未同时满足 κ>0.8 与 Reject Jaccard>0.75，不能按该规则直接判为高度冗余。应优先结合固定预算 overlap、challenge 表现与 source 差值决定下一步。")
    if all(x['own_minus_other_accept_rate'] > 0 for x in metrics['self_preference_differences'].values()):
        text.append("两模型都呈正向 own-source 接受率差值，但是否为稳定 self-preference 要结合差值区间和 source 质量混杂；尚不能断言 cross-verifier > self-verifier。")
    else:
        text.append("没有同时出现 Qwen 更接受 Qwen、Llama 更接受 Llama 的双向 own-source 现象；本轮不能支持双向 self-preference，也不能得出 cross-verifier > self-verifier。")
    if any(x['pass_with_label_mismatch'] or x['fail_with_high_acceptability_score'] for x in metrics['semantic_consistency'].values()):
        text.append("模型有输出字段之间的逻辑矛盾，建议先验证 prompt/判定规则的可靠性，再扩大实验。若后续引入 predicted_label 与 presented_label 不同则强制 FAIL 的程序规则，必须单列为新协议，不能覆盖这轮原始判断。")
    text += ["", "用户给定 κ / Jaccard 阈值均作为 pilot heuristic。n=200、每 source/class=20、challenge=40 的单次抽样不能作为统计定律或最终模型选择依据。TEMPLATE_OR_REPETITIVE 仅评价单条文本的明显模板痕迹，不是跨 pool 重复率。", "", "## 输出与复现", "", "- samples.csv / samples.jsonl：样本、source、原始 ID、true/presented label、challenge parent ID。", "- blind_prompts.jsonl / prompt_template.txt：两模型完全相同的输入内容。", "- qwen_responses.jsonl / llama_responses.jsonl：未改写的模型原始输出与解析结果。", "- paired_results.csv / metrics.json：逐条配对判断与完整指标。", "- redundancy_results.xlsx：可直接查看的汇总、逐类分析、challenge 与逐条数据。", "- redundancy_summary.png：接受率、score 分布与 challenge 错误的静态图。", "- manifest.json / *_runtime.json：输入 hash、模型、解码与复现配置。", "", "```bash", "/opt/anaconda/bin/python SyntheticData/redundancy/run_redundancy.py prepare", "env LD_LIBRARY_PATH=/opt/anaconda/lib CUDA_VISIBLE_DEVICES=0 /opt/anaconda/bin/python SyntheticData/redundancy/run_redundancy.py verify --model qwen", "env LD_LIBRARY_PATH=/opt/anaconda/lib CUDA_VISIBLE_DEVICES=0 /opt/anaconda/bin/python SyntheticData/redundancy/run_redundancy.py verify --model llama", "/opt/anaconda/bin/python SyntheticData/redundancy/run_redundancy.py analyze", "/opt/anaconda/bin/python SyntheticData/redundancy/report_redundancy.py", "/opt/anaconda/bin/python -m unittest discover -s SyntheticData/redundancy -p 'test_*.py' -v", "```", "", "已存在的样本与 responses 会复用，不会重新抽样或重复推理。更改 prompt/协议必须使用新的输出目录，避免混入旧结果。"]
    (OUT / "redundancy_report.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    with pd.ExcelWriter(OUT / "redundancy_results.xlsx",engine="openpyxl") as writer:
        pd.DataFrame(agreement).to_excel(writer,sheet_name="Agreement",index=False)
        pd.DataFrame(metrics['self_bias']).to_excel(writer,sheet_name="SelfBias",index=False)
        pd.read_csv(OUT/'self_bias_by_class.csv').to_excel(writer,sheet_name="SelfBiasByClass",index=False)
        pd.DataFrame([dict(policy=k,**v) for k,v in cm.items() if k != 'true_label_prediction_accuracy']).to_excel(writer,sheet_name="ChallengeSummary",index=False)
        pairs[pairs.subset=='challenge'].to_excel(writer,sheet_name="ChallengeSamples",index=False)
        pd.read_csv(OUT/'matched_budget_membership.csv').to_excel(writer,sheet_name="MatchedBudget",index=False)
        pairs.to_excel(writer,sheet_name="PairedResults",index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes='A2'
            sheet.auto_filter.ref=sheet.dimensions
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width=min(60,max(14,len(str(col[0].value))+3))
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    rates=np.array([[next(x['accept_rate'] for x in metrics['self_bias'] if x['verifier']==model and x['source']==source) for source in ['qwen','llama','real']] for model in ['qwen','llama']])
    axes[0].imshow(rates,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    axes[0].set_xticks(range(3),['Qwen source','Llama source','RealReference'])
    axes[0].set_yticks(range(2),['Qwen verifier','Llama verifier'])
    axes[0].set_title('Acceptance rates (main n=200)')
    for i in range(2):
        for j in range(3):
            axes[0].text(j,i,f'{rates[i,j]:.1%}',ha='center',va='center',color='white' if rates[i,j]>.6 else 'black')
    coords=main_rows.groupby(['qwen_confidence','llama_confidence']).size().reset_index(name='n')
    axes[1].scatter(coords.qwen_confidence,coords.llama_confidence,s=coords.n*8,alpha=.6)
    axes[1].plot([0,1],[0,1],color='gray',linestyle='--',linewidth=1)
    axes[1].set(xlabel='Qwen acceptability score',ylabel='Llama acceptability score',xlim=(-.04,1.04),ylim=(-.04,1.04),title='Paired scores (point area = count)')
    axes[2].bar(['Qwen only','Llama only','Both'],[a['qwen_errors']-a['shared_errors'],a['llama_errors']-a['shared_errors'],a['shared_errors']],color=['#3e77b6','#e49c3e','#9e5eab'])
    axes[2].set(ylabel='Number of challenge errors',ylim=(0,max(5,a['qwen_errors'],a['llama_errors'])+1),title='Strict label-rejection errors (n=40)')
    fig.savefig(OUT/'redundancy_summary.png',dpi=180)
    plt.close(fig)
    print('Saved redundancy_report.md, redundancy_results.xlsx, redundancy_summary.png')


if __name__ == '__main__':
    main()
