#!/usr/bin/env python3
"""Build a narrative Chinese research report as a DOCX without touching project code."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/root/workspace")
SYN = ROOT / "SyntheticData"
ASSET = ROOT / "report_docx_assets"
OUT_DOCX = ROOT / "synthetic_data_research_report.docx"
OUT_MD = ROOT / "synthetic_data_research_report.md"

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 160,
    "savefig.dpi": 220,
})


def savefig(name: str) -> str:
    path = ASSET / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return f"report_docx_assets/{name}"


def copy_asset(src: Path, name: str) -> str:
    dst = ASSET / name
    shutil.copy2(src, dst)
    return f"report_docx_assets/{name}"


def heatmap(matrix: pd.DataFrame, title: str, filename: str, cmap: str = "YlGnBu", fmt: str = ".3f") -> str:
    fig, ax = plt.subplots(figsize=(6.1, 4.45))
    arr = matrix.to_numpy(dtype=float)
    im = ax.imshow(arr, cmap=cmap, vmin=np.nanmin(arr), vmax=np.nanmax(arr))
    ax.set_xticks(range(len(matrix.columns)), matrix.columns)
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    ax.set_xlabel("Test source")
    ax.set_ylabel("Train source")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, format(arr[i, j], fmt), ha="center", va="center", fontsize=9,
                    color="white" if arr[i, j] > np.nanmean(arr) + 0.35 * np.nanstd(arr) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return savefig(filename)


def build_assets() -> dict[str, str]:
    ASSET.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}

    # Reuse the project's existing visualizations where they already match the HTML.
    existing = {
        "mmd": SYN / "MMD/outputs/distribution_analysis/figures/mmd_heatmap_overall.png",
        "mmd_gain": SYN / "MMD/outputs/distribution_analysis/figures/mmd_vs_dimp_gain.png",
        "curation": SYN / "SmaExp/outputs/visualizations/grouped_comparison.png",
        "dimp": SYN / "dimp/outputs/protocol_dimp_llama_qwen/visualizations/accuracy_comparison.png",
    }
    for key, src in existing.items():
        if src.exists():
            out[key] = copy_asset(src, f"{key}.png")

    # DIMP baseline from the values shown in the HTML.
    labels = ["Qwen CE", "Qwen Q-only", "Qwen DIMP", "Llama CE", "Llama Q-only", "Llama DIMP"]
    vals = [0.7083, 0.7017, 0.7042, 0.6850, 0.7008, 0.6858]
    colors = ["#4c87ad", "#4a9878", "#d2a357", "#4c87ad", "#4a9878", "#d2a357"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0.60, 0.76)
    ax.set_ylabel("Accuracy")
    ax.set_title("DIMP baseline on Legacy400")
    ax.tick_params(axis="x", rotation=25)
    for x, y in enumerate(vals):
        ax.text(x, y + 0.003, f"{y:.4f}", ha="center", fontsize=8)
    out["dimp_chart"] = savefig("dimp_baseline.png")

    # Selector comparison from the formal Dev2000 summary.
    methods = ["Random", "Original DSIR", "LN-DSIR", "RA-CaR", "TopK"]
    acc = [0.7259, 0.7262, 0.7392, 0.7378, 0.7526]
    f1 = [0.7017, 0.7045, 0.7228, 0.7227, 0.7398]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - 0.18, acc, 0.36, label="Accuracy", color="#4c87ad")
    ax.bar(x + 0.18, f1, 0.36, label="Macro-F1", color="#4a9878")
    ax.set_xticks(x, methods, rotation=18)
    ax.set_ylim(0.65, 0.79)
    ax.set_ylabel("Score")
    ax.set_title("Selector comparison on Dev2000")
    ax.legend(frameon=False, ncol=2)
    out["selector"] = savefig("selector_comparison.png")

    # ESS diagnostic.
    fig, ax = plt.subplots(figsize=(5.3, 3.5))
    ax.bar(["Original DSIR", "LN-DSIR"], [1.00, 45.97], color=["#c47777", "#4a9878"])
    ax.set_ylabel("Effective sample size")
    ax.set_title("Length normalization reduces weight concentration")
    ax.set_ylim(0, 52)
    ax.text(0, 2.5, "1.00", ha="center", fontsize=9)
    ax.text(1, 47.2, "45.97", ha="center", fontsize=9)
    out["ess"] = savefig("ess_comparison.png")

    # Allocation response from the HTML table.
    quotas = ["20Q/60L", "40Q/40L", "60Q/20L"]
    alloc = {
        "World": [0.7893, 0.8026, 0.7983],
        "Sports": [0.8955, 0.9001, 0.9008],
        "Business": [0.6994, 0.6975, 0.6976],
        "Sci/Tech": [0.5696, 0.5693, 0.5413],
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for name, ys in alloc.items():
        ax.plot(quotas, ys, marker="o", linewidth=2, label=name)
    ax.set_ylim(0.50, 0.94)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Coarse source allocation response")
    ax.legend(frameon=False, ncol=2)
    out["allocation"] = savefig("allocation_response.png")

    # Candidate pool scaling from the HTML summary.
    pools = [80, 160, 250, 500]
    pool_acc = [0.7482, 0.7503, 0.7187, 0.7378]
    pool_f1 = [0.7356, 0.7382, 0.6998, 0.7275]
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.plot(pools, pool_acc, marker="o", linewidth=2, label="Accuracy", color="#4c87ad")
    ax.plot(pools, pool_f1, marker="o", linewidth=2, label="Macro-F1", color="#d2a357")
    ax.set_xticks(pools)
    ax.set_xlabel("Candidates per source / class")
    ax.set_ylabel("Dev score")
    ax.set_title("Candidate-pool scaling with fixed final budget")
    ax.legend(frameon=False)
    out["pool"] = savefig("pool_scaling.png")

    # New 4x4 source compatibility matrix: 3-seed mean and std.
    csv = SYN / "SmaExp/outputs/visualizations/comparison_summary_500_matrix.csv"
    raw = pd.read_csv(csv)
    order = ["real", "qwen", "llama", "mix"]
    acc_m = raw.groupby(["Train_Source", "Test_Source"]).Accuracy.mean().unstack().reindex(index=order, columns=order)
    f1_m = raw.groupby(["Train_Source", "Test_Source"]).Macro_F1.mean().unstack().reindex(index=order, columns=order)
    loss_m = raw.groupby(["Train_Source", "Test_Source"]).Loss.mean().unstack().reindex(index=order, columns=order)
    out["source_acc"] = heatmap(acc_m, "4×4 source matrix: Accuracy mean (3 seeds)", "source_accuracy_heatmap.png", "YlGnBu")
    out["source_f1"] = heatmap(f1_m, "4×4 source matrix: Macro-F1 mean (3 seeds)", "source_f1_heatmap.png", "YlGnBu")
    out["source_loss"] = heatmap(loss_m, "4×4 source matrix: Loss mean (3 seeds)", "source_loss_heatmap.png", "YlOrRd")
    return out


def fmt_mean_std(df: pd.DataFrame, metric: str, train: str, test: str) -> str:
    row = df[(df.Train_Source == train) & (df.Test_Source == test)][metric]
    return f"{row.mean():.4f} ± {row.std(ddof=1):.4f}"


def source_matrix_table(raw: pd.DataFrame, metric: str) -> str:
    order = ["real", "qwen", "llama", "mix"]
    names = {"real": "Real", "qwen": "Qwen", "llama": "Llama", "mix": "Mix"}
    lines = ["| Train \\ Test | Real | Qwen | Llama | Mix |", "|---|---:|---:|---:|---:|"]
    for train in order:
        vals = [fmt_mean_std(raw, metric, train, test) for test in order]
        lines.append("| **%s** | %s |" % (names[train], " | ".join(vals)))
    return "\n".join(lines)


def build_markdown(assets: dict[str, str]) -> str:
    raw = pd.read_csv(SYN / "SmaExp/outputs/visualizations/comparison_summary_500_matrix.csv")
    return f"""---
title: 多来源合成文本的数据选择、分配与 synthetic–real mismatch
author: SyntheticData 项目研究记录
date: 2026-09-29
toc: true
---

# 摘要

这份报告把目前项目中的实验放在同一条研究主线上：我们一开始问的是“能不能通过给 synthetic data 加权，让模型学得更好”，随后逐渐发现，仅仅改变 loss 中的权重并不能稳定解决问题。于是研究问题自然转向两个更直接的问题：**在固定训练预算下，哪些 synthetic samples 值得进入训练集？不同 generator 和类别应该各自贡献多少样本？**

当前最清晰的证据是：在已经完成的 Dev2000 选择实验中，Real-Anchored TopK 的表现高于 Random、Original DSIR、LN-DSIR 和 RA-CaR；但 source allocation 还没有显示稳定、单调的 quota response。新完成的 500-per-class source matrix 进一步说明，synthetic→synthetic 的迁移非常容易得到高分，而训练在真实测试集上的结果明显更低，这正是 synthetic–real mismatch 需要被单独诊断的原因。

报告中的“高分”都只表示在当前数据划分、训练预算和随机种子下观察到的结果，不把任何一个方法写成已经证明最优。Legacy400、Dev2000、synthetic holdout 和 Final Holdout 也分别说明，避免把不同实验口径混在一起。

# 1. 我们究竟想解决什么问题？

任务是 AG News 四分类，downstream classifier 使用 `bert-base-uncased`。最终训练预算固定为 320 条样本，即每个类别 80 条；因此研究重点不是“再多训练一些数据”，而是在同样预算下改变数据的**来源、质量和组成**。

研究路线可以概括为：

1. **Training weighting**：DIMP、Quality-only 和 CE 比较，测试训练时加权是否有效。
2. **Distribution diagnostics**：用 MMD、word overlap 等指标描述 synthetic 与 real 的差异。
3. **Sample curation**：在训练前筛选样本，重点验证 Real-Anchored TopK。
4. **Mechanism stress tests**：检查 DSIR 的权重塌缩、length bias、diversity 和 semantic mode mismatch。
5. **Source allocation**：在固定总预算下调整 Qwen/Llama 配额。
6. **Candidate-pool scaling**：扩大候选池，观察“候选更多”是否真正带来更好的可选样本。

这条路线并不割裂：前一阶段的负结果推动了下一阶段。DIMP 的收益有限，说明要把注意力从“样本进来以后怎么加权”移到“哪些样本一开始就应该进来”；MMD 能告诉我们来源之间有多远，但不能直接告诉我们某个样本对 downstream utility 有多大；因此最终需要把局部 relevance、source composition 和固定预算放到同一框架中。

# 2. 统一实验设置

早期 baseline 使用 Real200、Qwen320 和 Llama320，并在 Legacy400 / RealTest400 上评估。selector 和 allocation 的正式开发口径是 Dev2000；Final Holdout 仍然保留到方法冻结之后使用。新增的 source matrix 使用扩展后的候选池：每个 synthetic source、每个类别有 500 条样本，固定拆成 400 条 train 和 100 条 synthetic holdout；real protocol 继续使用已有 Real200 / RealTest400。

| 资源 | 训练数据 | 测试数据 | 用途 |
|---|---|---|---|
| Real | 已有 Real200 | RealTest400 | real baseline 与真实泛化 |
| Qwen | 400 / class | 100 / class holdout | source compatibility |
| Llama | 400 / class | 100 / class holdout | source compatibility |
| Mix | Qwen200 + Llama200 / class | Qwen100 + Llama100 / class | 混合来源诊断 |

新增矩阵共运行 16 个 train-source × test-source 组合，每个组合使用模型 seed 42、43、44，合计 48 个 BERT runs。这个实验的目的，是回答“训练来源与测试来源之间是否存在明显兼容性差异”，而不是直接比较 selector 或 allocation。

# 3. 从训练加权开始：DIMP baseline

DIMP 用一个由少量真实标注训练的 Quality Checker 得到质量概率 $q_i$，再结合 BERT 当前的预测概率 $p_i$，近似使用 $q_i/p_i$ 形成 weighted cross-entropy。对照方法是所有样本等权的 CE，以及只使用质量分数的 Quality-only。

![DIMP baseline]({assets.get("dimp_chart", assets.get("dimp", ""))})

在 Legacy400 上，Qwen 的 CE、Quality-only、DIMP Accuracy 分别为 0.7083、0.7017、0.7042；Llama 分别为 0.6850、0.7008、0.6858。变化幅度有限，而且 Qwen 与 Llama 的方向并不一致。这意味着当前证据不支持“DIMP 对所有 generator 都稳定有效”。更谨慎的解释是：样本质量评分可能有帮助，但仅在 loss 中重新分配权重，未必能修复 source-specific 的数据分布和语义覆盖问题。

# 4. 为什么需要 distribution diagnostics？

MMD 结果显示，Real–Qwen 的 overall MMD 为 0.119653，Real–Llama 为 0.095390，而 Qwen–Llama 只有 0.013731。换句话说，两个 synthetic source 彼此很接近，但它们与 real 的距离都更大；Llama 在整体 MMD 上更接近 real，却不自动意味着 downstream accuracy 更高。

![MMD overall heatmap]({assets.get("mmd", "")})

这组结果帮助我们理解 mismatch，但不能把 MMD 当成训练效用的替代指标。早期 repeated-sampling 的 std 记录曾出现实现问题，因此 MMD 数值在本报告中主要用于诊断和排序解释，不作为严格统计不确定性结论。类似地，content-word overlap（Real–Qwen 0.1874、Real–Llama 0.2120、Qwen–Llama 0.3156）可以提示 lexical structure 的差异，却不能单独决定应该选哪些样本。

# 5. 从 weighting 转向 sample curation

当总预算固定时，训练前筛选样本比训练中给所有样本分权更直接。最初的 2×2 pilot 让 Uniform + TopK 得到 0.7100 Accuracy / 0.6923 Macro-F1，高于 Uniform + Random 的 0.7000 / 0.6698；MMD Soft + TopK 反而降到 0.6875 / 0.6617。这推动研究将 selector 的核心信号从全局距离转向与真实数据的局部相关性。

![Curation comparison]({assets.get("curation", "")})

Real-Anchored TopK 的做法是：对每个 synthetic sample，在同一类别的 RealReference 中寻找 Top-5 nearest neighbors，用平均 cosine similarity 作为 relevance，再在 source × class 内选出固定数量。Dev2000 的 25 个 Random runs 与 5 个 TopK runs 比较中，TopK 平均 Accuracy 为 0.7526 ± 0.0316，Macro-F1 为 0.7398 ± 0.0405；Random 为 0.7259 ± 0.0329 和 0.7017 ± 0.0446。对应提升约为 +2.67 pp Accuracy 和 +3.81 pp Macro-F1。

![Selector comparison]({assets["selector"]})

按类别看，Sci/Tech 的 Macro-F1 从 0.4563 提升到 0.5576，增加约 10.13 pp；World 和 Business 约提升 2.2 pp，Sports 变化较小。这个结果更像是“真实数据锚定的局部相关性在当前设置下有用”，而不是“TopK 已经解决了所有 mismatch”。

# 6. Stress tests：解释 TopK，也保留负结果

后续方法不是为了不断刷分，而是为了检查 TopK 可能依赖的机制。

**DSIR / LN-DSIR。** Original DSIR 在短文本场景下出现严重的 weight concentration：ESS 约为 1.00，ESS ratio 约 0.0125，length 与 log weight 的 Spearman 相关中位数为 0.933。Length-normalized DSIR 把 ESS 提升到约 45.97，ESS ratio 约 0.574，相关中位数降到 0.452，说明诊断问题被修复；但 downstream 表现仍没有稳定超过 TopK。

![ESS comparison]({assets["ess"]})

**CaR。** CaR 可以提高 cluster coverage，部分设置覆盖 6/6 clusters，quality retention 接近 1，且与 TopK 的 Jaccard overlap 约 0.90。这说明 CaR 的确改变了 diversity 行为，但 RA-CaR 的 Dev2000 结果仍略低于 TopK，因此当前结论是“diversity 增加不自动等价于 utility 增加”。

**ARA-TopK。** ARA 根据 semantic mode mismatch 对 TopK 做 adaptive correction，部分 JSD / cluster diagnostics 得到改变，但 ARA minus TopK F1 = -0.0103。当前把它记录为 distribution repair、no downstream advantage，而不是继续包装成成功方法。

**Length confound。** Llama 的部分类别存在 length–proximity 相关。Shortest、Length-Matched Random 与 TopK 的对照表明 length correlation 确实存在，但单独的 length effect 不足以解释 TopK 的全部行为。

# 7. Source allocation：从“选哪些”到“每个来源分多少”

每个类别的最终预算为 80，baseline 是 40 Qwen + 40 Llama；早期 allocation test 比较 20/60、40/40、60/20。四个类别的 Macro-F1 反应方向并不一致：Sci/Tech 在 Qwen60/Llama20 时下降明显，而 Business 几乎不变，World 和 Sports 也没有呈现稳定单调趋势。

![Allocation response]({assets["allocation"]})

因此目前不能说 allocation 无效，只能说在小候选池和当前 selection depth 下，没有观察到可重复的 quota response。还要注意一个重要混杂：小池中每个 source/class 只有 80 个候选，select20、40、60 分别相当于取 25%、50%、75%；改变 quota 的同时，也改变了 source 内部筛选的深度，所以 allocation 与 selector 并非完全独立。

# 8. Candidate-pool scaling

候选池大小与最终训练预算是两个不同变量。池子从 80、160、250 扩大到 500 candidates/source/class，但最终仍然只选 40 Qwen + 40 Llama / class，总训练量保持 320。

![Candidate pool scaling]({assets["pool"]})

Dev2000 的平均结果为：Pool80 Accuracy / Macro-F1 = 0.7482 / 0.7356，Pool160 = 0.7503 / 0.7382，Pool250 = 0.7187 / 0.6998，Pool500 = 0.7378 / 0.7275。Pool160 相对 Pool80 的提升很小，Pool250 和 Pool500 没有超过 Pool80。当前更稳妥的表述是：候选池扩大可能在某个范围内改善可选空间，但并不支持“候选越多越好”；还需要结合 saturation 和 seed variance 分析。

# 9. 新增实验：4×4 source compatibility matrix

## 9.1 实验目的与口径

新增实验把 train source 和 test source 都设为 Real、Qwen、Llama、Mix，形成 4×4 矩阵。synthetic 的 500 条 / class 被拆成 400 train + 100 holdout，训练和测试 sample ID 不重叠；每个 cell 跑 3 个 model seeds。这里的 `mix` 是按类别混合 Qwen 与 Llama，不是额外生成的第三个 generator。

最重要的解释边界是：synthetic→synthetic 的高分只说明模型在相近的 synthetic domain 中容易迁移，不能代替 real generalization。真正能体现 synthetic–real mismatch 的，是同一个训练来源在 RealTest400 上的表现。

## 9.2 Accuracy、Macro-F1 和 Loss

以下表格为 3 个 model seeds 的均值 ± 样本标准差。

### Accuracy

{source_matrix_table(raw, "Accuracy")}

![Source matrix accuracy]({assets["source_acc"]})

### Macro-F1

{source_matrix_table(raw, "Macro_F1")}

![Source matrix macro F1]({assets["source_f1"]})

### Loss

{source_matrix_table(raw, "Loss")}

![Source matrix loss]({assets["source_loss"]})

## 9.3 结果怎么读？

第一，Real 训练在 Real 测试上的 Accuracy 为 0.6983 ± 0.0496，Qwen、Llama、Mix 训练在 Real 测试上分别为 0.6800 ± 0.0075、0.7033 ± 0.0063、0.7017 ± 0.0076。synthetic 训练并没有在真实测试集上显示出压倒性优势，且真实训练本身的 seed 波动更大，这提醒我们不能只看 synthetic holdout。

第二，Qwen、Llama、Mix 训练在 Qwen/Llama/Mix 测试上的 Accuracy 大多接近 0.97–1.00，Macro-F1 也很高；Real 训练在这些 synthetic 测试上的结果只有约 0.50–0.52 Accuracy。这种明显的 domain asymmetry 正是 mismatch 的直接可视化证据。

第三，Llama 训练在 Real 测试上的平均 Accuracy 为 0.7033，略高于 Qwen 的 0.6800；但这不能被解释成 Llama 普遍更好，因为它只描述了当前固定 split、固定训练配置和 3 个 model seeds。矩阵适合用作 source compatibility diagnostic，不能直接推出最终 allocation quota，也不能替代 selector 的固定预算实验。

# 10. 当前结论与下一步

当前最强证据链可以压缩为：

- DIMP 的收益有限且依赖 generator，促使研究从 training weighting 转向 data curation。
- MMD 发现 synthetic–real gap，但 distance 本身不等于 utility。
- 在当前小池和 Dev2000 口径下，Real-Anchored TopK 是最稳定的 selector signal。
- Original DSIR 的 ESS collapse、LN-DSIR 的“诊断修复但未超越”、CaR 的“diversity 增加但 utility 未超越”、ARA 的“distribution repair 但 no advantage”，共同收缩了设计空间。
- 粗粒度 source allocation 尚无稳定 quota signal；candidate pool scaling 也没有表现出单调增益。
- 新的 4×4 matrix 说明必须把 synthetic→synthetic 高分与 real generalization 分开看，source compatibility 本身值得作为后续 allocation 建模的输入。

下一阶段更自然的方向，是在固定总预算下估计 source/class 的 marginal utility：通过小扰动实验改变某一来源的样本量，拟合 validation loss 或 real-oriented utility response curve，再求解受总预算约束的 source quota。方法冻结后再使用 Final Holdout，避免继续用 holdout 做开发。

# 附录：可复现性和实验口径

| 口径 | 用途 | 状态 |
|---|---|---|
| Legacy400 / RealTest400 | 早期 baseline、历史一致性 | 不与 Dev2000 混排 |
| Dev2000 | selector / allocation development | 当前主开发集 |
| Synthetic holdout | 500-per-class source matrix | 只用于来源兼容性诊断 |
| Final Holdout | 最终一次验证 | 尚未用于方法开发 |

主要结果来自项目已有的 CSV、Excel、JSON 和可视化输出；新增 Word 只读取这些结果并生成文档资产，不修改训练脚本、实验脚本或原始 HTML。矩阵实验的原始汇总见 `SyntheticData/SmaExp/outputs/visualizations/comparison_summary_500_matrix.xlsx`，其中保留了逐 seed 的 `Cell_Results`、均值方差、三个指标矩阵和设计说明。

报告中的限制也需要保留：部分早期 QC reload 与 MMD uncertainty 的工程检查尚未完全闭环；MMD 的 repeated-sampling std=0 不能当作可靠的不确定性估计；allocation 受 selection depth confound 影响；新矩阵只用了 3 个 model seeds，适合做诊断，不宜过度外推。
"""


def main() -> None:
    assets = build_assets()
    OUT_MD.write_text(build_markdown(assets), encoding="utf-8")
    subprocess.run([
        "pandoc", str(OUT_MD), "-o", str(OUT_DOCX),
        "--from=gfm", "--to=docx", "--toc",
    ], check=True)
    print(f"Wrote {OUT_DOCX} ({OUT_DOCX.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
