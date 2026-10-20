# GitHub 上传范围

可以把 `SyntheticData/` 内的全部实验源代码公开，包括早期探索、验证脚本、分析脚本、测试代码和运行脚本。这些不同阶段的实现体现了研究过程，README 再说明推荐从哪里开始即可。保留历史阶段的原名有助于与已有结果对应。

| 内容 | 建议 | 当前情况 |
| --- | --- | --- |
| `SyntheticData/` 中的 `.py`、`.sh`、`.ps1`、README、配置 | 上传全部实验源代码 | 未按实验结果好坏删除代码 |
| `tools/`、根目录报告生成脚本 | 上传，用于重新生成展示材料 | `tools/build_research_presentation.py` 已包含 |
| `README.md`、依赖文件、`.gitignore` | 上传 | 已准备 |
| `成果/` 的总结、展示 Excel、PPT、PNG/SVG、图表数据 | 可以上传，构成项目展示材料 | 本次材料不含逐条新闻文本 |
| `成果/关键Excel/` | 可保留，供查看细表和核对 | 六份原始结果工作簿副本；其中 CaR 表的汇总问题见使用说明 |
| `dsir-main/` | 可随依赖保留，注明上游来源并保留原 LICENSE | DSIR 方法实际调用其实现 |
| `CaR-main/` | 本地参考副本暂时保持排除 | 该目录含上游代码与数据，未发现随目录提供的 LICENSE；本项目的 RA-CaR 实现位于 `SyntheticData/CaRTry/`，可以上传 |
| `cloud/model/`、权重、虚拟环境、缓存、`outputs/`、日志 | 留在本地 | `.gitignore` 已排除 |
| 根目录临时提示词、旧任务记录 | 按内容另行选择 | 不必和实验代码一起提交 |

`.gitignore` 排除的是生成目录与本地依赖，不会排除 `SyntheticData/` 中的实验脚本。若以后发现某个 `outputs/` 里也放了应公开的源代码，应先把源代码移到对应实验目录再添加。

从工作区根目录显式添加以下内容，能够把全部实验代码和本次展示材料放入待提交列表：

```bash
git add README.md requirements.txt requirements-presentation.txt .gitignore \
  SyntheticData dsir-main tools 成果 generate_research_report_docx.py
git status --short
git diff --cached --stat
```

这只是本地暂存操作；远端仓库与正式推送仍按你创建仓库后的流程进行。上面的清单不意味着改变 `dsir-main/` 的上游版权归属。
