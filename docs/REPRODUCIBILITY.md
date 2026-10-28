# 实际重训与论文结果的比较 / Measured retraining results

## 应当期待什么

运行论文原设置，是用公开数据、固定划分和原训练配方从随机初始化实际训练。
同一环境下可能精确重建原数值；不同系统或依赖版本下也可能出现小幅差异。
本仓库分别报告：程序是否完成、实际指标与论文的差值、显示数值是否完全相同。
**测试通过不表示所有平台的每一个小数完全相同。**

新随机起点试跑是可选项，不改变论文数据、模型配方或已经发表的原结果。
保存预测复算、固定权重重新推理、从头训练是不同操作，网页已分别列出。

## 2026-10-28 已测结果

代码版本：`dae371bb4e72453c470e27a711ed44fa3517c946`（v0.7.0）。
公开运行记录：[GitHub Actions 40669210977](https://github.com/diudiu-aa/polymer-dmpnn-thermal-conductivity/actions/runs/40669210977)。
Linux Python 3.10/3.11/3.12各通过30项程序测试，均实际完成15个原设置模型重训；
测试时不提供任何已训练.pt文件。运行产物可在上述记录下载。
原macOS环境也完成15个模型重训，34项主结果显示统计全部一致。

下面是五次配对运行的均值；完整均值±样本标准差见同目录三份CSV。

| 指标 | 原论文 | Linux Python 3.10 / 3.11 | Linux Python 3.12 |
|---|---:|---:|---:|
| 开链 MAE | 0.02313 | 0.02321 | 0.02321 |
| 周期 MAE | 0.02213 | 0.02216 | 0.02216 |
| 双任务 MAE | 0.02255 | 0.02269 | 0.02265 |
| 周期 R² | 0.5864 | 0.5861 | 0.5861 |
| 双任务尾部 MAE | 0.04902 | 0.04933 | 0.04915 |
| 双任务回归排序 Precision@100 | 73.4% | 72.8% | 73.4% |

MAE单位均为 W/(m·K)。三个重训环境中，主模型整体MAE相对原论文变化均不到1%；
周期模型整体MAE优于开链，双任务尾部MAE优于周期，以及双任务回归排序的Top-100
命中率高于周期的均值关系仍保持。这里只陈述本次观察，**不把1%设为人为通过阈值**，
不等同于统计等效性证明、新数据泛化验证或所有单次运行都保持优势。
Top-10等更小候选集合指标变化较大，不能笼统说全部端点同样稳定。

Linux三环境均有32/34项显示统计与论文不完全相同；这个核对结果为false，保留在
paper_consistency.json中，不会改成true。逐条新旧预测也不满足固定权重重推理的1e-5
精确容差；该容差不是从头训练模型应达到的性能接近标准。
训练日志和结果不会因为小数不一致而被删掉，也不会用论文旧数值替换。

## 下载与运行

下载整个仓库后执行 `python start_ui.py`，选择“按论文原设置运行”。
完成后下载本次 `paper_main_results.csv` 和包含模型、预测、曲线、配置与日志的ZIP。
也可先安装依赖，再执行：

```bash
python reproduce/train_from_scratch.py --paper-settings --output paper-main-results
```

退出码0表示本次34个显示统计精确一致；退出码2表示运行完成但对照存在差异，
仍会保留新结果。其他错误应查看日志，不把失败计算当作复现完成。
这套重训覆盖表5开链/周期/双任务及分类排序；其余主表另提供保存预测指标复算。

## English

The three Linux CI jobs each completed 15 from-scratch fits using the original paper
settings without published checkpoints. Aggregate MAEs differed by less than 1% from
the paper in these measured runs, while the principal mean comparisons in the table
were retained. This observation is not an equivalence test, a universal tolerance,
or evidence on new data. Exact display matching failed for 32 of 34 statistics and
is explicitly reported as false. The 30 passing software tests verify execution and
reporting, not exact numerical reproduction. Saved-prediction metric recalculation
and frozen-checkpoint inference are separate operations.
