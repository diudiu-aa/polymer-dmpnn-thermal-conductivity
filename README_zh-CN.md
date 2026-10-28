# 聚合物热导率模型与论文复现

本仓库包含预测工具、论文**表1–5的逐条结果复算包**，以及主要模型的固定输入、
划分、权重和训练入口。目标是约300 K、1 atm下未拉伸无定形体相的**模拟热导率**，
单位W/(m K)，不能直接视为实验测量。

## 下载后运行论文主结果

下载并解压整个仓库，安装Python 3.10–3.13（建议3.12），在目录中运行：

```bash
python start_ui.py
# macOS通常使用 python3 start_ui.py
```

网页默认 **“按论文原设置运行：生成主结果表”**，点击 **“运行论文主结果”** 即可。
程序固定原数据划分、训练随机设置和参数，实际从头训练15个核心模型，
根据本次新预测生成 **paper_main_results.csv**，核对表5核心模型对应的34个显示数值。
CSV、新权重、逐条预测、训练曲线和日志均可下载。发现差异时如实报告，不用旧表覆盖。

这里是让读者运行并得到论文对应的主结果，不要求重新做数据收集、MD模拟或历史全部探索。
范围为开链/周期/双任务及其分类排序，未覆盖表5其他基线或整篇历史训练。
其他表的保存预测复算仍在“其他核对”中；“新随机起点”只是可选稳定性试跑，不是默认复现模式。

命令行也能直接运行：

```bash
python -m pip install .
python reproduce/train_from_scratch.py --paper-settings --output paper-main-results
```

首次界面启动需联网安装依赖并保持终端运行；后续复用.ui-venv。计算在本机完成，
无须API密钥。输出目录须为新目录。详细操作见[中文教程](docs/USER_GUIDE_zh-CN.md)。

![论文原设置实际运行结果](docs/paper-results.png)

## 文件导航与下载

- [五表复算数据](reproduce/data/)：保存的预测、已评价参考值及其清单。
- [训练输入与字段说明](reproduce/inputs/README.md)：14,112条整理记录、固定划分、读取方法。
- [论文模型权重](polymer_dmpnn/paper_weights/)：65份表3–5图模型参数。
- [输入CSV示例](examples/polymers.csv)：自己的数据按该格式填写。

可通过仓库绿色 **Code → Download ZIP** 下载并解压整个项目；不要仅下载单个脚本。
在解压后的仓库根目录打开终端，再运行下列命令。`.csv.gz`是已包含在下载包里的压缩CSV，
网页不展开其内容，程序可以直接读取。部分macOS环境中请将命令的`python`换成`python3`。

## 下载后重算论文五张主表

```bash
git clone https://github.com/diudiu-aa/polymer-dmpnn-thermal-conductivity.git
cd polymer-dmpnn-thermal-conductivity
python reproduce/paper_tables.py --verify --output reproduced-paper-tables
```

仅需Python 3.10及以上的标准库，无须安装模型，也不需要原作者的本机项目文件。
程序从117组、40,848行保存预测和参考值重新计算182个均值/样本标准差，
生成五张CSV/Markdown表、逐种子指标及`verification.json`。输出目录须为新目录。
预期数值只用于核验，不作为生成表格的输入结果。

成功时终端显示：

```text
Recomputed Tables 1–5: 182 displayed statistics from 117 saved prediction groups; verified=True
```

在`reproduced-paper-tables/`打开`table_1.csv`至`table_5.csv`查看重算值。
这一步只复算指标，不训练，也不需要先安装PyTorch。重复运行时使用新的输出目录名。


## 安装与预测

```bash
python -m pip install .
polymer-kappa --smiles '*CC*'
polymer-kappa --batch examples/polymers.csv --format csv --output predictions.csv
```

结构必须为连通的重复单元，恰好包含两个通过单键连接的末端`*`。
默认使用早期三种子L1000周期模型集成，保持旧接口兼容。
**三个模型预测的平均值，与论文“分别计算各种子指标再求平均”不是同一统计量。**
输出的种子标准差不是校准后的预测区间。

可明确选择主研究中一个真实种子模型：

```bash
polymer-kappa --list-models
polymer-kappa --paper-model table5_B600_multitask_32101 --smiles '*CC*'
polymer-kappa --paper-model table5_B600_periodic_32101 --batch examples/polymers.csv
```

仓库另含65份表3–5模型权重：开链、周期、排序目标、学习曲线、平衡采样和双任务。
权重逐字节保留，标准化参数和SHA256均可核对。双任务分类头输出原始排序分数，
不是经过校准的高导热概率。

## 权重核验与重新训练

```bash
# 从65份保存权重重新推理，与已保存逐条结果核对。
python reproduce/evaluate_checkpoints.py --output checkpoint-verification

# 先核对固定训练配置；默认不训练。
python reproduce/train_graph_models.py --experiment table5 --arm multitask --seed 32101

# 显式启动该配置的50轮CPU训练。
python reproduce/train_graph_models.py --experiment table5 --arm multitask --seed 32101 --run --output retrained-multitask
```

表3–5所需的固定训练/评价记录已经包含，无须私有的`--project-root`目录。
Ridge/ExtraTrees的训练内分组调参见[详细复现说明](reproduce/README.md)，
额外依赖通过`python -m pip install '.[reproduce]'`安装。

目前覆盖：五张主表的保存预测复算、主要图模型的权重重新评价，以及表3–5从固定
整理输入开始的训练配方。**不宣称从完整原始数据库重新构造整个历史研究**：
旧结构聚类及角色生成、表1的GIN/GPR训练不在上述训练入口中；表1/2有保存结果复算。
同源MD数据、公开OPC队列和复算成功均不等于独立实验验证。

## 测试与来源

```bash
python -m unittest discover -s tests -v
```

CI覆盖Python 3.10/3.11/3.12的安装、推理、逐表复算和模型核验。
原训练环境为Python 3.12.14、PyTorch 2.14.1、NumPy 2.5.4、RDKit 2026.03.3，CPU单线程。
其他支持版本允许使用，但不同库可能带来数值差别；权重推理核验绝对容差为
`1e-5 W/(m K)`。跨平台重训不承诺逐位一致，五表保存结果复算不依赖这些数值库。

代码和项目权重为MIT许可；PolyOmics整理子集保留CC BY 4.0许可，OPC数据为MIT。
来源、固定快照和修改说明见[数据许可与溯源](THIRD_PARTY_DATA.md)。
未上传受限数据库、原始整库、论文或内部项目日志。请记录实际Git提交号来固定版本。

本次原环境实测：65份权重的26,000条推理结果全部与保存值一致；60标签周期模型、
600标签双任务模型和Ridge的三个固定配置复跑，也与原结果一致。
详见[验证记录](reproduce/release_validation.json)。这不是全部配置跨平台重训保证。
