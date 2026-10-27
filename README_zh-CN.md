# 聚合物热导率周期图预测工具

本仓库提供已保存的 R04 周期三聚体 D-MPNN 模型集成，三份权重总计约 1 MB，可在 CPU 上预测。输出单位为 W/(m·K)，目标对应约 **300 K、1 atm、未拉伸的各向同性无定形聚合物体相模拟值**。仓库不包含训练数据。

模型学习的是 PolyOmics/RadonPy 分子动力学计算结果，预测值不能直接当作实验测量值。OPC 结果是公开计算数据的回顾性评价。对于陌生结构和高导热候选，应结合进一步计算或实验验证。

## 安装

使用 Python 3.10–3.13，先安装 Python，然后在终端运行：

```bash
git clone <YOUR_REPOSITORY_URL>
cd polymer-dmpnn-github-repo
python -m pip install .
```

若还没有发布到 GitHub，可直接在本目录运行最后一行。依赖也列在 `requirements.txt` 中。

## 单条和批量预测

重复单元 SMILES 需要恰好两个连接点 `*`，两个连接点都只能通过单键连接，并且结构需连通。终端中请用引号包住输入：

```bash
polymer-kappa --smiles '*CC*'
python -m polymer_dmpnn --smiles '*CC(*)C'
```

批量 CSV 需要 `smiles` 列，可选 `id` 列：

```bash
polymer-kappa --batch examples/polymers.csv --format csv --output predictions.csv
```

也可输入每行一条 SMILES 的 UTF-8 文本文件。`--column` 可指定 CSV 中的结构列，`--format json` 输出 JSON。无效结构会保留为错误行；只要有一行失败，程序退出状态为 2。

输出包含三个模型的预测、平均值和三个模型的总体标准差。**这个标准差不是经过校准的预测区间**。旧脚本中的固定保形区间未纳入本包。

## 测试与兼容性

```bash
python -m unittest discover -s tests -v
```

模型训练时使用 Python 3.12.14、PyTorch 2.14.1、NumPy 2.5.4 和 RDKit 2026.03.3；同一 Python/PyTorch 环境下，RDKit 2025.09.6 也通过了六项本地测试，包含四个参考结构。在 macOS ARM64 的干净 Python 3.12 环境中安装时，自动选用的 RDKit 2026.09.1 通过了单条和批量预测检查。代码已放宽 RDKit 的硬版本限制；其他版本允许安装，但兼容性尚未验证，运行时会提示版本差异；请先用参考结构检查结果。`tests/golden_predictions.json` 保存了原始锁定实现对四个示例的参考预测。远程 CI 矩阵尚未运行。

## 来源

三份 L1000 周期模型权重和各自目标标准化参数、SHA-256 均保存在 `polymer_dmpnn/weights/manifest.json`。模型使用 PolyOmics 数据，数据来源为 [DOI 10.57967/hf/7475](https://doi.org/10.57967/hf/7475) 和 [数据集页面](https://huggingface.co/datasets/yhayashi1986/PolyOmics)。D-MPNN 与周期聚合物图的先行文献及许可说明见英文 README 和 `NOTICE.md`。GitHub 仓库地址为 https://github.com/diudiu-aa/polymer-dmpnn-thermal-conductivity；目前尚未申请归档 DOI。


严格结构簇五种子训练与评价入口、额外依赖及所需预处理输入见 [复现说明](reproduce/README.md)。从原始数据库完整生成预处理表的代码仍待整理。
