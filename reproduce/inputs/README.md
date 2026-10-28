# Fixed training and evaluation inputs 数据字段与读取方法

This directory contains the complete curated input subset required by the
published Table 3–5 graph-model training and checkpoint-evaluation entry points.
It does not contain the full original PolyOmics database.

本目录已包含主要模型所需的固定输入，无须向作者索取私人项目文件。

| 文件 | 内容 |
|---|---|
| `records.csv.gz` | 14,112条不同记录的结构、来源标识、簇/组与模拟热导率 |
| `splits.json` | 随机种子32101–32105下，table3/table4/table5的训练和评价记录ID及原有顺序 |
| `manifest.json` | 来源快照、原始文件与整理文件SHA256、记录数和原训练环境 |

## records.csv.gz 字段

| 列名 | 含义 |
|---|---|
| `record_id` | 本研究记录ID，对应划分文件中的ID |
| `source_uuid` | 原PolyOmics来源UUID，支持回溯 |
| `smiles` | 两个末端`*`连接点的重复单元SMILES |
| `isolation_group` | 保守结构隔离组ID |
| `cluster_id` | 高层相似结构簇ID |
| `kappa_W_mK` | 约300 K、1 atm下未拉伸无定形体相的MD热导率，单位W/(m K)，不是新实验测量 |

## 不解压也可以读取

复现脚本自动处理gzip。若只想查看数据，可在仓库根目录运行以下Python代码：

```python
import csv
import gzip
import json

with gzip.open("reproduce/inputs/records.csv.gz", "rt", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    print(reader.fieldnames)
    first_record = next(reader)
    print(first_record)

with open("reproduce/inputs/splits.json", encoding="utf-8") as f:
    splits = json.load(f)
print(len(splits["32101"]["table5"]["train"]))       # 600
print(len(splits["32101"]["table5"]["validation"]))  # 400
```

如果希望在Excel等软件中打开，可解压为普通CSV。下面命令不修改原压缩文件，
也不会覆盖已经存在的`records.csv`：

```python
import gzip
import shutil

with gzip.open("reproduce/inputs/records.csv.gz", "rb") as source:
    with open("records.csv", "xb") as output:
        shutil.copyfileobj(source, output)
```

这些整理标签已经公开，用于复现实验；不要把它们当成未查看过的新盲测标签。
同一记录可被多个种子在同一角色中复用。拟合/评价角色按已发布簇、隔离组、
记录及来源UUID分离；不要自行重排训练序列后声称完全相同的配方。
Table4的600标签点复用Table3模型。

## 来源和复现范围

来源为[PolyOmics](https://huggingface.co/datasets/yhayashi1986/PolyOmics)，
固定快照`041e5834ea1a48682fae12dc39ccd723bcd4f771`，
[DOI 10.57967/hf/7475](https://doi.org/10.57967/hf/7475)，保留CC BY 4.0许可。
原始记录经筛选、规范化和分组后形成该子集；修改及许可见
[THIRD_PARTY_DATA.md](../../THIRD_PARTY_DATA.md)。

五张主表的保存预测数据另见[../data/](../data/)。
本目录支持固定整理输入起步的训练，不负责从完整原始数据库重新构建所有历史划分。
