# SCG-VTNet 修回代码

此目录包含 2026-11-15 修回工作区的识别头、对照实验和特征提取代码。仓库根目录的 AC-GCAT 是较早的实现；其运行示例和全局视觉输入不能直接用于本目录。发布版本只含源码，不附原始数据、特征文件、划分清单、权重或审稿材料。

| 目录 | 用途 |
|---|---|
| `core/` | SCG-VTNet 主模型、消融和鲁棒性评估 |
| `baselines/` | E1–E7 对照实验 |
| `extraction/` | MediaPipe 骨架、ResNet-50 空间 token、EfficientNet-B0 和 ViT-S/16 特征提取 |
| `protocol/` | NUS Hand Posture 分组隔离实验入口 |

## 环境与数据

开发环境使用了 Python 3.10、PyTorch 2.7.1 和 torchvision 0.22.1。按本机配置安装对应的 PyTorch / torchvision 后，从仓库根目录执行：

```sh
python -m pip install -r scg_vtnet/core/requirements.txt
```

从原图提取特征还需 MediaPipe（开发环境使用 0.10.14）、OpenCV、Pillow 和 tqdm；提取 ViT 特征还需 timm。提取脚本的参数用 `--help` 查看。

骨架 CSV 与视觉特征数组必须按相同样本顺序保存。主模型的 ResNet-50 输入为 `N × 49 × 2048` 空间 token。运行主模型或基线时，使用放在独立工作目录中的骨架 CSV 副本：训练加载器会在该 CSV 旁写入划分清单。原始数据和既有实验结果保留只读备份。

## 运行入口

下列命令在仓库根目录运行；先替换数据路径，再选择实验和种子。示例不等同于论文所有实验的复现设置。

```sh
python scg_vtnet/core/main.py --study_name sota_only --seed 42 --train_csv_path scratch/zerofill_landmarks_mask.csv --image_features_path data/zerofill_visual_tokens_layer4_compact.npy --output_dir results/core
python scg_vtnet/baselines/main.py --study_name fair_e1_e7 --seed 42 --train_csv_path scratch/zerofill_landmarks_mask.csv --image_features_path data/zerofill_visual_tokens_layer4_compact.npy --global_image_features_path data/zerofill_visual_global_from_tokens.npy --efficientnet_features_path data/efficientnet_b0_global.npy --vit_features_path data/vit_s16_tokens.npy --output_dir results/e1_e7
python scg_vtnet/protocol/run_nus_strict.py --processed-dir data/NUS_Hand_Posture/processed_spatial_tokens --split-manifest splits/NUS_grouped_frozen_seed20261115.csv --output-dir results/nus_grouped_strict
```

完整 E1–E7 对照需要各自的特征文件，不能把代理特征的结果当成原始 E1–E7 对照。NUS 入口要求提供与特征行序一致的分组划分清单（含 `original_index`、`split`、`group_id`），输出目录不得已存在。分组隔离不等同于受试者独立划分。


## Revised evaluation code (2026-11-21)

See [reproduction/README.md](reproduction/README.md) for the source and entry points for the matched ASL, paired degradation, grouped NUS, timing and figure evaluations.
