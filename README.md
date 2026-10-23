# Directional Support Sampling

Directional Support Sampling 面向无人机影像中的旋转目标定位，在固定输入像素预算下，根据目标尺度、方向和定位需求分配细节画布。均匀上下文分支与自适应细节分支共享 ResNet-50 编码器，通过可逆坐标映射保持特征读取和旋转框预测的坐标一致性。

## 安装

使用 Python 3.10 或以上版本。先安装适合计算设备的 [PyTorch](https://pytorch.org/get-started/locally/)，再安装项目：

```bash
pip install -e ".[test]"
```

所有命令均在项目根目录执行。旋转 IoU 使用 PyTorch 实现的凸多边形交集，支持 CPU 和 CUDA。模型训练通过 `--device cuda` 使用 GPU。

## 项目结构

```text
dds/
  networks/              ResNet-50、多尺度编码器和旋转注意力
  detector.py            上下文与细节双分支检测
  query_geometry.py      分布式中心、Cholesky 形状和周期方向
  density_router.py      语义、视觉特征和查询状态的密度预测
  transport.py           连续密度 CDF 正向与逆向映射
  geometry.py            方向厚度、非仿射残差和分配目标
  dual_reader.py         四重周期的输入坐标特征读取
  teacher.py             冻结 CLIP 场景、目标区域和文本编码
  losses.py              共享 Hungarian 匹配与联合监督
  data.py                四边形标注、裁块和训练数据
  evaluation.py          原图合并、旋转 NMS 和 AP
configs/                 CODrone、UAV-ROD 和对照配置
tools/                   数据准备、训练、推理和评估入口
tests/                   几何、梯度、检测器和评估测试
```

## 数据准备

使用 DOTA 风格四边形标注。每行依次为四个顶点的八个坐标、类别名和难例标志：

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
```

`difficult=0` 表示普通目标，非零表示难例。`ignore`、`ignored` 和 `###` 表示忽略区域。图片与标注文件使用相同的文件名主体。
CODrone 的 `annfile/` 标注目录也会自动识别。

```text
datasets/CODrone/
  train/images/       train/labelTxt/
  val/images/         val/labelTxt/
  test/images/        test/labelTxt/
datasets/UAV_ROD/
  train/images/       train/labelTxt/
  test/images/        test/labelTxt/
```

CODrone 类别顺序为 `car, truck, traffic-sign, people, motor, bicycle, traffic-light, tricycle, bridge, bus, boat, ship`。UAV-ROD 使用单类别 `car`。

UAV-ROD 的 XML 标注可按官方划分列表转换。`robndbox` 中的 `angle` 按弧度读取：

```bash
python -m tools.convert_uav_rod --images datasets/UAV_ROD_raw/JPEGImages --annotations datasets/UAV_ROD_raw/Annotations --split-file datasets/UAV_ROD_raw/ImageSets/train.txt --output datasets/UAV_ROD/train
python -m tools.convert_uav_rod --images datasets/UAV_ROD_raw/JPEGImages --annotations datasets/UAV_ROD_raw/Annotations --split-file datasets/UAV_ROD_raw/ImageSets/test.txt --output datasets/UAV_ROD/test
```

```bash
python -m tools.prepare_data --config configs/codrone.yaml --source datasets/CODrone --output data/CODrone
python -m tools.prepare_data --config configs/uav_rod.yaml --source datasets/UAV_ROD --output data/UAV_ROD --splits train test
```

原图按 1024×1024 裁块，步长 512，目标可见面积比例至少为 0.7。裁块保留原图身份、偏移和标注路径；训练、验证和测试按原图划分。背景裁块保留用于训练。难例、忽略区域及可见比例不足的目标进入忽略集合。

## 语义向量与预训练权重

```bash
python -m tools.prepare_text --config configs/codrone.yaml --device cuda
python -m tools.prepare_text --config configs/uav_rod.yaml --device cuda
```

文本向量按配置中的类别顺序保存到 `assets/`。训练时，冻结的 CLIP 图像教师提供场景和旋转目标区域特征。`semantic.teacher` 可以使用 Hugging Face 模型名或包含完整 CLIP 权重、配置与 tokenizer 的本地目录。

ResNet-50-vd 的预训练权重在首次训练时自动加载。也可将 `model.backbone_weights` 设置为本地权重路径。推理直接从检测器检查点读取模型参数和文本向量。

## 训练

```bash
python -m tools.train --config configs/codrone.yaml --output outputs/codrone --device cuda
python -m tools.train --config configs/uav_rod.yaml --output outputs/uav_rod --device cuda
```

默认训练参数：

| 参数 | 数值 |
| --- | --- |
| 编码器 | ResNet-50-vd，共享权重 |
| 查询数 / 解码层数 | 300 / 6 |
| 上下文 / 细节画布 | 1024 / 1024 |
| 密度网格 | 33×33 |
| 训练轮数 / 有效批量 | 36 / 10 |
| 优化器 | AdamW |
| 学习率 / 骨干学习率 | 5e-5 / 1e-5 |
| 权重衰减 | 5e-5 |
| 预热 / 调度 | 1 轮 / 余弦衰减 |
| 基础厚度需求 | 16 像素 |
| 非仿射阈值 | 0.1 |
| 密度下界参数 | 0.15 |
| 训练 / 几何评估网格 | 5×5 / 17×17 |

`train.batch_size` 为每步样本数，`train.effective_batch_size` 为梯度累积后的批量。检查点保存模型、优化器、混合精度状态、随机数状态及配置。启用 `train.amp` 可使用混合精度；几何计算保留单精度。
启用 `model.gradient_checkpointing` 可重计算骨干特征，启用 `train.activation_offload` 可将反向计算所需的中间张量暂存到 CPU，以降低训练显存占用。

CODrone 每轮按原图级验证 AP 选择 `best.pth`；UAV-ROD 使用第 36 轮结束后的 `last.pth`。

```bash
python -m tools.train --config configs/codrone.yaml --output outputs/codrone --device cuda --resume outputs/codrone/last.pth
```

五次配对实验分别使用相同的一组随机种子，例如 `42, 43, 44, 45, 46`。使用 `--seed` 指定每次运行的种子，并为每次运行设置独立输出目录。
完成各次评估后，使用 `python -m tools.summarize_runs --metrics run42.json run43.json run44.json run45.json run46.json --output summary.json` 汇总均值和标准差。

## 推理与评估

```bash
python -m tools.predict --checkpoint outputs/codrone/best.pth --index data/CODrone/test.jsonl --output outputs/codrone/test_predictions.jsonl --merged outputs/codrone/test_original.json --device cuda
python -m tools.evaluate --config configs/codrone.yaml --index data/CODrone/test.jsonl --predictions outputs/codrone/test_predictions.jsonl --output outputs/codrone/test_metrics.json
```

推理输出的旋转框采用 `[cx, cy, w, h, theta]`，前四项为裁块像素坐标，`theta` 为弧度。正角方向遵循图像坐标系。原图合并依据裁块偏移恢复位置，并按类别执行 IoU 阈值 0.5 的旋转 NMS。

AP 对 0.50–0.95 的十个 IoU 阈值取均值，采用连续精度包络积分；同时输出 AP50、AP75、各类别 AP 和各阈值 AP。R75 为同类别正确定位目标数占有效目标数的比例，FPI75 为每张原图的误检数。使用 `--score-threshold` 设置 R75 和 FPI75 的置信度工作点。

```bash
python -m tools.evaluate --config configs/codrone.yaml --index data/CODrone/test.jsonl --predictions outputs/codrone/test_predictions.jsonl --output outputs/codrone/small_metrics.json --small
python -m tools.evaluate --config configs/codrone.yaml --index data/CODrone/test.jsonl --predictions outputs/codrone/test_predictions.jsonl --output outputs/codrone/extended_small_metrics.json --small --include-difficult
```

小目标按原图长边归一化到 1024 后的标注多边形面积小于 32² 像素定义。默认评估普通目标；`--include-difficult` 将有有效四边形标注的难例纳入评估。忽略区域不计为误检。

## 分配与需求对照

| 配置 | 分配目标或需求来源 |
| --- | --- |
| `codrone.yaml` | 方向厚度与校正离散度需求 |
| `area.yaml` | 匹配需求的面积分配 |
| `svd.yaml` | 最小奇异值分配 |
| `fixed_demand.yaml` | 固定需求 |
| `scale_demand.yaml` | 短边尺度需求 |
| `raw_variance.yaml` | 原始中心方差需求 |
| `no_deformation.yaml` | 关闭非仿射正则 |

```bash
python -m tools.train --config configs/area.yaml --output outputs/area --seed 42 --device cuda
python -m tools.train --config configs/svd.yaml --output outputs/svd --seed 42 --device cuda
```

厚度为目标坐标中变换后平行四边形面积与最长边之比的下包络。面积分配使用 `S sqrt(abs(det(J_U B)))`，需求阈值乘以目标长宽比的平方根。两者均使用相同目标权重、非仿射约束和画布预算。

中心需求扣除相邻离散节点插值所需的最小方差，再以目标形状归一化。向密度预测器提供的查询几何、前景概率和需求停止梯度，视觉及语义特征保留梯度。

## 几何诊断与速度测量

```bash
python -m tools.diagnose --checkpoint outputs/codrone/best.pth --index data/CODrone/val.jsonl --output outputs/codrone/geometry.jsonl --device cuda
python -m tools.benchmark --checkpoint outputs/codrone/best.pth --size 1024 --device cuda --output outputs/codrone/timing.json
```

几何诊断使用 17×17 目标网格，输出厚度、长度等效面积、最小奇异值和非仿射残差。速度测量预热后使用设备同步计时，输出平均延迟、吞吐率和显存峰值。

## 测试

```bash
python -m pytest -q
```

测试覆盖 CDF 逆映射、解析 Jacobian、密度归一化、逆映射梯度、厚度与面积需求、超额方差、坐标变换不变性、旋转 IoU、四分之一周几何等价性、完整检测器更新、检查点读写及原图级评估。

## 参考

网络组件见 [EAV-DETR](https://github.com/zzzhak/EAV-DETR) 与 [RT-DETR](https://github.com/lyuwenyu/RT-DETR)，语义教师使用 [CLIP](https://huggingface.co/docs/transformers/model_doc/clip)。第三方资源归属见 `THIRD_PARTY_NOTICES.md`。
