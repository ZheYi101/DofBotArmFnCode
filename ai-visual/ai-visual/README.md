# DOFBOT OpenCV 视觉识别抓取

程序先让机械臂上的摄像头在底座 60°、90°、120° 三个观察姿态粗扫地图，再围绕发现目标的最佳角度做 5° 间隔的局部复扫，并在线性插值得到的中心角两侧各补拍一帧。坐标换算采用底座舵机的实际读回角度，而不是只相信指令角度。每个视角等待舵机稳定后采集多帧，将像素坐标旋转到统一工作坐标系并去重，再使用每个目标的实时 `(x, y)` 坐标求逆解并抓取，依次堆到 1～4 层。

地图上印刷的彩色方块带有完整白色圆环且下方没有盒身；真正积木的彩色顶面下方有盒身边缘和纹理。识别器同时使用这两个条件过滤平面图案，因此也能在浅色木桌上识别立体积木。

坐标换算使用彩色顶面轮廓中心。这与设备自带颜色抓取程序的标定方式一致；像素到工作区公式本身已经包含相机俯角和物块高度补偿，不能再次把中心外推到可见底边。`scan_map.json` 会保存全部复扫候选坐标、最终采用视角和多视角离散度。

## 安全设计

- 默认仅识别，不初始化机械臂、不发送舵机命令。
- 只有显式添加 `--execute` 才会动作。
- 多帧检测结果必须达到命中数和抖动阈值。
- 接触画面边缘、中心位置不可靠的截断轮廓会被丢弃。
- 执行抓取前，目标必须被至少两个视角确认，且多视角坐标离散度不能超过配置阈值。
- 抓取时先在高位对准底座，再慢速下探；程序读取实际关节角，偏差超过 2° 时修正一次，超过 5° 时不会闭合夹爪。
- 所有抓取姿态会在第一次抓取前统一检查舵机范围；动态模式还会先完成全部逆解。
- 扫描原图保存为 `scan/raw_*.jpg`，标注图保存为 `scan/scan_*.jpg`，融合地图保存在 `scan_map.json`。

## 使用

先停止占用摄像头的 Jupyter 单元（务必执行其中的 `cap.release()` 或 `image.release()`），然后：

```bash
cd ~/workspace/ai-visual

# 只看当前画面，机械臂不会动作
python3 visual_pick_stack.py

# 舵机和摄像头配合扫描地图；不操作夹爪、不抓取
python3 visual_pick_stack.py --scan-map

# 检查 scan/ 和 scan_map.json 后，重新扫描并按实时坐标抓取堆叠
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
python3 visual_pick_stack.py --execute
```

程序不保存、也不读取按颜色预设的抓取坐标。`--execute` 只能使用本轮扫描生成并通过多视角一致性检查的坐标。

指定识别/堆叠顺序：

```bash
python3 visual_pick_stack.py --colors yellow,red,green,blue
```

离线图片测试：

```bash
python3 visual_pick_stack.py --image test.jpg --snapshot annotated.jpg
```

## 调参

颜色阈值、立体积木过滤阈值、扫描姿态、夹爪角度和堆叠位均在 `config.json`。当前扫描姿态已经在本机实拍验证，其他舵机参数沿用给定静态代码。

如果报摄像头占用：

```bash
fuser -v /dev/video0
```

不要同时运行 Jupyter 相机程序和本脚本。若颜色漏检，优先根据现场光照调整 `config.json` 的 HSV 饱和度/亮度下限和 `min_area`，先反复运行无动作模式确认。
