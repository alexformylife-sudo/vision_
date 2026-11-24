# Termux 实车视觉识别

给车载安卓手机（Termux，目标 Python 3.14 + OpenCV 5.1.0 + NumPy）用的视觉程序。无 GUI，只依赖标准库、`cv2`、`numpy`。AprilTag 只用 `cv2.aruco`（`DICT_APRILTAG_36h11` + `ArucoDetector`）。

本目录不覆盖上一级已验收的练习台模块与日志。练习台报文不含 ALIGN；本程序按实车约定输出：

`SIDE=L;TAG=0;COLOR=R;ALIGN=1` 或 `SIDE=N;TAG=-1;COLOR=N;ALIGN=0`

`ALIGN=1` 表示本次任务 Tag 与颜色都已各自确认，不是当前帧两路同时有效。确认值锁存到进程结束，重启才重新识别。只确认一路时另一路仍为无效值，例如 `SIDE=L;TAG=0;COLOR=N;ALIGN=0`。

模块与数据流见同目录 `设计.md`。

## 运行环境说明

本机盒子自测是 Python 3.13.5 + OpenCV 5.1.0 + NumPy 2.5.5，按 OpenCV 5.x 的 `getPredefinedDictionary` / `DetectorParameters` / `ArucoDetector` / `DICT_APRILTAG_36h11` / `CORNER_REFINE_SUBPIX` 编写。尚未在 Termux Python 3.14 实机上跑过。Termux 上若已安装 OpenCV 与 NumPy，把下面的 `.venv/bin/python` 换成 `python` 即可。

## 目录与运行位置

仓库根目录（本 README 所在层）下有一层 Python 包目录 `termux_vision/`（含 `app.py` 等）。**请在仓库根目录执行** `python -m termux_vision`，这样 `from termux_vision.xxx` 才能正确解析。若工程仍嵌在更大的「月光杯视觉报文」父项目里，则 `cd` 到**能看见子文件夹 `termux_vision` 的那一层**即可，命令不变。

```bash
cd /path/to/termux_vision   # 仓库根，不是内层 termux_vision/ 包目录
```

实时流（HTTP / RTSP，独立线程只保留最新帧）：

```bash
.venv/bin/python -m termux_vision --source stream --path http://127.0.0.1:8080/video
.venv/bin/python -m termux_vision --source rtsp --path rtsp://127.0.0.1:8554/live
```

本地图片或图片文件夹、本地视频：

```bash
.venv/bin/python -m termux_vision --source images --path /path/to/dir
.venv/bin/python -m termux_vision --source images --path /path/to/one.png
.venv/bin/python -m termux_vision --source video --path /path/to/clip.mp4
```

调试存图（标注两个 ROI 和检测框，只用 `imwrite`）：

每次运行会在 `--debug-dir` 下新建时间子目录（如 `20251124_194530/`），其内 `frame_0001.png` 连续编号，避免与历次测试文件混在同一层。

```bash
.venv/bin/python -m termux_vision --source images --path /path/to/dir \
  --save-debug --debug-dir /sdcard/vision_debug
```

HSV 调参（保存标出 Tag ROI、颜色 ROI 及其中心区的截图，并打印中心区 HSV 均值与分位数）：

```bash
.venv/bin/python -m termux_vision --source images --path /path/to/dir --hsv-calib
```

离线看防抖（同一张图重复 5 次，默认连续 5 帧后才 ALIGN=1）：

```bash
.venv/bin/python -m termux_vision --source images --path /path/to/dir \
  --emit-each 5 --stable-frames 5 --fps 0 --send-policy heartbeat --heartbeat 0
```

现场常用覆盖项：

```bash
python -m termux_vision --source http --path http://手机IP:8080/video \
  --tag-map 0:L,1:R --tag-roi 0,0,1,1 --color-roi 0.05,0.2,0.95,0.98 \
  --stable-frames 5 --min-color-area 80 --fps 15 \
  --tcp 127.0.0.1:9000 --send-policy on_change \
  --save-debug --debug-dir /sdcard/vision_debug --hsv-stats
```

生成合成烟雾测试图（不是赛场样本）：

```bash
.venv/bin/python -m termux_vision.make_smoke --out /tmp/termux_vision_smoke
```

## 待现场标定

已确认：`tag36h11`、`TAG_SIZE_M=0.05`、立牌离地 10cm、默认工作分辨率 640×480、判侧只用 `tag_id_to_side`（示例 `{0: "L", 1: "R"}`）。

仍待赛场灯光、支架视角和电控对端确认：

- HSV 上下限（当前为模拟样本导出初值）
- Tag ROI / 颜色 ROI 比例（Tag 默认全图，颜色默认画面中下部）
- `tag_id_to_side` 的最终 ID
- `color_min_area`、高宽比、extent、形态学核大小
- `min_marker_perimeter_rate`、`min_tag_side_px`、多 Tag 择优策略
- 防抖帧数 / 秒数、目标帧率
- `stale_frame_timeout_s`（默认 1.0s。超过后只清未确认连续计数，已锁存结果保留；未超时的空读不清计数）、`open_timeout_ms` / `read_timeout_ms`（打开流时传给 FFmpeg）
- IP Webcam / RTSP 地址
- TCP 主机与端口
- 相机内参（已留接口，本版不测距）
