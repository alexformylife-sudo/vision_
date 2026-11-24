"""顶部集中配置。已确认项与待现场标定项在字段注释中分开标明。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


HUE_LIMIT = (0, 179)
SV_LIMIT = (0, 255)
MAX_TCP_PORT = 65535
ALLOWED_SIDES = ("L", "R")
ALLOWED_COLORS = ("R", "Y", "B")
SOURCE_TYPES = ("images", "video", "stream")
SEND_POLICIES = ("on_change", "heartbeat")
MULTI_TAG_POLICIES = ("largest_area", "nearest_center")


def clamp_int(value: float, low: int, high: int) -> int:
    return min(max(int(round(value)), low), high)


@dataclass(frozen=True)
class HsvRange:
    h_min: int
    s_min: int
    v_min: int
    h_max: int
    s_max: int
    v_max: int

    def validate(self, name: str) -> None:
        _check_bounds(name + ".h", self.h_min, self.h_max, HUE_LIMIT)
        _check_bounds(name + ".s", self.s_min, self.s_max, SV_LIMIT)
        _check_bounds(name + ".v", self.v_min, self.v_max, SV_LIMIT)

    def lower(self) -> tuple[int, int, int]:
        return (self.h_min, self.s_min, self.v_min)

    def upper(self) -> tuple[int, int, int]:
        return (self.h_max, self.s_max, self.v_max)


@dataclass(frozen=True)
class RatioRoi:
    x0: float
    y0: float
    x1: float
    y1: float

    def validate(self, name: str) -> None:
        for field_name, value in (("x0", self.x0), ("y0", self.y0), ("x1", self.x1), ("y1", self.y1)):
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name}.{field_name} 必须在 [0, 1]，实际 {value}")
        if self.x0 >= self.x1 or self.y0 >= self.y1:
            raise ValueError(f"{name} 必须满足 x0 < x1 且 y0 < y1")

    def pixel_box(self, width: int, height: int) -> tuple[int, int, int, int]:
        x0 = clamp_int(self.x0 * width, 0, width)
        y0 = clamp_int(self.y0 * height, 0, height)
        x1 = clamp_int(self.x1 * width, 0, width)
        y1 = clamp_int(self.y1 * height, 0, height)
        return x0, y0, x1, y1

    def crop(self, frame) -> "CroppedRoi":
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = self.pixel_box(width, height)
        if x1 <= x0 or y1 <= y0:
            return CroppedRoi(image=frame[0:0, 0:0], x0=x0, y0=y0)
        return CroppedRoi(image=frame[y0:y1, x0:x1], x0=x0, y0=y0)


@dataclass(frozen=True)
class CroppedRoi:
    image: object
    x0: int
    y0: int


@dataclass(frozen=True)
class Config:
    # 输入。默认分辨率 640x480 已确认，可改配置。
    source_type: str = "images"
    source_path: str = ""
    frame_width: int = 640
    frame_height: int = 480
    target_fps: float = 15.0  # 可调，防止手机发烫；0 表示不休眠
    reconnect_delay_s: float = 1.0  # 可调
    open_timeout_ms: int = 2000  # 可调，打开流时传给 FFmpeg
    read_timeout_ms: int = 2000  # 可调，打开流时传给 FFmpeg
    stale_frame_timeout_s: float = 1.0  # 可调。超时无新帧只清未确认连续计数，已锁存结果保留
    thread_join_timeout_s: float = 2.0
    image_emit_each: int = 1
    image_suffixes: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    idle_poll_s: float = 0.05
    no_resize: bool = False

    # 已确认：tag36h11，黑码 5cm，立牌下沿离地 10cm。
    tag_family: str = "tag36h11"
    TAG_SIZE_M: float = 0.05
    tag_bottom_height_m: float = 0.10
    tag_id_to_side: Mapping[int, str] = field(default_factory=lambda: {0: "L", 1: "R"})
    tag_roi: RatioRoi = field(default_factory=lambda: RatioRoi(0.0, 0.0, 1.0, 1.0))
    min_marker_perimeter_rate: float = 0.01  # 可调，默认低于 OpenCV 0.03，照顾远处 5cm 标签
    min_tag_side_px: float = 8.0  # 可调，过滤过小噪点
    multi_tag_policy: str = "largest_area"  # 可调：largest_area / nearest_center；同框多码策略待现场确认
    # 可选相机内参，预留单目测距；本版不测距。未标定前保持 None。
    camera_matrix: tuple[tuple[float, float, float], ...] | None = None
    dist_coeffs: tuple[float, ...] | None = None

    # 颜色 ROI / HSV / 形态学：下列数值均为模拟样本导出初值，待赛场灯光与支架视角标定。
    color_roi: RatioRoi = field(default_factory=lambda: RatioRoi(0.05, 0.20, 0.95, 0.98))
    color_roi_scale: float = 1.0  # 可调，<1 时在缩小后的 ROI 上认色以省算力
    red_hsv_low: HsvRange = field(default_factory=lambda: HsvRange(0, 75, 60, 12, 255, 255))
    red_hsv_high: HsvRange = field(default_factory=lambda: HsvRange(168, 75, 60, 179, 255, 255))
    yellow_hsv: HsvRange = field(default_factory=lambda: HsvRange(22, 68, 52, 33, 255, 255))
    color_yellow_area_weight: float = 0.72  # 选 dominant 时对黄掩膜面积打折，削弱误抢蓝/红
    blue_hsv: HsvRange = field(default_factory=lambda: HsvRange(100, 75, 60, 125, 255, 255))
    color_rb_mask_min_mean_s: float = 78.0  # 红/蓝连通域均值，滤掉黑屏 hue 漂移
    color_rb_mask_min_mean_v: float = 62.0
    color_min_area: int = 200  # 按掩膜总像素计，不靠碎轮廓面积
    color_min_height_width: float = 0.35
    color_min_extent: float = 0.15  # 暗部/高光导致掩膜稀疏时仍过
    color_morph_kernel: int = 1  # 开运算核；1 近似不做
    color_morph_close_kernel: int = 7  # 连通色块内缝隙；过大易把桌面噪点并成一片黄
    hsv_center_ratio: float = 0.3  # 调参模式：颜色 ROI 中心区域占比，可调
    hsv_percentiles: tuple[int, ...] = (10, 50, 90)

    # 防抖：默认连续 5 帧；T 秒为 0 表示只用帧数。均可调。
    stable_frame_count: int = 5
    stable_seconds: float = 0.0

    # 输出。TCP 主机与端口待确认，以下只是可改初值。
    print_enabled: bool = True
    tcp_enabled: bool = False
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 9000
    tcp_connect_timeout_s: float = 0.3
    tcp_send_timeout_s: float = 0.2
    send_policy: str = "on_change"
    heartbeat_interval_s: float = 0.5

    save_debug: bool = False
    debug_dir: str = "debug_frames"  # 相对路径落在本包目录，避免写入上级已验收 logs/
    print_hsv_stats: bool = False
    hsv_calib: bool = False

    invalid_side: str = "N"
    invalid_tag_id: int = -1
    invalid_color: str = "N"

    def __post_init__(self) -> None:
        _validate_config(self)


def default_debug_dir() -> str:
    return str((__import__("pathlib").Path(__file__).resolve().parent / "debug_frames"))


def make_config(**overrides) -> Config:
    if "debug_dir" not in overrides:
        overrides["debug_dir"] = default_debug_dir()
    return Config(**overrides)


def _check_bounds(name: str, low: int, high: int, allowed: tuple[int, int]) -> None:
    min_value, max_value = allowed
    if low < min_value or high > max_value or low > high:
        raise ValueError(f"{name} 必须满足 {min_value} <= min <= max <= {max_value}")


def _validate_config(config: Config) -> None:
    if config.source_type not in SOURCE_TYPES:
        raise ValueError("source_type 必须是 images、video 或 stream")
    if config.tag_family != "tag36h11":
        raise ValueError("家族仅支持 tag36h11")
    if config.frame_width < 1 or config.frame_height < 1:
        raise ValueError("画面宽高必须 >= 1")
    if config.stable_frame_count < 1 or config.image_emit_each < 1:
        raise ValueError("stable_frame_count 与 image_emit_each 必须 >= 1")
    if config.stable_seconds < 0 or config.target_fps < 0:
        raise ValueError("stable_seconds 与 target_fps 不能为负")
    if config.stale_frame_timeout_s <= 0:
        raise ValueError("stale_frame_timeout_s 必须 > 0")
    if config.color_min_area < 1 or config.color_morph_kernel < 1:
        raise ValueError("color_min_area 与 color_morph_kernel 必须 >= 1")
    if config.color_morph_close_kernel < 1:
        raise ValueError("color_morph_close_kernel 必须 >= 1")
    if not 0.0 < config.color_yellow_area_weight <= 1.0:
        raise ValueError("color_yellow_area_weight 必须在 (0, 1]")
    if config.color_rb_mask_min_mean_s < 0 or config.color_rb_mask_min_mean_v < 0:
        raise ValueError("color_rb_mask_min_mean_s/v 不能为负")
    if config.TAG_SIZE_M <= 0.0 or config.tag_bottom_height_m <= 0.0:
        raise ValueError("TAG_SIZE_M 与 tag_bottom_height_m 必须 > 0")
    if config.min_marker_perimeter_rate <= 0.0 or config.min_tag_side_px <= 0.0:
        raise ValueError("min_marker_perimeter_rate 与 min_tag_side_px 必须 > 0")
    if not 0.0 < config.color_roi_scale <= 1.0:
        raise ValueError("color_roi_scale 必须在 (0, 1]")
    if not 0.0 < config.hsv_center_ratio <= 1.0:
        raise ValueError("hsv_center_ratio 必须在 (0, 1]")
    if config.tcp_port < 1 or config.tcp_port > MAX_TCP_PORT:
        raise ValueError(f"tcp_port 必须在 1..{MAX_TCP_PORT}")
    if config.send_policy not in SEND_POLICIES:
        raise ValueError("send_policy 必须是 on_change 或 heartbeat")
    if config.heartbeat_interval_s < 0:
        raise ValueError("heartbeat_interval_s 不能为负")
    if config.multi_tag_policy not in MULTI_TAG_POLICIES:
        raise ValueError("multi_tag_policy 必须是 largest_area 或 nearest_center")
    _validate_rois_and_hsv(config)
    _validate_tag_map(config.tag_id_to_side)


def _validate_rois_and_hsv(config: Config) -> None:
    config.tag_roi.validate("tag_roi")
    config.color_roi.validate("color_roi")
    config.red_hsv_low.validate("red_hsv_low")
    config.red_hsv_high.validate("red_hsv_high")
    config.yellow_hsv.validate("yellow_hsv")
    config.blue_hsv.validate("blue_hsv")


def _validate_tag_map(tag_id_to_side: Mapping[int, str]) -> None:
    if not tag_id_to_side:
        raise ValueError("tag_id_to_side 不能为空")
    for tag_id, side in tag_id_to_side.items():
        if not isinstance(tag_id, int) or isinstance(tag_id, bool) or tag_id < 0:
            raise ValueError(f"映射表键必须是非负整数，实际 {tag_id}")
        if side not in ALLOWED_SIDES:
            raise ValueError(f"映射表侧别必须是 L 或 R，实际 {side}")
