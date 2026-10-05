"""生成应用图标 voicenote.ico。

用代码画而不是往仓库里塞一个 .ico：图标要调的时候改代码就行，
而且 .ico 是二进制，进 git 之后每次改动都是一团看不清的 diff。

用法：
    .venv\\Scripts\\python.exe packaging\\make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

# 和托盘「监听中」用同一个绿，视觉上对得上
GREEN = (46, 160, 67)
WHITE = (255, 255, 255)
SIZES = [16, 24, 32, 48, 64, 128, 256]

# 按 4 倍超采样再缩回去，小尺寸下边缘才不会有锯齿
SUPERSAMPLE = 4


def render(size: int) -> Image.Image:
    s = size * SUPERSAMPLE
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 圆角方块底
    draw.rounded_rectangle(
        (0, 0, s - 1, s - 1), radius=int(s * 0.22), fill=GREEN
    )

    # 话筒胶囊
    cx = s / 2
    cap_w = s * 0.26
    cap_h = s * 0.40
    cap_top = s * 0.19
    draw.rounded_rectangle(
        (cx - cap_w / 2, cap_top, cx + cap_w / 2, cap_top + cap_h),
        radius=cap_w / 2,
        fill=WHITE,
    )

    # 托住话筒的弧
    arc_w = s * 0.48
    arc_top = cap_top + cap_h * 0.42
    draw.arc(
        (cx - arc_w / 2, arc_top, cx + arc_w / 2, arc_top + arc_w * 0.55),
        start=0,
        end=180,
        fill=WHITE,
        width=max(1, int(s * 0.055)),
    )

    # 立柱
    stem_w = s * 0.07
    stem_top = arc_top + arc_w * 0.27
    draw.rounded_rectangle(
        (cx - stem_w / 2, stem_top, cx + stem_w / 2, s * 0.81),
        radius=stem_w / 2,
        fill=WHITE,
    )

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    out = Path(__file__).with_name("voicenote.ico")
    # PIL 的 ICO 插件会拿这张基准图去缩放出 sizes 里列的各档
    render(max(SIZES)).save(out, format="ICO", sizes=[(s, s) for s in SIZES])
    print(f"已生成 {out}（{out.stat().st_size / 1024:.1f} KB，{len(SIZES)} 档尺寸）")


if __name__ == "__main__":
    main()
