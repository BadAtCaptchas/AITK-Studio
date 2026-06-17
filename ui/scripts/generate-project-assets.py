from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public" / "assets" / "projects"
OUT.mkdir(parents=True, exist_ok=True)

SCALE = 3

BG = (10, 10, 10, 255)
SURFACE = (14, 16, 18, 255)
PANEL = (23, 23, 23, 255)
PANEL_2 = (18, 20, 23, 255)
BORDER = (38, 38, 38, 255)
BORDER_SOFT = (32, 32, 32, 255)
MUTED = (82, 82, 82, 255)
CYAN = (103, 232, 249, 255)
CYAN_DIM = (22, 111, 122, 255)
EMERALD = (52, 211, 153, 255)
EMERALD_DIM = (22, 101, 52, 255)
AMBER = (245, 158, 11, 255)


def sc(value):
    if isinstance(value, tuple):
        return tuple(int(v * SCALE) for v in value)
    if isinstance(value, list):
        return [sc(item) for item in value]
    return int(value * SCALE)


def rgba(color, alpha):
    return color[:3] + (alpha,)


def new_canvas(size):
    image = Image.new("RGBA", sc(size), BG)
    draw = ImageDraw.Draw(image, "RGBA")
    return image, draw


def finish(image, name):
    base = Image.new("RGBA", image.size, BG)
    base.alpha_composite(image)
    image = base
    image = image.resize((image.width // SCALE, image.height // SCALE), Image.Resampling.LANCZOS)
    image.convert("RGB").save(OUT / name)


def line(draw, points, color=BORDER, width=1):
    draw.line([sc(point) for point in points], fill=color, width=sc(width), joint="curve")


def rect(draw, box, fill=None, outline=BORDER, width=1, radius=2):
    draw.rounded_rectangle(sc(box), radius=sc(radius), fill=fill, outline=outline, width=sc(width))


def glow_line(draw, points, color=CYAN, width=1):
    line(draw, points, rgba(color, 42), width + 3)
    line(draw, points, rgba(color, 95), width)


def grid(draw, size, step=24):
    width, height = size
    for x in range(step, width, step):
        line(draw, [(x, 0), (x, height)], rgba(BORDER, 28), 1)
    for y in range(step, height, step):
        line(draw, [(0, y), (width, y)], rgba(BORDER, 22), 1)


def node(draw, x, y, color=CYAN, r=3):
    draw.ellipse(sc((x - r - 2, y - r - 2, x + r + 2, y + r + 2)), fill=rgba(color, 26))
    draw.ellipse(sc((x - r, y - r, x + r, y + r)), fill=rgba(color, 165), outline=rgba(color, 230), width=sc(1))


def folder_mark(draw, x, y, w, h, color=CYAN):
    tab_h = h * 0.26
    tab_w = w * 0.36
    points = [
        (x, y + tab_h),
        (x + tab_w, y + tab_h),
        (x + tab_w + 10, y),
        (x + w * 0.6, y),
        (x + w * 0.6 + 10, y + tab_h),
        (x + w, y + tab_h),
        (x + w, y + h),
        (x, y + h),
        (x, y + tab_h),
    ]
    draw.polygon(sc(points), fill=rgba(PANEL_2, 210), outline=rgba(color, 120))
    line(draw, [(x + 9, y + h * 0.55), (x + w - 9, y + h * 0.55)], rgba(BORDER, 190), 1)


def tiny_image(draw, box, color=CYAN):
    x1, y1, x2, y2 = box
    rect(draw, box, fill=rgba(PANEL, 150), outline=rgba(color, 95), width=1, radius=2)
    line(draw, [(x1 + 10, y2 - 14), (x1 + 25, y2 - 31), (x1 + 37, y2 - 19), (x2 - 10, y2 - 36)], rgba(color, 95), 1)
    draw.ellipse(sc((x2 - 22, y1 + 10, x2 - 15, y1 + 17)), fill=rgba(color, 90))


def tiny_table(draw, box, color=CYAN):
    x1, y1, x2, y2 = box
    rect(draw, box, fill=rgba(PANEL, 150), outline=rgba(color, 92), width=1, radius=2)
    for x in (x1 + 18, x1 + 37):
        line(draw, [(x, y1 + 8), (x, y2 - 8)], rgba(BORDER, 150), 1)
    for y in (y1 + 17, y1 + 30, y1 + 43):
        line(draw, [(x1 + 8, y), (x2 - 8, y)], rgba(BORDER, 150), 1)
    rect(draw, (x1 + 8, y1 + 8, x2 - 8, y1 + 15), fill=rgba(color, 28), outline=None)


def tiny_stack(draw, box, color=EMERALD):
    x1, y1, x2, y2 = box
    for i in range(4):
        offset = i * 5
        rect(draw, (x1 + offset, y1 + offset, x2 + offset, y2 + offset), fill=rgba(PANEL, 120), outline=rgba(color, 70 - i * 9), width=1, radius=2)


def badge():
    image, draw = new_canvas((512, 512))
    grid(draw, (512, 512), 36)
    rect(draw, (38, 38, 474, 474), fill=rgba(SURFACE, 245), outline=BORDER, width=1, radius=6)
    rect(draw, (72, 72, 440, 440), fill=rgba(BG, 175), outline=BORDER_SOFT, width=1, radius=4)
    rect(draw, (112, 112, 400, 400), fill=rgba(PANEL, 118), outline=BORDER, width=1, radius=4)
    folder_mark(draw, 154, 164, 204, 142, CYAN)
    folder_mark(draw, 176, 190, 204, 142, EMERALD)
    glow_line(draw, [(150, 354), (230, 306), (298, 318), (366, 260)], CYAN, 1)
    node(draw, 150, 354, CYAN, 5)
    node(draw, 298, 318, EMERALD, 5)
    node(draw, 366, 260, AMBER, 4)
    rect(draw, (112, 112, 400, 134), fill=rgba(CYAN, 8), outline=rgba(BORDER, 170), width=1, radius=2)
    rect(draw, (112, 378, 400, 400), fill=rgba(EMERALD, 6), outline=rgba(BORDER, 170), width=1, radius=2)
    finish(image, "project-badge-default.png")


if __name__ == "__main__":
    badge()
