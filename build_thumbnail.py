#!/usr/bin/env python3
"""YouTube thumbnail generator — Jeremy AI × Gemma 4 — 1280×720"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math, os

W, H = 1280, 720
OUT  = os.path.expanduser("~/Desktop/jeremy_gemma4_thumb.jpg")

FONT_BLACK  = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
FONT_IMPACT = "/System/Library/Fonts/Supplemental/Impact.ttf"
FONT_SERIF  = "/System/Library/Fonts/Supplemental/Iowan Old Style.ttc"
FONT_DIN    = "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf"

INK    = (10,  17,  36)
INK2   = (18,  28,  58)
GOLD   = (212, 161,  58)
GOLD_H = (240, 199, 104)
GEMMA  = (91,  179, 248)
WHITE  = (255, 255, 255)
PAPER  = (246, 239, 225)
BLACK  = (0,   0,   0)


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i]-c1[i])*t) for i in range(3))


img = Image.new("RGB", (W, H), INK)
draw = ImageDraw.Draw(img)

# ── BACKGROUND gradient (ink → slightly lighter ink) ──────────────────────
for y in range(H):
    t = y / H
    c = lerp_color(INK, INK2, t)
    draw.line([(0, y), (W, y)], fill=c)

# ── HERO PHOTO — faded finger-touch, centered ─────────────────────────────
HERO_PATH = os.path.expanduser(
    "~/Documents/Money_Codes/pro-se-first-mile/docs/heroes/hero-04-chart.png"
)
if os.path.exists(HERO_PATH):
    hero = Image.open(HERO_PATH).convert("RGBA")
    # Scale to cover canvas while keeping aspect ratio
    hero_ratio = hero.width / hero.height
    canvas_ratio = W / H
    if hero_ratio > canvas_ratio:
        new_h = H
        new_w = int(H * hero_ratio)
    else:
        new_w = W
        new_h = int(W / hero_ratio)
    hero = hero.resize((new_w, new_h), Image.LANCZOS)
    # Center crop
    x_off = (new_w - W) // 2
    y_off = (new_h - H) // 2
    hero = hero.crop((x_off, y_off, x_off + W, y_off + H))
    # Set opacity — faded but visible (40 out of 255 ≈ 16%)
    r, g, b, a = hero.split()
    a = a.point(lambda p: int(p * 0.72))
    hero.putalpha(a)
    img.paste(hero, (0, 0), hero)

# ── DARK SCRIM — left half heavier so text stays readable ─────────────────
scrim = Image.new("RGBA", (W, H), (0, 0, 0, 0))
sd = ImageDraw.Draw(scrim)
# Left third: heavy scrim so JEREMY AI text pops
for x in range(W // 2):
    t = 1 - (x / (W // 2))          # 1 at left edge → 0 at center
    alpha = int(160 * t)
    sd.line([(x, 0), (x, H)], fill=(8, 14, 30, alpha))
img = Image.alpha_composite(img.convert("RGBA"), scrim).convert("RGB")

# ── SUBTLE VIGNETTE ────────────────────────────────────────────────────────
vign = Image.new("RGBA", (W, H), (0, 0, 0, 0))
vd   = ImageDraw.Draw(vign)
steps = 60
for i in range(steps):
    alpha = int(120 * (i / steps) ** 2)
    shrink = i * 6
    vd.rectangle([shrink, shrink, W-shrink, H-shrink],
                 outline=(0, 0, 0, alpha))
img.paste(Image.alpha_composite(img.convert("RGBA"), vign).convert("RGB"))

# ── LEFT GOLD ACCENT BAR ───────────────────────────────────────────────────
draw = ImageDraw.Draw(img)
bar_x = 64
draw.rectangle([bar_x, 90, bar_x+6, H-90], fill=GOLD)

# ── TOP GEMMA ACCENT LINE ─────────────────────────────────────────────────
draw.rectangle([0, 0, W, 6], fill=GEMMA)

# ── BOTTOM GOLD LINE ──────────────────────────────────────────────────────
draw.rectangle([0, H-6, W, H], fill=GOLD)

# ── "JEREMY AI" — large stacked Impact, gold, left-aligned ────────────────
f_jeremy = ImageFont.truetype(FONT_IMPACT, 178)
f_x      = ImageFont.truetype(FONT_BLACK,  52)
f_gemma4 = ImageFont.truetype(FONT_IMPACT, 96)
f_tag    = ImageFont.truetype(FONT_DIN,    36)
f_sub    = ImageFont.truetype(FONT_SERIF,  34)
f_stat   = ImageFont.truetype(FONT_IMPACT, 80)
f_stat_l = ImageFont.truetype(FONT_DIN,    28)

margin_l = 90

# Shadow helper
def draw_text_shadow(draw, pos, text, font, fill, shadow_color=(0,0,0), shadow_offset=4, layers=3):
    x, y = pos
    for i in range(layers, 0, -1):
        a = int(180 * (i/layers))
        draw.text((x + shadow_offset*i//layers, y + shadow_offset*i//layers),
                  text, font=font, fill=shadow_color)
    draw.text(pos, text, font=font, fill=fill)

# "JEREMY" — huge
draw_text_shadow(draw, (margin_l, 60), "JEREMY", f_jeremy, GOLD, shadow_offset=6, layers=4)
# "AI" — slightly smaller, paper color offset right
bb = draw.textbbox((0,0), "JEREMY", font=f_jeremy)
jeremy_w = bb[2] - bb[0]
ai_x = margin_l + jeremy_w + 20
f_ai = ImageFont.truetype(FONT_IMPACT, 178)
draw_text_shadow(draw, (ai_x, 60), "AI", f_ai, PAPER, shadow_offset=6, layers=4)

# "×" separator
bb2 = draw.textbbox((0,0), "AI", font=f_ai)
ai_w = bb2[2] - bb2[0]
x_pos = margin_l
draw_text_shadow(draw, (x_pos, 248), "×", f_x, GEMMA, shadow_offset=3)

# "GEMMA 4" — gold, below
bb3 = draw.textbbox((0,0), "×", font=f_x)
x_w = bb3[2] - bb3[0]
draw_text_shadow(draw, (x_pos + x_w + 18, 240), "GEMMA 4", f_gemma4, GOLD_H, shadow_offset=5, layers=3)

# ── TAGLINE ────────────────────────────────────────────────────────────────
tag_y = 358
tag_text = "ACCESS TO JUSTICE FOR ALL"
bb_tag = draw.textbbox((0,0), tag_text, font=f_tag)
tag_w = bb_tag[2] - bb_tag[0]
# pill background
pad_x, pad_y = 24, 10
draw.rounded_rectangle(
    [margin_l - pad_x, tag_y - pad_y,
     margin_l + tag_w + pad_x, tag_y + (bb_tag[3]-bb_tag[1]) + pad_y],
    radius=6, fill=GEMMA
)
draw.text((margin_l, tag_y), tag_text, font=f_tag, fill=INK)

# ── STAT BLOCK — right side ────────────────────────────────────────────────
stat_x = 760
stat_y = 90

stats = [
    ("80%",        "of Americans\ncannot afford a lawyer"),
    ("$5,355",     "average cost of\na NY divorce"),
    ("$0",         "Jeremy AI\nforever"),
]

for number, label in stats:
    # number
    draw_text_shadow(draw, (stat_x, stat_y), number, f_stat, GOLD, shadow_offset=4, layers=3)
    bb_n = draw.textbbox((0,0), number, font=f_stat)
    n_h  = bb_n[3] - bb_n[1]
    # label
    for i, line in enumerate(label.split("\n")):
        draw.text((stat_x, stat_y + n_h + 4 + i*32), line, font=f_stat_l, fill=PAPER)
    # separator line
    line_y = stat_y + n_h + 4 + 2*32 + 14
    draw.rectangle([stat_x, line_y, stat_x + 380, line_y + 2], fill=GOLD)
    stat_y = line_y + 22

# ── BOTTOM — prosenetwork.org/demo ────────────────────────────────────────
f_url = ImageFont.truetype(FONT_DIN, 30)
url_text = "prosenetwork.org/demo  ·  Open Weights  ·  Free Forever"
draw.text((margin_l, H - 70), url_text, font=f_url, fill=(180, 170, 150))

# ── HACKATHON BADGE — top right ───────────────────────────────────────────
f_badge = ImageFont.truetype(FONT_DIN, 22)
badge_lines = ["GOOGLE", "GEMMA 4", "HACKATHON"]
bx, by = W - 220, 24
for i, bl in enumerate(badge_lines):
    bb = draw.textbbox((0,0), bl, font=f_badge)
    bw = bb[2] - bb[0]
    cx = bx + (180 - bw) // 2
    color = GEMMA if i == 1 else GOLD
    draw.text((cx, by + i*28), bl, font=f_badge, fill=color)

# ── SAVE ──────────────────────────────────────────────────────────────────
img.save(OUT, "JPEG", quality=97)
print(f"Saved: {OUT}")
print(f"Size: {W}×{H}")
