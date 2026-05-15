#!/usr/bin/env python3
"""Gemma 4 pitch video — text overlay + audio bed post-processor (v2 redesign).

Usage:
    python3 gemma4_overlay.py
    python3 gemma4_overlay.py --input ~/Desktop/test_v2.mp4 --out ~/Desktop/gemma4_v2.mp4
    python3 gemma4_overlay.py --no-bed
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── Config ───────────────────────────────────────────────────────────────────
RES_W, RES_H = 1920, 1080
FPS          = 30
LUFS_SPEAKER = -16
LUFS_BED     = -22

PALETTE = {
    "ink":      "#0a1124",
    "ink_soft": "#131c38",
    "paper":    "#f6efe1",
    "gold":     "#d4a13a",
    "gold_hi":  "#f0c768",
    "gemma":    "#5bb3f8",   # Gemma accent blue
    "white":    "#ffffff",
}

FONT_SERIF   = "/System/Library/Fonts/Supplemental/Iowan Old Style.ttc"
FONT_SANS    = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_BLACK   = "/System/Library/Fonts/Supplemental/Arial Black.ttf"
FONT_IMPACT  = "/System/Library/Fonts/Supplemental/Impact.ttf"
FONT_DIN     = "/System/Library/Fonts/Supplemental/DIN Alternate Bold.ttf"
FONT_MONO    = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"

PROJECT_ROOT = Path(__file__).parent
DEFAULT_INPUT = Path.home() / "Desktop" / "test_v2.mp4"
DEFAULT_BED   = PROJECT_ROOT / "video" / "Pieces - After The Fall.mp3"
DEFAULT_OUT   = Path.home() / "Desktop" / "gemma4_v2.mp4"

# ── Subtitle cues (Whisper-timed to test_v2.mp4) ─────────────────────────────
# Demo section (22–40s) intentionally clear — badge only
SUBTITLE_CUES: list[tuple[float, float, str]] = [
    (4.3,  9.0,  "80% of low-income Americans\ncannot afford a lawyer."),
    (11.3, 16.5, "When you're handed a lawsuit —\nyou have 20 days to respond."),
    (19.4, 21.6, "Jeremy gives you the steps."),
    (40.7, 44.2, "Gemma 4 handles the language.\nThe rule engine handles the law."),
    (44.4, 48.0, "The model never decides.\nThe rules decide."),
    (48.4, 53.0, "This is already in active litigation.\nMcDaniel v. City of New York."),
    (57.0, 65.8, "It works in the justice gap.\nAnyone with an internet connection."),
    (66.0, 71.2, "Jeremy was built for everyone\nwho could not afford Jeremy before."),
]

# LIVE DEMO badge — gold pill over demo section
LIVE_DEMO_CUE: tuple[float, float] = (22.0, 40.0)

# Kinetic word drops — fly in from right, fade out
KINETIC_CUES: list[tuple[float, float, str]] = [
    (53.4, 54.5, "OPEN WEIGHTS."),
    (54.7, 55.7, "SELF-HOSTABLE."),
    (56.0, 56.6, "FREE."),
    (56.6, 57.2, "FOREVER."),
]

# Opening punch — "80%" flies in huge before the title settles
# (start, end, text)
OPENING_PUNCH: tuple[float, float, str] = (0.3, 4.2, "80%")

# Floating stat cards — judge-facing facts, Number Lock values
# Each: (start, end, number_text, label_text, side)  side = "left" | "right"
STAT_CARDS: list[tuple[float, float, str, str, str]] = [
    (5.0,  8.8,  "$5,355",   "average legal cost\nfor a NY divorce",    "right"),
    (9.2,  11.1, "46%",      "abandon their cases\ndue to cost",        "left"),
    (46.5, 50.3, "672,500",  "divorces filed\nper year in the US",      "right"),
    (50.5, 52.8, "$20/mo",   "Jeremy vs $5,000\nretainer",              "left"),
]


# ── PIL helpers ───────────────────────────────────────────────────────────────
def hex_to_rgba(hexcolor: str, alpha: float) -> tuple[int, int, int, int]:
    h = hexcolor.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return r, g, b, int(round(alpha * 255))


def _gradient_rect(draw: ImageDraw.Draw, x0: int, y0: int, x1: int, y1: int,
                   color_top: tuple, color_bot: tuple) -> None:
    """Draw a vertical gradient rectangle using horizontal scanlines."""
    h = y1 - y0
    for i in range(h):
        t = i / max(h - 1, 1)
        r = int(color_top[0] + (color_bot[0] - color_top[0]) * t)
        g = int(color_top[1] + (color_bot[1] - color_top[1]) * t)
        b = int(color_top[2] + (color_bot[2] - color_top[2]) * t)
        a = int(color_top[3] + (color_bot[3] - color_top[3]) * t)
        draw.line([(x0, y0 + i), (x1, y0 + i)], fill=(r, g, b, a))


def render_title_card_png(out_path: Path) -> Path:
    """Title card — text only, no background veil (Resolve handles BG fade).
    Text lingers then floats off via ffmpeg drift expression."""
    img  = Image.new("RGBA", (RES_W, RES_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    tmp = Image.new("RGBA", (1, 1))
    d   = ImageDraw.Draw(tmp)

    f_title = ImageFont.truetype(FONT_BLACK, 92)
    f_sub   = ImageFont.truetype(FONT_SERIF, 54)
    f_tag   = ImageFont.truetype(FONT_MONO, 22)

    title = "JEREMY AI × GEMMA 4"
    sub   = "Access to Justice for All"
    tag   = "$0  ·  OPEN WEIGHTS  ·  SELF-HOSTABLE  ·  FOREVER"

    # Measure
    t_bb  = d.textbbox((0, 0), title, font=f_title)
    s_bb  = d.textbbox((0, 0), sub,   font=f_sub)
    g_bb  = d.textbbox((0, 0), tag,   font=f_tag)

    t_h = t_bb[3] - t_bb[1]
    s_h = s_bb[3] - s_bb[1]
    g_h = g_bb[3] - g_bb[1]

    GAP = 22
    BAR = 3
    total_h = BAR + GAP + t_h + GAP + BAR + GAP + s_h + GAP * 2 + g_h
    cy = (RES_H - total_h) // 2

    # Gemma accent bar above title
    bar_w = 600
    bx    = (RES_W - bar_w) // 2
    draw.rectangle((bx, cy, bx + bar_w, cy + BAR),
                   fill=hex_to_rgba(PALETTE["gemma"], 0.9))

    # Title — large gold, heavy shadow
    title_y = cy + BAR + GAP
    tw       = t_bb[2] - t_bb[0]
    tx       = (RES_W - tw) // 2
    for ox, oy in [(6, 6), (4, 4)]:
        draw.text((tx + ox, title_y + oy - t_bb[1]), title, font=f_title,
                  fill=hex_to_rgba(PALETTE["ink"], 0.95))
    draw.text((tx, title_y - t_bb[1]), title, font=f_title,
              fill=hex_to_rgba(PALETTE["gold_hi"], 1.0))

    # Rule below title
    rule_y = title_y + t_h + GAP
    draw.rectangle(((RES_W - 700) // 2, rule_y, (RES_W + 700) // 2, rule_y + BAR),
                   fill=hex_to_rgba(PALETTE["gold"], 0.55))

    # Subtitle — paper serif, subtle backing pill
    sub_y  = rule_y + BAR + GAP
    sw     = s_bb[2] - s_bb[0]
    sx     = (RES_W - sw) // 2
    pad    = 16
    draw.rounded_rectangle((sx - pad, sub_y - pad // 2,
                             sx + sw + pad, sub_y + s_h + pad // 2),
                            radius=8,
                            fill=hex_to_rgba(PALETTE["ink"], 0.45))
    draw.text((sx, sub_y - s_bb[1]), sub, font=f_sub,
              fill=hex_to_rgba(PALETTE["paper"], 0.97))

    # Tag — gemma blue, bottom
    tag_y = sub_y + s_h + GAP * 2
    gw    = g_bb[2] - g_bb[0]
    draw.text(((RES_W - gw) // 2, tag_y - g_bb[1]), tag, font=f_tag,
              fill=hex_to_rgba(PALETTE["gemma"], 0.80))

    img.save(out_path, "PNG")
    return out_path


def render_end_card_png(out_path: Path) -> Path:
    """End card — movement energy. Dynamic layout so nothing overlaps."""
    f_hero = ImageFont.truetype(FONT_BLACK, 96)
    f_url  = ImageFont.truetype(FONT_BLACK, 62)
    f_zero = ImageFont.truetype(FONT_IMPACT, 82)
    f_repo = ImageFont.truetype(FONT_MONO, 23)

    hero      = "JUSTICE FOR ALL."
    url       = "prosenetwork.org/demo"
    zero_line = "$0.  FOREVER.  FOR EVERYONE."
    repo      = "israelburns/jeremy-gemma4  ·  HuggingFace  ·  Open Weights  ·  Self-Hostable"

    # Measure all elements first — no guessing
    tmp  = Image.new("RGBA", (1, 1))
    d    = ImageDraw.Draw(tmp)
    h_bb = d.textbbox((0, 0), hero,      font=f_hero)
    u_bb = d.textbbox((0, 0), url,       font=f_url)
    z_bb = d.textbbox((0, 0), zero_line, font=f_zero)
    r_bb = d.textbbox((0, 0), repo,      font=f_repo)

    hero_h = h_bb[3] - h_bb[1]
    url_h  = u_bb[3] - u_bb[1]
    zero_h = z_bb[3] - z_bb[1]
    repo_h = r_bb[3] - r_bb[1]

    RULE   = 3
    G1     = 28   # hero → rule
    G2     = 28   # rule → url
    G3     = 52   # url → zero
    G4     = 40   # zero → rule
    G5     = 22   # rule → repo

    total = hero_h + G1 + RULE + G2 + url_h + G3 + zero_h + G4 + RULE + G5 + repo_h
    start_y = (RES_H - total) // 2

    # Compute y for each element
    hero_y  = start_y
    rule1_y = hero_y  + hero_h  + G1
    url_y   = rule1_y + RULE    + G2
    zero_y  = url_y   + url_h   + G3
    rule2_y = zero_y  + zero_h  + G4
    repo_y  = rule2_y + RULE    + G5

    img  = Image.new("RGBA", (RES_W, RES_H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background gradient — ink, top lighter, bottom nearly opaque
    ink_rgb = tuple(int(PALETTE["ink"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    _gradient_rect(draw, 0, 0, RES_W, RES_H,
                   color_top=(*ink_rgb, 190),
                   color_bot=(*ink_rgb, 250))

    # Gemma accent stripe — left edge
    draw.rectangle((0, 0, 6, RES_H), fill=hex_to_rgba(PALETTE["gemma"], 0.7))
    draw.rectangle((RES_W - 6, 0, RES_W, RES_H), fill=hex_to_rgba(PALETTE["gemma"], 0.7))

    # Hero — "JUSTICE FOR ALL." white, heavy shadow
    hw = h_bb[2] - h_bb[0]
    hx = (RES_W - hw) // 2
    draw.text((hx + 6, hero_y - h_bb[1] + 6), hero, font=f_hero,
              fill=hex_to_rgba(PALETTE["ink"], 0.95))
    draw.text((hx, hero_y - h_bb[1]), hero, font=f_hero,
              fill=hex_to_rgba(PALETTE["white"], 1.0))

    # Rule 1 — full gold
    rule_w = min(hw + 200, RES_W - 160)
    draw.rectangle(((RES_W - rule_w) // 2, rule1_y,
                    (RES_W + rule_w) // 2, rule1_y + RULE),
                   fill=hex_to_rgba(PALETTE["gold"], 0.85))

    # URL — "prosenetwork.org/demo" — gold, big, clear
    uw = u_bb[2] - u_bb[0]
    ux = (RES_W - uw) // 2
    draw.text((ux + 4, url_y - u_bb[1] + 4), url, font=f_url,
              fill=hex_to_rgba(PALETTE["ink"], 0.9))
    draw.text((ux, url_y - u_bb[1]), url, font=f_url,
              fill=hex_to_rgba(PALETTE["gold_hi"], 1.0))

    # $0 FOREVER — Impact, paper white, gemma pill behind
    zw   = z_bb[2] - z_bb[0]
    zx   = (RES_W - zw) // 2
    pad  = 24
    draw.rounded_rectangle(
        (zx - pad, zero_y - pad // 2,
         zx + zw + pad, zero_y + zero_h + pad // 2),
        radius=16,
        fill=hex_to_rgba(PALETTE["gemma"], 0.16)
    )
    # Thin border on pill
    draw.rounded_rectangle(
        (zx - pad, zero_y - pad // 2,
         zx + zw + pad, zero_y + zero_h + pad // 2),
        radius=16,
        outline=hex_to_rgba(PALETTE["gemma"], 0.45),
        width=2
    )
    draw.text((zx, zero_y - z_bb[1]), zero_line, font=f_zero,
              fill=hex_to_rgba(PALETTE["paper"], 1.0))

    # Rule 2 — thin gold
    draw.rectangle(((RES_W - 700) // 2, rule2_y,
                    (RES_W + 700) // 2, rule2_y + RULE),
                   fill=hex_to_rgba(PALETTE["gold"], 0.38))

    # Repo — small, gold
    rw = r_bb[2] - r_bb[0]
    draw.text(((RES_W - rw) // 2, repo_y - r_bb[1]), repo, font=f_repo,
              fill=hex_to_rgba(PALETTE["gold"], 0.68))

    img.save(out_path, "PNG")
    return out_path


def render_opening_punch_png(text: str, out_path: Path) -> tuple[Path, int, int]:
    """Opening kinetic punch — "80%" huge Impact gold, no background, full impact."""
    font = ImageFont.truetype(FONT_IMPACT, 240)
    tmp  = Image.new("RGBA", (1, 1))
    d    = ImageDraw.Draw(tmp)
    bb   = d.textbbox((0, 0), text, font=font)
    pad_x, pad_top, pad_bot = 60, 20, 70
    w    = (bb[2] - bb[0]) + pad_x * 2
    h    = (bb[3] - bb[1]) + pad_top + pad_bot
    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    x_off = pad_x - bb[0]
    y_off = pad_top - bb[1]
    # Multi-layer shadow for depth
    for ox, oy in [(8, 8), (5, 5), (3, 3)]:
        draw.text((x_off + ox, y_off + oy), text, font=font,
                  fill=hex_to_rgba(PALETTE["ink"], 0.85))
    # Gold fill, heavy stroke
    draw.text((x_off, y_off), text, font=font,
              fill=hex_to_rgba(PALETTE["gold_hi"], 1.0),
              stroke_width=8, stroke_fill=hex_to_rgba(PALETTE["ink"], 1.0))
    img.save(out_path, "PNG")
    return out_path, w, h


def render_subtitle_png(text: str, out_path: Path, font_size: int = 42) -> tuple[Path, int, int]:
    """Subtitle cue: ink background box, paper serif text."""
    font = ImageFont.truetype(FONT_SERIF, font_size)
    lines: list[str] = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, width=52) or [""])
    lines = lines[:3]

    pad_x, pad_y, gap = 36, 22, 14
    tmp  = Image.new("RGBA", (1, 1))
    d    = ImageDraw.Draw(tmp)
    dims    = [d.textbbox((0, 0), L, font=font) for L in lines]
    widths  = [b[2] - b[0] for b in dims]
    heights = [b[3] - b[1] for b in dims]
    text_w  = max(widths) if widths else 0
    text_h  = sum(heights) + gap * (len(lines) - 1)
    box_w, box_h = text_w + pad_x * 2, text_h + pad_y * 2

    img  = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, box_w, box_h), fill=hex_to_rgba(PALETTE["ink"], 0.72))

    # Left accent bar
    draw.rectangle((0, 0, 4, box_h), fill=hex_to_rgba(PALETTE["gold"], 0.9))

    y = pad_y
    for L, lw, lh in zip(lines, widths, heights):
        x = (box_w - lw) // 2
        draw.text((x + 2, y + 2), L, font=font, fill=hex_to_rgba(PALETTE["ink"], 0.9))
        draw.text((x, y), L, font=font, fill=hex_to_rgba(PALETTE["paper"], 1.0),
                  stroke_width=1, stroke_fill=hex_to_rgba(PALETTE["ink"], 0.8))
        y += lh + gap
    img.save(out_path, "PNG")
    return out_path, box_w, box_h


def render_kinetic_png(text: str, out_path: Path) -> tuple[Path, int, int]:
    """Kinetic word drop — Impact font, full and strong, fixed bottom padding."""
    font = ImageFont.truetype(FONT_IMPACT, 128)
    tmp  = Image.new("RGBA", (1, 1))
    d    = ImageDraw.Draw(tmp)
    bb   = d.textbbox((0, 0), text, font=font)
    # Generous padding so descenders never clip
    pad_x, pad_top, pad_bot = 48, 20, 52
    w = (bb[2] - bb[0]) + pad_x * 2
    h = (bb[3] - bb[1]) + pad_top + pad_bot

    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    x_off = pad_x - bb[0]
    y_off = pad_top - bb[1]
    # Dark shadow
    draw.text((x_off + 5, y_off + 5), text, font=font,
              fill=hex_to_rgba(PALETTE["ink"], 0.95))
    # Gold fill, strong ink stroke
    draw.text((x_off, y_off), text, font=font,
              fill=hex_to_rgba(PALETTE["gold_hi"], 1.0),
              stroke_width=5, stroke_fill=hex_to_rgba(PALETTE["ink"], 1.0))
    img.save(out_path, "PNG")
    return out_path, w, h


def render_stat_card_png(number: str, label: str, out_path: Path,
                         side: str = "left") -> tuple[Path, int, int]:
    """Floating stat card — 2x size, editorial feel, professional spacing."""
    f_num   = ImageFont.truetype(FONT_IMPACT, 148)
    f_label = ImageFont.truetype(FONT_DIN, 40)
    f_unit  = ImageFont.truetype(FONT_MONO, 26)

    lines = label.split("\n")
    tmp   = Image.new("RGBA", (1, 1))
    d     = ImageDraw.Draw(tmp)

    bb_num    = d.textbbox((0, 0), number, font=f_num)
    num_w     = bb_num[2] - bb_num[0]
    num_h     = bb_num[3] - bb_num[1] + 24  # extra for Impact descenders

    label_dims = [d.textbbox((0, 0), L, font=f_label) for L in lines]
    label_ws   = [b[2] - b[0] for b in label_dims]
    label_hs   = [b[3] - b[1] for b in label_dims]
    label_w    = max(label_ws) if label_ws else 0
    label_h    = sum(label_hs) + 10 * (len(lines) - 1)

    min_w     = 340
    content_w = max(num_w, label_w, min_w)
    content_h = num_h + 20 + label_h

    pad_x, pad_y = 52, 38
    box_w = content_w + pad_x * 2
    box_h = content_h + pad_y * 2

    img  = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dark backing — rounded, semi-transparent
    draw.rounded_rectangle((0, 0, box_w, box_h), radius=18,
                            fill=hex_to_rgba(PALETTE["ink"], 0.86))

    # Thick accent bar (8px) — left for left cards, right for right
    accent = hex_to_rgba(PALETTE["gemma"], 1.0)
    bar_h_inner = box_h - 36
    bar_y_start = 18
    if side == "right":
        draw.rounded_rectangle((box_w - 8, bar_y_start, box_w, bar_y_start + bar_h_inner),
                                radius=4, fill=accent)
    else:
        draw.rounded_rectangle((0, bar_y_start, 8, bar_y_start + bar_h_inner),
                                radius=4, fill=accent)

    # Thin gold rule below number (separates number from label)
    rule_y = pad_y + num_h + 8
    draw.rectangle((pad_x, rule_y, pad_x + content_w, rule_y + 2),
                   fill=hex_to_rgba(PALETTE["gold"], 0.5))

    # Number — gold_hi, Impact
    nx = pad_x + (content_w - num_w) // 2 - bb_num[0]
    ny = pad_y - bb_num[1]
    draw.text((nx + 4, ny + 4), number, font=f_num,
              fill=hex_to_rgba(PALETTE["ink"], 0.9))
    draw.text((nx, ny), number, font=f_num,
              fill=hex_to_rgba(PALETTE["gold_hi"], 1.0))

    # Labels — paper, DIN
    y = pad_y + num_h + 20
    for L, lw, lh in zip(lines, label_ws, label_hs):
        lx = pad_x + (content_w - lw) // 2
        draw.text((lx, y), L, font=f_label,
                  fill=hex_to_rgba(PALETTE["paper"], 0.88))
        y += lh + 10

    img.save(out_path, "PNG")
    return out_path, box_w, box_h


def render_live_demo_badge_png(out_path: Path) -> tuple[Path, int, int]:
    """Gold pill badge: ● LIVE DEMO — top-right corner."""
    font = ImageFont.truetype(FONT_MONO, 28)
    text = "● LIVE DEMO"
    tmp  = Image.new("RGBA", (1, 1))
    d    = ImageDraw.Draw(tmp)
    bb   = d.textbbox((0, 0), text, font=font)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    pad_x, pad_y = 24, 12
    w, h = tw + pad_x * 2, th + pad_y * 2
    img  = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, w, h), radius=h // 2,
                            fill=hex_to_rgba(PALETTE["gold"], 0.88))
    draw.text((pad_x, pad_y - 2), text, font=font,
              fill=hex_to_rgba(PALETTE["ink"], 1.0))
    img.save(out_path, "PNG")
    return out_path, w, h


# ── Grade ─────────────────────────────────────────────────────────────────────
def grade_filter(label: str) -> str:
    return (f"[{label}]eq=saturation=1.08:contrast=1.04:gamma=1.02,"
            f"noise=alls=3:allf=t+u[v_graded]")


# ── Audio bed ─────────────────────────────────────────────────────────────────
def mix_audio_bed(video: Path, bed: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(video),
        "-stream_loop", "-1", "-i", str(bed),
        "-filter_complex",
        f"[0:a]aformat=channel_layouts=stereo:sample_rates=48000,"
        f"loudnorm=I={LUFS_SPEAKER}:TP=-1.5:LRA=11[spk];"
        f"[1:a]aformat=channel_layouts=stereo:sample_rates=48000,"
        f"loudnorm=I={LUFS_BED}:TP=-2:LRA=7[bed];"
        f"[spk][bed]amix=inputs=2:duration=first:dropout_transition=3[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest", "-movflags", "+faststart",
        str(dst),
    ]
    print(f"[audio-bed] mixing {bed.name} → {dst.name}")
    subprocess.run(cmd, check=True)


# ── Main overlay render ───────────────────────────────────────────────────────
def build_overlay_cmd(
    src: Path,
    work: Path,
    punch_png: Path,
    punch_w: int,
    punch_h: int,
    title_png: Path,
    end_png: Path,
    badge_png: Path,
    badge_w: int,
    badge_h: int,
    sub_pngs: list[tuple[Path, int, int, float, float]],
    kinetic_pngs: list[tuple[Path, int, int, float, float]],
    stat_pngs: list[tuple[Path, int, int, float, float, str]],
    out_path: Path,
    video_duration: float,
) -> list[str]:
    title_start, title_end = 0.0, 7.5
    float_start            = 4.8
    end_start              = video_duration - 6.5

    p_start, p_end = OPENING_PUNCH[0], OPENING_PUNCH[1]

    inputs = ["-y", "-hide_banner", "-loglevel", "warning", "-i", str(src)]
    idx    = 1

    inputs += ["-loop", "1", "-t", f"{video_duration:.3f}", "-i", str(punch_png)]
    punch_idx = idx; idx += 1

    inputs += ["-loop", "1", "-t", f"{video_duration:.3f}", "-i", str(title_png)]
    title_idx = idx; idx += 1

    inputs += ["-loop", "1", "-t", f"{video_duration:.3f}", "-i", str(end_png)]
    end_idx = idx; idx += 1

    inputs += ["-loop", "1", "-t", f"{video_duration:.3f}", "-i", str(badge_png)]
    badge_idx = idx; idx += 1

    sub_indices: list[int] = []
    for (p, *_) in sub_pngs:
        inputs += ["-loop", "1", "-t", f"{video_duration:.3f}", "-i", str(p)]
        sub_indices.append(idx); idx += 1

    kin_indices: list[int] = []
    for (p, *_) in kinetic_pngs:
        inputs += ["-loop", "1", "-t", f"{video_duration:.3f}", "-i", str(p)]
        kin_indices.append(idx); idx += 1

    stat_indices: list[int] = []
    for (p, *_) in stat_pngs:
        inputs += ["-loop", "1", "-t", f"{video_duration:.3f}", "-i", str(p)]
        stat_indices.append(idx); idx += 1

    fc: list[str] = []
    cur = "0:v"

    # ── Opening punch "80%" — flies in from left, fades out as title holds ────
    fc.append(
        f"[{punch_idx}:v]"
        f"fade=t=in:st={p_start:.3f}:d=0.2:alpha=1,"
        f"fade=t=out:st={p_end - 1.2:.3f}:d=1.2:alpha=1"
        f"[punch_faded]"
    )
    # Fly in from left: x goes from -w to centered over 0.3s
    punch_fly = (
        f"if(lt(t,{p_start:.3f}+0.3),"
        f"0-w+(w+(W-w)/2)*((t-{p_start:.3f})/0.3),"
        f"(W-w)/2)"
    )
    punch_y = 72  # near top, clear of title text
    fc.append(
        f"[{cur}][punch_faded]"
        f"overlay=x='{punch_fly}':y={punch_y}:format=auto:"
        f"enable='between(t,{p_start:.3f},{p_end:.3f})'[after_punch]"
    )
    cur = "after_punch"

    # ── Title card: fade in 1.2s, hold, then fade out + float up ─────────────
    fade_out_dur = title_end - float_start
    fc.append(
        f"[{title_idx}:v]"
        f"fade=t=in:st=0:d=1.2:alpha=1,"
        f"fade=t=out:st={float_start:.3f}:d={fade_out_dur:.3f}:alpha=1"
        f"[title_faded]"
    )
    drift_y = f"if(gt(t,{float_start:.3f}),-(t-{float_start:.3f})*28,0)"
    fc.append(
        f"[{cur}][title_faded]"
        f"overlay=x=0:y='{drift_y}':format=auto:"
        f"enable='between(t,{title_start:.3f},{title_end:.3f})'[after_title]"
    )
    cur = "after_title"

    # ── End card: fade in 1s ──────────────────────────────────────────────────
    fc.append(
        f"[{end_idx}:v]"
        f"fade=t=in:st={end_start:.3f}:d=1.0:alpha=1"
        f"[end_faded]"
    )
    fc.append(
        f"[{cur}][end_faded]"
        f"overlay=0:0:format=auto:"
        f"enable='between(t,{end_start:.3f},{video_duration:.3f})'[after_end]"
    )
    cur = "after_end"

    # ── Live demo badge ───────────────────────────────────────────────────────
    badge_x = RES_W - badge_w - 48
    badge_y = 48
    fc.append(
        f"[{cur}][{badge_idx}:v]"
        f"overlay={badge_x}:{badge_y}:format=auto:"
        f"enable='between(t,{LIVE_DEMO_CUE[0]:.3f},{LIVE_DEMO_CUE[1]:.3f})'[after_badge]"
    )
    cur = "after_badge"

    # ── Subtitle cues (fade in 0.2s) ─────────────────────────────────────────
    for i, (in_idx, (p, w, h, start, end)) in enumerate(zip(sub_indices, sub_pngs)):
        label = f"sub_faded_{i:02d}"
        fc.append(
            f"[{in_idx}:v]"
            f"fade=t=in:st={start:.3f}:d=0.2:alpha=1,"
            f"fade=t=out:st={end - 0.2:.3f}:d=0.2:alpha=1"
            f"[{label}]"
        )
        x = (RES_W - w) // 2
        y = RES_H - h - 90
        nxt = f"sub_{i:02d}"
        fc.append(
            f"[{cur}][{label}]"
            f"overlay={x}:{y}:format=auto:"
            f"enable='between(t,{start:.3f},{end:.3f})'[{nxt}]"
        )
        cur = nxt

    # ── Kinetic cues: fly in from right, fade in 0.15s, fade out 0.2s ────────
    for i, (in_idx, (p, w, h, start, end)) in enumerate(zip(kin_indices, kinetic_pngs)):
        label = f"kin_faded_{i:02d}"
        fc.append(
            f"[{in_idx}:v]"
            f"fade=t=in:st={start:.3f}:d=0.15:alpha=1,"
            f"fade=t=out:st={end - 0.2:.3f}:d=0.2:alpha=1"
            f"[{label}]"
        )
        target_y = (RES_H - h) // 2 + 60
        # Fly in from right: x goes from W → centered over 0.25s
        fly_expr = (
            f"if(lt(t,{start:.3f}+0.25),"
            f"W-(W-(W-w)/2)*((t-{start:.3f})/0.25),"
            f"(W-w)/2)"
        )
        nxt = f"kin_{i:02d}"
        fc.append(
            f"[{cur}][{label}]"
            f"overlay=x='{fly_expr}':y={target_y}:format=auto:"
            f"enable='between(t,{start:.3f},{end:.3f})'[{nxt}]"
        )
        cur = nxt

    # ── Stat cards: fly in from side, fade in/out ─────────────────────────────
    for i, (in_idx, (p, w, h, start, end, side)) in enumerate(zip(stat_indices, stat_pngs)):
        label = f"stat_faded_{i:02d}"
        fc.append(
            f"[{in_idx}:v]"
            f"fade=t=in:st={start:.3f}:d=0.3:alpha=1,"
            f"fade=t=out:st={end - 0.3:.3f}:d=0.3:alpha=1"
            f"[{label}]"
        )
        margin = 80
        target_y = RES_H // 3
        if side == "right":
            target_x = RES_W - w - margin
            # Fly from right edge
            fly_x = (
                f"if(lt(t,{start:.3f}+0.3),"
                f"W-({RES_W - target_x})*((t-{start:.3f})/0.3),"
                f"{target_x})"
            )
        else:
            target_x = margin
            # Fly from left edge
            fly_x = (
                f"if(lt(t,{start:.3f}+0.3),"
                f"{target_x}-{target_x + w}*((t-{start:.3f})/0.3)+{target_x + w}*((t-{start:.3f})/0.3)-{w}*(1-((t-{start:.3f})/0.3)),"
                f"{target_x})"
            )
            # Simpler left fly-in: start off left edge, slide right
            fly_x = (
                f"if(lt(t,{start:.3f}+0.3),"
                f"0-w+(w+{target_x})*((t-{start:.3f})/0.3),"
                f"{target_x})"
            )
        nxt = f"stat_{i:02d}"
        fc.append(
            f"[{cur}][{label}]"
            f"overlay=x='{fly_x}':y={target_y}:format=auto:"
            f"enable='between(t,{start:.3f},{end:.3f})'[{nxt}]"
        )
        cur = nxt

    # ── Grade ─────────────────────────────────────────────────────────────────
    fc.append(grade_filter(cur))

    # ── Audio loudnorm ────────────────────────────────────────────────────────
    fc.append(
        f"[0:a]aformat=channel_layouts=stereo:sample_rates=48000,"
        f"loudnorm=I={LUFS_SPEAKER}:TP=-1.5:LRA=11[a_out]"
    )

    output = [
        "-filter_complex", ";".join(fc),
        "-map", "[v_graded]", "-map", "[a_out]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart",
        str(out_path),
    ]
    return ["ffmpeg"] + inputs + output


def probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True
    )
    return float(r.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description="Gemma 4 pitch video overlay renderer")
    ap.add_argument("--input",    type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--bed",      type=Path, default=DEFAULT_BED)
    ap.add_argument("--out",      type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-bed",   action="store_true")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"[!] input not found: {args.input}")
    if not args.no_bed and not args.bed.exists():
        sys.exit(f"[!] audio bed not found: {args.bed}")

    duration = probe_duration(args.input)
    print(f"[info] input: {args.input.name}  duration: {duration:.1f}s  fps: {FPS}")

    work = Path(tempfile.mkdtemp(prefix="gemma4_overlay_"))
    print(f"[work] {work}")

    try:
        print("[render] opening punch...")
        punch_png, punch_w, punch_h = render_opening_punch_png(
            OPENING_PUNCH[2], work / "opening_punch.png"
        )

        print("[render] title card...")
        title_png = render_title_card_png(work / "title_card.png")

        print("[render] end card...")
        end_png = render_end_card_png(work / "end_card.png")

        print("[render] subtitle cues...")
        sub_pngs: list[tuple[Path, int, int, float, float]] = []
        for i, (start, end, text) in enumerate(SUBTITLE_CUES):
            p, w, h = render_subtitle_png(text, work / f"sub_{i:02d}.png")
            sub_pngs.append((p, w, h, start, end))

        print("[render] kinetic cues...")
        kin_pngs: list[tuple[Path, int, int, float, float]] = []
        for i, (start, end, text) in enumerate(KINETIC_CUES):
            p, w, h = render_kinetic_png(text, work / f"kin_{i:02d}.png")
            kin_pngs.append((p, w, h, start, end))

        print("[render] stat cards...")
        stat_pngs: list[tuple[Path, int, int, float, float, str]] = []
        for i, (start, end, num, lbl, side) in enumerate(STAT_CARDS):
            p, w, h = render_stat_card_png(num, lbl, work / f"stat_{i:02d}.png", side)
            stat_pngs.append((p, w, h, start, end, side))

        print("[render] live demo badge...")
        badge_png, badge_w, badge_h = render_live_demo_badge_png(work / "live_demo_badge.png")

        overlaid = work / "overlaid.mp4"
        cmd = build_overlay_cmd(
            args.input, work,
            punch_png, punch_w, punch_h,
            title_png, end_png,
            badge_png, badge_w, badge_h,
            sub_pngs, kin_pngs, stat_pngs, overlaid, duration
        )
        n_overlays = len(sub_pngs) + len(kin_pngs) + len(stat_pngs)
        print(f"\n[ffmpeg] {n_overlays} overlays + punch + title + end card...")
        subprocess.run(cmd, check=True)

        if args.no_bed:
            shutil.move(str(overlaid), str(args.out))
        else:
            mix_audio_bed(overlaid, args.bed, args.out)

        print(f"\n[done] → {args.out}")

    finally:
        if not args.keep_tmp:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"[keep-tmp] {work}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
