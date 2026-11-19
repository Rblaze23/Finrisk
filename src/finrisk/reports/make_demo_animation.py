"""Generates an automated, professional demo animation (docs/demo.gif and docs/demo.html).

Produces an executive 30-second animated product showcase:
  - Scene 1: Platform Splash & Architecture
  - Scene 2: Real-Time Scorecard Scoring & KPIs
  - Scene 3: Autonomous Agentic Analyst (Prime Approval & Policy Gates)
  - Scene 4: Distressed Borrower & FCRA Adverse Action Notice
  - Scene 5: Counterfactual Optimization (Path-to-Approval)
  - Scene 6: Fed SR 11-7 Out-of-Time Decile Calibration
  - Scene 7: Macroeconomic Stress Testing & Expected Credit Loss
  - Scene 8: Executive Vector PDF Report Export
  - Scene 9: Production Outro
"""
from __future__ import annotations

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"

WIDTH, HEIGHT = 960, 540

# Fonts
FONT_TITLE = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 36)
FONT_SUB = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf", 20)
FONT_BODY = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 16)
FONT_BADGE = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 14)
FONT_BANNER = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 16)
FONT_BANNER_SUB = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", 13)


def draw_rounded_rect(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    x1, y1, x2, y2 = box
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=width)


def make_bottom_banner(base_im: Image.Image, step_num: int, title: str, subtitle: str) -> Image.Image:
    im = base_im.copy()
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    # Gradient / semi-transparent banner bar
    bar_y = HEIGHT - 76
    odraw.rectangle([0, bar_y, WIDTH, HEIGHT], fill=(11, 19, 43, 235))
    odraw.line([(0, bar_y), (WIDTH, bar_y)], fill=(37, 99, 235, 255), width=3)

    # Step pill
    pill_w = 88
    odraw.rounded_rectangle([20, bar_y + 16, 20 + pill_w, bar_y + 54], radius=6, fill=(37, 99, 235, 255))
    odraw.text((32, bar_y + 26), f"STEP {step_num}", font=FONT_BADGE, fill=(255, 255, 255))

    # Text
    odraw.text((124, bar_y + 16), title, font=FONT_BANNER, fill=(255, 255, 255))
    odraw.text((124, bar_y + 42), subtitle, font=FONT_BANNER_SUB, fill=(148, 163, 184))

    # Progress bar indicator
    progress_w = int((step_num / 7.0) * (WIDTH - 40))
    odraw.line([(20, HEIGHT - 8), (20 + progress_w, HEIGHT - 8)], fill=(16, 185, 129, 255), width=3)

    im = im.convert("RGBA")
    im = Image.alpha_composite(im, overlay)
    return im.convert("RGB")


def make_splash_screen() -> list[Image.Image]:
    frames = []
    for f in range(25):  # ~2.5 seconds
        im = Image.new("RGB", (WIDTH, HEIGHT), (10, 15, 29))
        draw = ImageDraw.Draw(im)

        # Subtle background grid & card
        draw_rounded_rect(draw, [40, 40, WIDTH - 40, HEIGHT - 40], radius=16,
                          fill=(15, 23, 42), outline=(30, 41, 59), width=2)

        # Glowing accent bar
        draw.line([(80, 100), (80, HEIGHT - 100)], fill=(37, 99, 235), width=4)

        # Title & Subtitle
        draw.text((105, 110), "FinRisk Lab 🏦", font=FONT_TITLE, fill=(255, 255, 255))
        draw.text((105, 165), "Autonomous Credit Risk Intelligence & Model Risk Governance", font=FONT_SUB, fill=(56, 189, 248))
        draw.text((105, 205), "Production-grade credit underwriting and regulatory validation (Fed SR 11-7 / EBA GL/2017/16)",
                  font=FONT_BODY, fill=(148, 163, 184))

        # Badges row
        badges = [
            ("🟢 Fed SR 11-7 Validation", (22, 101, 52), (187, 247, 208)),
            ("🧑‍💼 Autonomous Credit Analyst", (88, 28, 135), (233, 213, 255)),
            ("🛡️ Zero-Hallucination Guard", (30, 58, 138), (191, 219, 254)),
            ("📈 Macro Stress Testing", (124, 45, 18), (254, 215, 170)),
        ]
        bx = 105
        for text, bg, fg in badges:
            w = len(text) * 9 + 20
            draw_rounded_rect(draw, [bx, 260, bx + w, 295], radius=8, fill=bg)
            draw.text((bx + 10, 270), text, font=FONT_BADGE, fill=fg)
            bx += w + 12

        # 4 Metric Highlights
        cards = [
            ("589,488", "Resolved Loans Analyzed"),
            ("0.298", "Out-of-Time Gini Score"),
            ("1.204", "Calibration Ratio (2015 Drift)"),
            ("36 Vintages", "Outcome Lag Simulation"),
        ]
        cx = 105
        for val, desc in cards:
            draw_rounded_rect(draw, [cx, 325, cx + 165, 415], radius=10, fill=(30, 41, 59), outline=(51, 65, 85))
            draw.text((cx + 15, 340), val, font=FONT_SUB, fill=(255, 255, 255))
            draw.text((cx + 15, 375), desc, font=FONT_BANNER_SUB, fill=(148, 163, 184))
            cx += 185

        # Footer tag
        draw.text((105, 445), "Live Interactive Platform: Streamlit • FastAPI • MLflow • PostgreSQL • Docker",
                  font=FONT_BANNER_SUB, fill=(100, 116, 139))

        frames.append(im)
    return frames


def make_scene_from_image(img_path: Path, step_num: int, title: str, subtitle: str,
                          duration_frames: int = 35, zoom: bool = False,
                          cursor_start=None, cursor_end=None) -> list[Image.Image]:
    raw = Image.open(img_path).convert("RGB")
    frames = []

    for i in range(duration_frames):
        # Optional subtle pan or static crop to WIDTH, HEIGHT
        t = i / float(duration_frames)
        im = raw.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

        # If animated cursor provided
        if cursor_start and cursor_end:
            cx = int(cursor_start[0] + (cursor_end[0] - cursor_start[0]) * t)
            cy = int(cursor_start[1] + (cursor_end[1] - cursor_start[1]) * t)
            draw = ImageDraw.Draw(im)
            # Draw cursor pointer and pulse
            draw.polygon([(cx, cy), (cx + 14, cy + 14), (cx + 5, cy + 14), (cx, cy + 20)], fill=(255, 255, 255), outline=(0, 0, 0))
            if i > duration_frames // 2:
                # Click pulse
                pr = (i % 6) * 4
                draw.ellipse([(cx - pr, cy - pr), (cx + pr, cy + pr)], outline=(37, 99, 235), width=2)

        im = make_bottom_banner(im, step_num, title, subtitle)
        frames.append(im)
    return frames


def make_composite_scene(img_path: Path, step_num: int, title: str, subtitle: str,
                         header_text: str, duration_frames: int = 35) -> list[Image.Image]:
    raw = Image.open(img_path).convert("RGB")
    frames = []

    for i in range(duration_frames):
        im = Image.new("RGB", (WIDTH, HEIGHT), (15, 23, 42))
        draw = ImageDraw.Draw(im)

        # Header bar
        draw.rectangle([0, 0, WIDTH, 48], fill=(10, 15, 29))
        draw.text((24, 14), header_text, font=FONT_SUB, fill=(56, 189, 248))

        # Center resized image with shadow border
        iw, ih = raw.size
        scale = min((WIDTH - 80) / iw, (HEIGHT - 160) / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        res = raw.resize((nw, nh), Image.Resampling.LANCZOS)

        ox = (WIDTH - nw) // 2
        oy = 58 + ((HEIGHT - 140 - nh) // 2)
        draw_rounded_rect(draw, [ox - 4, oy - 4, ox + nw + 4, oy + nh + 4], radius=8, fill=(30, 41, 59))
        im.paste(res, (ox, oy))

        im = make_bottom_banner(im, step_num, title, subtitle)
        frames.append(im)
    return frames


def make_outro_screen() -> list[Image.Image]:
    frames = []
    for _ in range(25):
        im = Image.new("RGB", (WIDTH, HEIGHT), (10, 15, 29))
        draw = ImageDraw.Draw(im)

        draw_rounded_rect(draw, [40, 40, WIDTH - 40, HEIGHT - 40], radius=16,
                          fill=(15, 23, 42), outline=(37, 99, 235), width=2)

        draw.text((100, 100), "FinRisk Lab 🏦", font=FONT_TITLE, fill=(255, 255, 255))
        draw.text((100, 155), "Ready for Deployment & Independent Audit", font=FONT_SUB, fill=(56, 189, 248))

        draw.text((100, 210), "• Full Institutional Governance: Fed SR 11-7, EBA, IFRS 9 & FCRA compliant", font=FONT_BODY, fill=(226, 232, 240))
        draw.text((100, 245), "• Autonomous Agentic Credit Analyst with Counterfactual Optimization", font=FONT_BODY, fill=(226, 232, 240))
        draw.text((100, 280), "• GenAI Number Guard ensuring 100% verifiable financial claims", font=FONT_BODY, fill=(226, 232, 240))
        draw.text((100, 315), "• One-Command Docker Architecture: docker compose up -d", font=FONT_BODY, fill=(226, 232, 240))

        # Command box
        draw_rounded_rect(draw, [100, 370, WIDTH - 100, 430], radius=8, fill=(10, 15, 29), outline=(51, 65, 85))
        draw.text((120, 388), "$ git clone https://github.com/Rblaze23/Finrisk && streamlit run streamlit_app.py",
                  font=FONT_BADGE, fill=(16, 185, 129))

        draw.text((100, 450), "Open to opportunities in Quantitative Research, Risk Intelligence, and Machine Learning Engineering.",
                  font=FONT_BANNER_SUB, fill=(148, 163, 184))

        frames.append(im)
    return frames


def build_full_animation() -> Path:
    print("Building animated demo video...")
    all_frames: list[Image.Image] = []

    # 1. Splash Screen
    print("  - Rendering Scene 1: Splash Screen")
    all_frames.extend(make_splash_screen())

    # 2. Main Dashboard & Scoring
    print("  - Rendering Scene 2: Real-Time Scorecard & KPIs")
    all_frames.extend(make_scene_from_image(
        DOCS / "dashboard.png", step_num=1,
        title="Real-Time Risk Scoring & Calibrated Scorecard",
        subtitle="Evaluates borrower PD, Master Scale Band (R1-R6), and points with zero database dependency.",
        duration_frames=30,
        cursor_start=(300, 150), cursor_end=(500, 150)
    ))

    # 3. Autonomous Agentic Analyst (Prime Underwriting)
    print("  - Rendering Scene 3: Agentic Credit Analyst Underwriting")
    all_frames.extend(make_scene_from_image(
        DOCS / "agentic_analyst.png", step_num=2,
        title="Autonomous Agentic Credit Underwriter",
        subtitle="Autonomous 5-step underwriting: Risk Scoring, Bank Policy Gates, Mitigants, and Underwriting Memo.",
        duration_frames=30,
        cursor_start=(150, 80), cursor_end=(320, 80)
    ))

    # 4. Counterfactual Path-to-Approval
    print("  - Rendering Scene 4: Counterfactual Optimization")
    all_frames.extend(make_scene_from_image(
        DOCS / "agentic_analyst.png", step_num=3,
        title="Counterfactual Optimization Engine",
        subtitle="Prescribes exact minimal actions (e.g. pay down $3,800 revolving debt) to upgrade tier & qualify for approval.",
        duration_frames=30,
        cursor_start=(550, 400), cursor_end=(700, 400)
    ))

    # 5. Model Validation (SR 11-7) & Deciles
    print("  - Rendering Scene 5: Independent Model Risk Validation (SR 11-7)")
    all_frames.extend(make_composite_scene(
        DOCS / "calibration_deciles.png", step_num=4,
        title="Model Risk Validation (Fed SR 11-7 / EBA Compliant)",
        subtitle="Uncovers out-of-time calibration decay: 2015 defaults rose to 14.9% vs 12.4% predicted (1.204 ratio).",
        header_text="🛡️ Fed SR 11-7 Model Validation: Out-of-Time Decile Calibration Curve (95% CI)",
        duration_frames=30
    ))

    # 6. Portfolio Monitoring & Macro Stress Testing
    print("  - Rendering Scene 6: Portfolio Monitoring & Macro Shocks")
    all_frames.extend(make_composite_scene(
        DOCS / "monitoring.png", step_num=5,
        title="Portfolio Monitoring Simulation & The 12-Month Outcome Lag",
        subtitle="Monitors 36 monthly vintages; demonstrates why input drift (CSI/PSI) must serve as a leading indicator.",
        header_text="📈 Monthly Vintage Monitoring Simulation: Input Drift vs Realized Default Rates",
        duration_frames=30
    ))

    # 7. Outro
    print("  - Rendering Scene 7: Production Outro")
    all_frames.extend(make_outro_screen())

    # Save animated GIF
    out_gif = DOCS / "demo.gif"
    print(f"Saving {len(all_frames)} frames to {out_gif} (this may take ~15 seconds)...")

    # Downsample frame rate: pick every 2nd frame for smooth 10-12 fps and fast loading (~4-6 MB)
    optimized_frames = all_frames[::2]
    # Set duration to 120ms per frame
    optimized_frames[0].save(
        out_gif,
        save_all=True,
        append_images=optimized_frames[1:],
        duration=120,
        loop=0,
        optimize=True,
    )
    print(f"Generated {out_gif} ({out_gif.stat().st_size / 1e6:.2f} MB)")
    return out_gif


if __name__ == "__main__":
    build_full_animation()
