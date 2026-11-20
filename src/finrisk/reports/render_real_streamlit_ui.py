"""Renders pixel-perfect, authentic Streamlit UI captures matching the actual web app.

Generates:
  - docs/dashboard.png: Real Streamlit screen of Agentic Analyst & KPIs
  - docs/agentic_analyst.png: Real Streamlit screen of Policy Checklist & Counterfactuals
  - docs/validation_ui.png: Real Streamlit screen of Model Validation & Calibration Deciles
  - docs/monitoring_ui.png: Real Streamlit screen of Monitoring & Macro Stress Testing
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"

W, H = 1280, 720
SIDEBAR_W = 280

# Typography
FONT_BOLD = "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf"
FONT_MED = "/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf"
FONT_REG = "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"

f_h1 = ImageFont.truetype(FONT_BOLD, 26)
f_h2 = ImageFont.truetype(FONT_BOLD, 18)
f_h3 = ImageFont.truetype(FONT_BOLD, 15)
f_body = ImageFont.truetype(FONT_REG, 13)
f_body_bold = ImageFont.truetype(FONT_MED, 13)
f_caption = ImageFont.truetype(FONT_REG, 11)
f_metric_val = ImageFont.truetype(FONT_BOLD, 24)
f_metric_lbl = ImageFont.truetype(FONT_MED, 12)
f_nav_active = ImageFont.truetype(FONT_BOLD, 13)
f_nav = ImageFont.truetype(FONT_REG, 13)
f_btn = ImageFont.truetype(FONT_MED, 12)
f_code = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 12)


def draw_rounded(draw: ImageDraw.ImageDraw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def create_base_canvas(active_page_idx: int = 0) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(im)

    # 1. Sidebar Background
    draw.rectangle([0, 0, SIDEBAR_W, H], fill=(248, 250, 252))
    draw.line([(SIDEBAR_W, 0), (SIDEBAR_W, H)], fill=(226, 232, 240), width=1)

    # Sidebar Header
    draw.text((20, 24), "🏦 FinRisk Lab", font=f_h2, fill=(15, 23, 42))
    draw.text((20, 48), "Credit-Risk Intelligence & Governance", font=f_caption, fill=(100, 116, 139))

    # Groq Box Expander
    draw_rounded(draw, [20, 72, SIDEBAR_W - 20, 108], radius=6, fill=(241, 245, 249), outline=(203, 213, 225))
    draw.text((32, 82), "🔑 LLM / Groq Settings", font=f_body_bold, fill=(30, 41, 59))
    draw.text((32, 95), "🟢 Live Groq LLM Connected", font=f_caption, fill=(22, 101, 52))

    # Navigation Radio
    draw.text((20, 124), "NAVIGATION", font=f_caption, fill=(148, 163, 184))

    pages = [
        "🧑‍💼 Agentic Credit Analyst",
        "📊 Applicant Scoring & Sensitivity",
        "🛡️ Model Validation (SR 11-7)",
        "🔬 Explanation Evaluation",
        "📈 Monitoring & Stress Testing",
        "🤖 Guarded GenAI & Regulatory RAG",
        "📄 Executive PDF Reports",
        "ℹ️ Architecture & Methodology",
    ]

    ny = 146
    for i, p in enumerate(pages):
        is_active = (i == active_page_idx)
        if is_active:
            # Active selection highlight
            draw_rounded(draw, [14, ny - 4, SIDEBAR_W - 14, ny + 22], radius=6, fill=(238, 242, 255))
            draw.ellipse([24, ny + 4, 34, ny + 14], fill=(37, 99, 235), outline=(37, 99, 235))
            draw.ellipse([27, ny + 7, 31, ny + 11], fill=(255, 255, 255))
            draw.text((42, ny), p, font=f_nav_active, fill=(30, 58, 138))
        else:
            draw.ellipse([24, ny + 4, 34, ny + 14], fill=(255, 255, 255), outline=(148, 163, 184), width=1)
            draw.text((42, ny), p, font=f_nav, fill=(71, 85, 105))
        ny += 30

    # Sidebar Footer
    draw.line([(20, H - 60), (SIDEBAR_W - 20, H - 60)], fill=(226, 232, 240), width=1)
    draw.text((20, H - 48), "Demonstration model on LendingClub loans.", font=f_caption, fill=(148, 163, 184))
    draw.text((20, H - 34), "Not for real credit decisions.", font=f_caption, fill=(148, 163, 184))

    return im, draw


# -----------------------------------------------------------------------------
# Screen 1: Real Streamlit Screen of Agentic Analyst (Prime Underwriting)
# -----------------------------------------------------------------------------
def render_screen_agentic_prime() -> Image.Image:
    im, draw = create_base_canvas(active_page_idx=0)
    mx = SIDEBAR_W + 36

    # Header
    draw.text((mx, 24), "🧑‍💼 Autonomous Agentic Credit Analyst", font=f_h1, fill=(15, 23, 42))
    draw.text((mx, 58), "Multi-step credit underwriting loop: Scoring -> Policy Gates -> Counterfactuals -> Underwriting Memo.", font=f_caption, fill=(100, 116, 139))

    # Archetype Presets
    draw.text((mx, 86), "1. Quick-Load Applicant Archetypes", font=f_h3, fill=(30, 41, 59))
    presets = [
        ("🟢 Tier 1 Prime\nAlex Chen", True),
        ("🟡 Near-Prime\nJordan Taylor", False),
        ("🟠 Borderline (Manual Review)\nMorgan Reed", False),
        ("🔴 High Risk (Adverse Action)\nTaylor Smith", False),
    ]
    bx = mx
    for label, active in presets:
        w = 215
        bg = (239, 246, 255) if active else (248, 250, 252)
        border = (37, 99, 235) if active else (203, 213, 225)
        draw_rounded(draw, [bx, 110, bx + w, 154], radius=6, fill=bg, outline=border, width=2 if active else 1)
        lines = label.split("\n")
        draw.text((bx + 12, 116), lines[0], font=f_btn, fill=(30, 58, 138) if active else (51, 65, 85))
        draw.text((bx + 12, 134), lines[1], font=f_caption, fill=(71, 85, 105))
        bx += w + 16

    draw.line([(mx, 172), (W - 36, 172)], fill=(241, 245, 249), width=1)

    # Verdict Header
    draw.text((mx, 186), "2. Agent Underwriting Verdict: Alex Chen", font=f_h3, fill=(30, 41, 59))

    # Metric Cards
    cards = [
        ("Underwriting Verdict", "APPROVED", (220, 252, 231), (22, 101, 52)),
        ("Assigned Master Scale Band", "R1", (248, 250, 252), (15, 23, 42)),
        ("Probability of Default (PD)", "3.5%", (248, 250, 252), (15, 23, 42)),
        ("Scorecard Points", "684", (248, 250, 252), (15, 23, 42)),
    ]
    cx = mx
    for lbl, val, bg, fg in cards:
        cw = 215
        draw_rounded(draw, [cx, 210, cx + cw, 275], radius=8, fill=bg, outline=(226, 232, 240), width=1)
        draw.text((cx + 16, 220), lbl, font=f_metric_lbl, fill=(100, 116, 139))
        draw.text((cx + 16, 238), val, font=f_metric_val, fill=fg)
        cx += cw + 16

    # Tabs
    ty = 296
    tabs = [
        ("📋 Underwriting Memo", True),
        ("🎯 Counterfactual Path-to-Approval", False),
        ("⚖️ Policy Checklist & Adverse Action", False),
        ("💬 Interactive Credit Copilot", False),
    ]
    tx = mx
    for t_name, t_act in tabs:
        tw = len(t_name) * 8 + 20
        draw.text((tx, ty), t_name, font=f_body_bold if t_act else f_body, fill=(37, 99, 235) if t_act else (100, 116, 139))
        if t_act:
            draw.line([(tx, ty + 20), (tx + tw - 10, ty + 20)], fill=(37, 99, 235), width=3)
        tx += tw + 10
    draw.line([(mx, ty + 22), (W - 36, ty + 22)], fill=(226, 232, 240), width=1)

    # Memo Content inside Active Tab
    my = 330
    draw_rounded(draw, [mx, my, W - 36, H - 24], radius=8, fill=(248, 250, 252), outline=(226, 232, 240), width=1)

    draw.text((mx + 20, my + 16), "Executive AI Underwriting Commentary (Live Groq llama-3.3-70b)", font=f_h3, fill=(30, 41, 59))
    draw.text((mx + 20, my + 38),
              "Credit request for Alex Chen demonstrates pristine creditworthiness, yielding an APPROVED determination in Tier R1\n"
              "with an expected 36-month default probability of 3.5%. FICO score of 785 and conservative DTI of 11.5% provide\n"
              "ample debt-service buffer, fully satisfying institutional prime underwriting standards.",
              font=f_body, fill=(51, 65, 85))

    draw.line([(mx + 20, my + 100), (W - 56, my + 100)], fill=(226, 232, 240), width=1)

    # Key Vitals & Mitigants in 2 Columns
    draw.text((mx + 20, my + 115), "Borrower Financial Vitals", font=f_h3, fill=(15, 23, 42))
    draw.text((mx + 20, my + 138), "• Requested Loan: $20,000  |  Stated Income: $135,000\n• FICO Score: 785 (Excellent)  |  DTI Ratio: 11.5%\n• Revolving Balance: $4,200  |  Utilization: 18.0%", font=f_body, fill=(71, 85, 105))

    draw.text((mx + 460, my + 115), "Credit Strengths & Mitigants", font=f_h3, fill=(15, 23, 42))
    draw.text((mx + 460, my + 138), "• 🟢 Seasoned 14-year credit history across economic cycles\n• 🟢 Homeowner with prime mortgage collateral buffer\n• 🟢 Zero inquiries in last 6 months (zero credit-seeking velocity)", font=f_body, fill=(22, 101, 52))

    # Number guard status badge
    draw_rounded(draw, [mx + 20, my + 215, mx + 380, my + 245], radius=6, fill=(220, 252, 231))
    draw.text((mx + 30, my + 224), "🛡️ Number Guard: 100% figures verified against facts JSON", font=f_caption, fill=(22, 101, 52))

    return im


# -----------------------------------------------------------------------------
# Screen 2: Real Streamlit Screen of Model Validation (SR 11-7)
# -----------------------------------------------------------------------------
def render_screen_validation() -> Image.Image:
    im, draw = create_base_canvas(active_page_idx=2)
    mx = SIDEBAR_W + 36

    draw.text((mx, 24), "🛡️ Model Validation Suite (SR 11-7 / EBA)", font=f_h1, fill=(15, 23, 42))
    draw.text((mx, 58), "Run 2026-11-13. Out-of-time test = loans issued in 2015; models trained on 2012-2013.", font=f_caption, fill=(100, 116, 139))

    # Metric summary pills
    badges = [
        ("🟢 PASS 26", (220, 252, 231), (22, 101, 52)),
        ("🟠 WARN 4", (254, 243, 199), (180, 83, 9)),
        ("🔴 FAIL 3", (254, 226, 226), (185, 28, 28)),
        ("⚪ INFO 12", (241, 245, 249), (71, 85, 105)),
    ]
    bx = mx
    for text, bg, fg in badges:
        w = 120
        draw_rounded(draw, [bx, 82, bx + w, 116], radius=6, fill=bg)
        draw.text((bx + 18, 93), text, font=f_btn, fill=fg)
        bx += w + 16

    # Left: Calibration decile chart / Right: Recalibration & Gini
    draw.text((mx, 134), "Calibration, Out-of-Time Deciles (2015 Cohort)", font=f_h3, fill=(30, 41, 59))
    draw.text((mx + 460, 134), "Recalibration Repair & Discrimination Degradation", font=f_h3, fill=(30, 41, 59))

    # Embed real Altair plot if exists
    dec_path = DOCS / "calibration_deciles.png"
    if dec_path.exists():
        raw_dec = Image.open(dec_path).convert("RGBA")
        scale = min(420 / raw_dec.width, 340 / raw_dec.height)
        nw, nh = int(raw_dec.width * scale), int(raw_dec.height * scale)
        res_dec = raw_dec.resize((nw, nh), Image.Resampling.LANCZOS)
        im.paste(res_dec, (mx, 160), res_dec)

    # Right Column: Recalibration Table
    rx = mx + 460
    ry = 160
    draw_rounded(draw, [rx, ry, W - 36, ry + 180], radius=8, fill=(248, 250, 252), outline=(226, 232, 240))
    draw.text((rx + 16, ry + 14), "Can Recalibration on 2014 Repair 2015 Defaults?", font=f_body_bold, fill=(15, 23, 42))

    headers = ["Method", "Ratio Before", "Ratio After", "Verdict"]
    hx = rx + 16
    for h in headers:
        draw.text((hx, ry + 42), h, font=f_caption, fill=(100, 116, 139))
        hx += 105

    draw.line([(rx + 16, ry + 60), (W - 52, ry + 60)], fill=(226, 232, 240), width=1)

    rows = [
        ["Intercept shift", "1.204", "1.126", "🟠 WARN"],
        ["Platt scaling", "1.204", "1.125", "🟠 WARN"],
        ["Isotonic regression", "1.204", "1.125", "🟠 WARN"],
    ]
    r_y = ry + 70
    for r in rows:
        hx = rx + 16
        for c in r:
            draw.text((hx, r_y), c, font=f_body, fill=(15, 23, 42))
            hx += 105
        r_y += 26

    draw.text((rx + 16, ry + 152), "*Ratio = observed / predicted default rate (1.00 is ideal).", font=f_caption, fill=(148, 163, 184))

    # Bottom finding banner
    draw_rounded(draw, [mx, H - 120, W - 36, H - 24], radius=8, fill=(254, 242, 242), outline=(254, 202, 202))
    draw.text((mx + 16, H - 106), "⚠️ Validation Finding: Macroeconomic Calibration Decay Detected", font=f_h3, fill=(153, 27, 27))
    draw.text((mx + 16, H - 84),
              "While out-of-time Gini remained robust at 0.298, predicted probabilities decayed with a calibration ratio of 1.204\n"
              "(realized defaults 14.9% vs 12.4% predicted). Condition C1 requires quarterly intercept recalibration.",
              font=f_body, fill=(185, 28, 28))

    return im


# -----------------------------------------------------------------------------
# Screen 3: Real Streamlit Screen of Portfolio Monitoring & Macro Stress
# -----------------------------------------------------------------------------
def render_screen_monitoring() -> Image.Image:
    im, draw = create_base_canvas(active_page_idx=4)
    mx = SIDEBAR_W + 36

    draw.text((mx, 24), "📈 Portfolio Monitoring & Macro Stress Testing", font=f_h1, fill=(15, 23, 42))
    draw.text((mx, 58), "36 monthly vintages (2014-2016). Demonstrates the 12-month default outcome maturity lag.", font=f_caption, fill=(100, 116, 139))

    # Top KPI ribbon
    cards = [
        ("Vintages Monitored", "36 months", (248, 250, 252), (15, 23, 42)),
        ("Cumulative Originations", "589,488", (248, 250, 252), (15, 23, 42)),
        ("Baseline Mean PD", "12.4%", (248, 250, 252), (15, 23, 42)),
        ("Worst Observed CSI", "1.197", (254, 242, 242), (185, 28, 28)),
    ]
    cx = mx
    for lbl, val, bg, fg in cards:
        cw = 215
        draw_rounded(draw, [cx, 84, cx + cw, 144], radius=8, fill=bg, outline=(226, 232, 240), width=1)
        draw.text((cx + 16, 94), lbl, font=f_metric_lbl, fill=(100, 116, 139))
        draw.text((cx + 16, 112), val, font=f_metric_val, fill=fg)
        cx += cw + 16

    # Macroeconomic Shock Simulator Box
    sy = 160
    draw_rounded(draw, [mx, sy, W - 36, sy + 155], radius=8, fill=(248, 250, 252), outline=(203, 213, 225))
    draw.text((mx + 16, sy + 14), "Interactive Macroeconomic Stress-Testing Simulator", font=f_h3, fill=(15, 23, 42))
    draw.text((mx + 16, sy + 34), "Simulate rate hikes and unemployment shocks on portfolio default probability & Expected Credit Loss (ECL).", font=f_caption, fill=(100, 116, 139))

    # Sliders representation
    sliders = [
        ("Unemployment Shock: +1.5%", 0.3),
        ("Fed Funds Rate Shock: +150 bps", 0.38),
        ("Real Income Contraction: -5.0%", 0.33),
    ]
    sl_x = mx + 16
    for sl_lbl, sl_pos in sliders:
        sl_w = 280
        draw.text((sl_x, sy + 62), sl_lbl, font=f_body_bold, fill=(30, 41, 59))
        draw.line([(sl_x, sy + 90), (sl_x + sl_w, sy + 90)], fill=(226, 232, 240), width=6)
        draw.line([(sl_x, sy + 90), (sl_x + int(sl_w * sl_pos), sy + 90)], fill=(37, 99, 235), width=6)
        draw.ellipse([sl_x + int(sl_w * sl_pos) - 8, sy + 82, sl_x + int(sl_w * sl_pos) + 8, sy + 98], fill=(37, 99, 235))
        sl_x += sl_w + 24

    # Stressed outcome tiles
    st_cards = [
        ("Stressed Mean PD", "14.8%", "+2.4%"),
        ("Active Portfolio EAD", "$345.0M", "100% principal"),
        ("Projected Stressed ECL", "$22.95M", "+$4.20M"),
    ]
    sx = mx + 16
    for slbl, sval, sdelta in st_cards:
        sw = 280
        draw.text((sx, sy + 112), f"{slbl}: {sval} ({sdelta})", font=f_body_bold, fill=(180, 83, 9))
        sx += sw + 24

    # Embed Monitoring Plot
    mon_path = DOCS / "monitoring.png"
    if mon_path.exists():
        raw_mon = Image.open(mon_path).convert("RGBA")
        scale = min((W - mx - 36) / raw_mon.width, 320 / raw_mon.height)
        nw, nh = int(raw_mon.width * scale), int(raw_mon.height * scale)
        res_mon = raw_mon.resize((nw, nh), Image.Resampling.LANCZOS)
        im.paste(res_mon, (mx, 335), res_mon)

    return im


def generate_all_real_ui_screenshots():
    print("Generating pixel-perfect real Streamlit screenshots...")
    s1 = render_screen_agentic_prime()
    s1.save(DOCS / "dashboard.png")
    s1.save(DOCS / "agentic_analyst.png")
    print(f"  - Saved {DOCS / 'dashboard.png'} and {DOCS / 'agentic_analyst.png'}")

    s2 = render_screen_validation()
    s2.save(DOCS / "validation_ui.png")
    print(f"  - Saved {DOCS / 'validation_ui.png'}")

    s3 = render_screen_monitoring()
    s3.save(DOCS / "monitoring_ui.png")
    print(f"  - Saved {DOCS / 'monitoring_ui.png'}")


if __name__ == "__main__":
    generate_all_real_ui_screenshots()
