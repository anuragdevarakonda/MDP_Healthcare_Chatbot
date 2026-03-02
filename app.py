import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import graphviz
import math
import io
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── APP CONFIG ───
st.set_page_config(page_title="Clinical MDP · Decision Engine", layout="wide", page_icon="🏥", initial_sidebar_state="expanded")

# ─── LIGHT THEME CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #f8fafc; color: #1e293b; }
[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e2e8f0; }
[data-testid="stSidebar"] * { color: #334155 !important; }
.hero-header {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 50%, #bfdbfe 100%);
    border: 1px solid #93c5fd; border-radius: 12px;
    padding: 28px 36px; margin-bottom: 24px; position: relative; overflow: hidden;
}
.hero-title { font-family:'JetBrains Mono',monospace; font-size:24px; font-weight:700; color:#1e40af; margin:0 0 6px 0; }
.hero-sub { font-size:12px; color:#64748b; font-family:'JetBrains Mono',monospace; letter-spacing:1.5px; text-transform:uppercase; }
.section-title {
    font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:2px;
    text-transform:uppercase; color:#2563eb; border-bottom:2px solid #dbeafe;
    padding-bottom:8px; margin:24px 0 14px 0; font-weight:600;
}
.rec-box {
    background: linear-gradient(135deg, #eff6ff, #dbeafe); border:1px solid #93c5fd;
    border-left:4px solid #2563eb; border-radius:10px; padding:20px 24px; margin:16px 0;
}
.rec-action { font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:700; color:#1e40af; margin-bottom:8px; }
.rec-confidence { font-size:12px; color:#64748b; font-family:'JetBrains Mono',monospace; }
.info-panel {
    background:#ffffff; border:1px solid #e2e8f0; border-radius:10px;
    padding:18px 20px; margin:12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.info-panel h4 { font-family:'JetBrains Mono',monospace; font-size:13px; color:#475569; margin:0 0 10px 0; font-weight:600; }
.info-panel p { font-size:14px; color:#334155; line-height:1.7; margin:0; }
.interp-box {
    background:#f0fdf4; border:1px solid #bbf7d0; border-left:3px solid #22c55e;
    border-radius:8px; padding:12px 16px; margin:8px 0 16px 0; font-size:13px; color:#166534;
}
.risk-tag {
    display:inline-block; background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);
    color:#dc2626; font-size:11px; padding:3px 10px; border-radius:20px; margin:2px 4px;
    font-family:'JetBrains Mono',monospace; font-weight:500;
}
.flow-step {
    display:flex; align-items:flex-start; gap:14px; margin:10px 0; padding:14px 16px;
    background:#ffffff; border:1px solid #e2e8f0; border-radius:10px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.flow-num {
    background:#2563eb; color:white; min-width:28px; height:28px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-family:'JetBrains Mono',monospace; font-size:12px; font-weight:700;
}
.flow-content h5 { font-family:'JetBrains Mono',monospace; font-size:13px; color:#1e40af; margin:0 0 4px 0; }
.flow-content p { font-size:13px; color:#475569; margin:0; line-height:1.6; }
.warning-box {
    background:#fffbeb; border:1px solid #fde68a; border-radius:8px;
    padding:12px 16px; font-size:12px; color:#92400e; margin:10px 0;
    font-family:'JetBrains Mono',monospace;
}
.styled-table { width:100%; border-collapse:collapse; font-size:12px; font-family:'JetBrains Mono',monospace; }
.styled-table th {
    background:#f1f5f9; color:#1e40af; padding:10px 12px; text-align:left;
    font-size:10px; letter-spacing:1px; text-transform:uppercase; border-bottom:2px solid #cbd5e1;
}
.styled-table td { padding:9px 12px; border-bottom:1px solid #e2e8f0; color:#334155; }
.styled-table tr:hover td { background:#f8fafc; }
hr { border-color: #e2e8f0 !important; }
[data-testid="stTabs"] button {
    font-family:'JetBrains Mono',monospace !important; font-size:11px !important;
    letter-spacing:1px !important; text-transform:uppercase !important;
}
</style>
""", unsafe_allow_html=True)

# ─── MDP ENGINE ───
def policy_iteration(states, actions, P, R, gamma=0.9):
    policy = {s: actions[0] for s in states}
    V = {s: 0.0 for s in states}
    for _ in range(100):
        for _ in range(2000):
            delta = 0
            for s in states:
                a = policy[s]
                new_v = sum(prob * (R[s][a] + gamma * V.get(sn, 0)) for sn, prob in P[s][a].items())
                delta = max(delta, abs(new_v - V[s]))
                V[s] = new_v
            if delta < 1e-9:
                break
        stable = True
        for s in states:
            old = policy[s]
            policy[s] = max(actions, key=lambda a: sum(
                prob * (R[s][a] + gamma * V.get(sn, 0)) for sn, prob in P[s][a].items()))
            if old != policy[s]:
                stable = False
        if stable:
            break
    return policy, V

def compute_qvalues(s, actions, P, R, V, gamma=0.9):
    return {a: sum(prob * (R[s][a] + gamma * V.get(sn, 0)) for sn, prob in P[s][a].items()) for a in actions}

# ─── DEFINITIONS ───
STATES = ['S0', 'S1', 'S2', 'S3']
STATE_SHORT = {'S0': 'Healthy', 'S1': 'Mild Sepsis', 'S2': 'Critical/Shock', 'S3': 'Death'}
STATE_LONG = {
    'S0': 'Healthy (No Active Infection)', 'S1': 'Mild Sepsis (Suspected/Confirmed Infection + SIRS)',
    'S2': 'Severe Sepsis / Septic Shock (Organ Dysfunction)', 'S3': 'Death (Absorbing State)'
}
ACTIONS = ['A0', 'A1', 'A2', 'A3']
ACTION_LABELS = {'A0': 'Watchful Waiting', 'A1': 'Antibiotic Bundle (3-hr)', 'A2': 'Antibiotics + Vasopressors', 'A3': 'Surgical Intervention'}
STATE_COLORS = {'S0': '#22c55e', 'S1': '#f59e0b', 'S2': '#ef4444', 'S3': '#6b7280'}
LIGHT_CHART = dict(paper_bgcolor='white', plot_bgcolor='#fafbfc',
                   font=dict(family='Inter', size=12, color='#334155'))

# ─── BASE PROBABILITIES ───
BASE_P = {
    'S0': {'A0': {'S0': 0.95, 'S1': 0.04, 'S2': 0.01, 'S3': 0.00}, 'A1': {'S0': 0.96, 'S1': 0.03, 'S2': 0.01, 'S3': 0.00},
           'A2': {'S0': 0.93, 'S1': 0.04, 'S2': 0.01, 'S3': 0.02}, 'A3': {'S0': 0.90, 'S1': 0.04, 'S2': 0.02, 'S3': 0.04}},
    'S1': {'A0': {'S0': 0.50, 'S1': 0.35, 'S2': 0.09, 'S3': 0.06}, 'A1': {'S0': 0.72, 'S1': 0.21, 'S2': 0.05, 'S3': 0.02},
           'A2': {'S0': 0.67, 'S1': 0.22, 'S2': 0.07, 'S3': 0.04}, 'A3': {'S0': 0.58, 'S1': 0.24, 'S2': 0.09, 'S3': 0.09}},
    'S2': {'A0': {'S0': 0.10, 'S1': 0.22, 'S2': 0.35, 'S3': 0.33}, 'A1': {'S0': 0.30, 'S1': 0.28, 'S2': 0.27, 'S3': 0.15},
           'A2': {'S0': 0.45, 'S1': 0.26, 'S2': 0.20, 'S3': 0.09}, 'A3': {'S0': 0.38, 'S1': 0.22, 'S2': 0.22, 'S3': 0.18}},
    'S3': {'A0': {'S3': 1.0}, 'A1': {'S3': 1.0}, 'A2': {'S3': 1.0}, 'A3': {'S3': 1.0}}
}
BASE_R = {
    'S0': {'A0': 10.0, 'A1': 9.5, 'A2': 8.0, 'A3': 5.0},
    'S1': {'A0': -2.0, 'A1': 3.5, 'A2': 2.0, 'A3': -1.0},
    'S2': {'A0': -20.0, 'A1': -8.0, 'A2': -5.0, 'A3': -10.0},
    'S3': {'A0': -100.0, 'A1': -100.0, 'A2': -100.0, 'A3': -100.0},
}

# ─── RISK MODIFIER ENGINE ───
def apply_risk_modifiers(P_base, R_base, age, is_smoker, has_cvd, has_diabetes, has_immune):
    import copy
    P, R = copy.deepcopy(P_base), copy.deepcopy(R_base)
    age_factor = 1.0 + max(0, (age - 50) / 100)
    mort_mult = min(3.5, age_factor * (1.30 if is_smoker else 1.0) * (1.35 if has_cvd else 1.0) * (1.20 if has_diabetes else 1.0) * (1.80 if has_immune else 1.0))
    eff_age = age + (10 if is_smoker else 0) + (8 if has_cvd else 0) + (5 if has_diabetes else 0) + (15 if has_immune else 0)
    rec_mod = 1.0 / (1 + math.exp((eff_age - 72) / 12))
    for state in ['S1', 'S2']:
        for a in ACTIONS:
            base = BASE_P[state][a]
            nm = min(0.75, base.get('S3', 0.05) * mort_mult)
            nr = max(0.02, base.get('S0', 0.4) * rec_mod)
            rem = max(0.01, 1.0 - nm - nr)
            b1, b2 = base.get('S1', 0.25), base.get('S2', 0.15)
            d = max(b1 + b2, 0.01)
            P[state][a] = {'S3': round(nm, 4), 'S0': round(nr, 4), 'S1': round(rem * b1 / d, 4), 'S2': round(rem * b2 / d, 4)}
            s = sum(P[state][a].values())
            for k in P[state][a]:
                P[state][a][k] = round(P[state][a][k] / s, 4)
    rp = (mort_mult - 1.0) * 3.0
    for s in ['S1', 'S2']:
        for a in ACTIONS:
            R[s][a] -= rp
    return P, R, mort_mult, rec_mod

# ─── PDF REPORT GENERATOR ───
def generate_pdf(P_dyn, R_dyn, opt_policy, state_values, age, gender,
                 is_smoker, has_cvd, has_diab, has_immune, mort_mult, rec_mod, risk_score, risk_label):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Page 1: Executive Summary ──
    pdf.add_page()
    pdf.set_fill_color(37, 99, 235)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_y(8)
    pdf.cell(0, 10, 'Clinical MDP Decision Engine', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, 'Executive Summary Report', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(40)
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}  |  Markov Decision Process  |  Policy Iteration Solver', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Patient Profile
    pdf.set_text_color(30, 64, 175)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Patient Bio-Profile', new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(219, 234, 254)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_text_color(51, 65, 85)
    pdf.set_font('Helvetica', '', 10)
    risks = []
    if is_smoker: risks.append('Active Smoker')
    if has_cvd: risks.append('Cardiovascular Disease')
    if has_diab: risks.append('Diabetes Mellitus')
    if has_immune: risks.append('Immunocompromised')
    risk_str = ', '.join(risks) if risks else 'None'
    for label, val in [('Age', str(age)), ('Biological Sex', gender), ('Risk Factors', risk_str)]:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(45, 7, f'{label}:')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 7, val, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Risk Assessment
    pdf.set_text_color(30, 64, 175)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Risk Assessment', new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_text_color(51, 65, 85)
    pdf.set_font('Helvetica', '', 10)
    for label, val in [('Composite Risk Score', f'{risk_score}/100 ({risk_label})'),
                       ('Mortality Multiplier', f'{mort_mult:.2f}x baseline'),
                       ('Recovery Modifier', f'{rec_mod:.2f}x baseline')]:
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(50, 7, f'{label}:')
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 7, val, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Optimal Policy Table
    pdf.set_text_color(30, 64, 175)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Optimal Treatment Policy (Solved via Policy Iteration)', new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_text_color(30, 64, 175)
    for h, w in [('State', 40), ('Optimal Action', 55), ('Value V(s)', 35), ('Interpretation', 60)]:
        pdf.cell(w, 7, h, border=1, fill=True, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(51, 65, 85)
    interps = {'S0': 'Best long-term prognosis', 'S1': 'Recoverable with treatment', 'S2': 'Urgent intervention needed', 'S3': 'Absorbing terminal state'}
    for s in STATES:
        pdf.cell(40, 7, STATE_SHORT[s], border=1, align='C')
        pdf.cell(55, 7, ACTION_LABELS[opt_policy[s]], border=1, align='C')
        pdf.cell(35, 7, f'{state_values[s]:.2f}', border=1, align='C')
        pdf.cell(60, 7, interps[s], border=1, align='C')
        pdf.ln()
    pdf.ln(2)

    # Value Function Chart (matplotlib)
    fig_v, ax_v = plt.subplots(figsize=(6, 2.2))
    colors_v = [STATE_COLORS[s] for s in STATES]
    bars = ax_v.bar([STATE_SHORT[s] for s in STATES], [state_values[s] for s in STATES], color=colors_v, edgecolor='white', linewidth=1.5)
    for bar, s in zip(bars, STATES):
        ax_v.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{state_values[s]:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax_v.set_title('Value Function V(s) — Expected Long-Term Reward per State', fontsize=9, fontweight='bold', color='#1e293b')
    ax_v.set_ylabel('V(s)', fontsize=8)
    ax_v.spines[['top','right']].set_visible(False)
    ax_v.tick_params(labelsize=8)
    fig_v.tight_layout()
    buf_v = io.BytesIO()
    fig_v.savefig(buf_v, format='png', dpi=150, bbox_inches='tight')
    buf_v.seek(0)
    plt.close(fig_v)
    pdf.image(buf_v, x=15, w=180)
    pdf.ln(3)

    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 4, 'Interpretation: Higher V(s) indicates a more favorable long-term prognosis. Healthy has the highest value; Death is the most negative (absorbing). Treatment actions aim to maximize transitions toward higher-value states.')
    pdf.ln(2)

    # ── Page 2: Technical Details ──
    pdf.add_page()
    pdf.set_text_color(30, 64, 175)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Transition Probability Matrix (Patient-Adjusted)', new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font('Helvetica', 'B', 7)
    pdf.set_fill_color(241, 245, 249)
    pdf.set_text_color(30, 64, 175)
    cols = [('From', 22), ('Action', 38), ('P->Healthy', 22), ('P->Mild', 22), ('P->Critical', 22), ('P->Death', 22), ('R(s,a)', 18), ('Optimal', 18)]
    for h, w in cols:
        pdf.cell(w, 6, h, border=1, fill=True, align='C')
    pdf.ln()
    pdf.set_font('Helvetica', '', 7)
    pdf.set_text_color(51, 65, 85)
    for s in STATES:
        for a in ACTIONS:
            p = P_dyn[s][a]
            is_opt = opt_policy[s] == a
            if is_opt:
                pdf.set_fill_color(239, 246, 255)
            pdf.cell(22, 5, STATE_SHORT[s], border=1, align='C', fill=is_opt)
            pdf.cell(38, 5, ACTION_LABELS[a], border=1, align='C', fill=is_opt)
            pdf.cell(22, 5, f'{p.get("S0",0):.4f}', border=1, align='C', fill=is_opt)
            pdf.cell(22, 5, f'{p.get("S1",0):.4f}', border=1, align='C', fill=is_opt)
            pdf.cell(22, 5, f'{p.get("S2",0):.4f}', border=1, align='C', fill=is_opt)
            pdf.cell(22, 5, f'{p.get("S3",0):.4f}', border=1, align='C', fill=is_opt)
            pdf.cell(18, 5, f'{R_dyn[s][a]:.1f}', border=1, align='C', fill=is_opt)
            pdf.cell(18, 5, 'YES' if is_opt else '', border=1, align='C', fill=is_opt)
            pdf.ln()
            if is_opt:
                pdf.set_fill_color(241, 245, 249)
    pdf.ln(2)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 4, 'Interpretation: Each row shows P(next_state | current_state, action). Blue-highlighted rows are the optimal policy actions selected by the MDP solver. R(s,a) is the immediate QALY-calibrated reward. Higher P->Healthy and lower P->Death indicate better treatment outcomes.')
    pdf.ln(3)

    # Q-Value chart
    fig_q, ax_q = plt.subplots(figsize=(6, 2.5))
    x_pos = np.arange(len(ACTIONS))
    width = 0.25
    for i, s in enumerate(['S0', 'S1', 'S2']):
        qv = compute_qvalues(s, ACTIONS, P_dyn, R_dyn, state_values)
        vals = [qv[a] for a in ACTIONS]
        ax_q.bar(x_pos + i * width, vals, width, label=STATE_SHORT[s], color=STATE_COLORS[s], edgecolor='white')
    ax_q.set_xticks(x_pos + width)
    ax_q.set_xticklabels([ACTION_LABELS[a] for a in ACTIONS], fontsize=7, rotation=-10)
    ax_q.set_title('Q-Value Comparison: Expected Value of Each Action per State', fontsize=9, fontweight='bold', color='#1e293b')
    ax_q.set_ylabel('Q(s,a)', fontsize=8)
    ax_q.legend(fontsize=7, loc='upper right')
    ax_q.spines[['top','right']].set_visible(False)
    ax_q.tick_params(labelsize=7)
    fig_q.tight_layout()
    buf_q = io.BytesIO()
    fig_q.savefig(buf_q, format='png', dpi=150, bbox_inches='tight')
    buf_q.seek(0)
    plt.close(fig_q)
    pdf.image(buf_q, x=15, w=180)
    pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 4, 'Interpretation: Q(s,a) represents the expected cumulative discounted reward of taking action a in state s. The optimal policy selects the action with the highest Q-value in each state. Wider gaps between bars indicate stronger preference for the optimal action.')
    pdf.ln(3)

    # Methodology
    pdf.set_text_color(30, 64, 175)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Methodology & Evidence Base', new_x="LMARGIN", new_y="NEXT")
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_text_color(51, 65, 85)
    pdf.set_font('Helvetica', '', 8)
    pdf.multi_cell(0, 4, 'This engine implements a finite-state, infinite-horizon Markov Decision Process (MDP) solved via Policy Iteration (Howard, 1960). Transition probabilities are calibrated to HCUP 2021 national mortality data, NEJM 2017 sepsis bundle trials, and SICU cohort studies (PMC 2020). Patient risk factors modify base probabilities using evidence-based mortality multipliers and logistic recovery decay. Discount factor = 0.9 (clinical MDP standard). Convergence threshold = 1e-9.')
    pdf.ln(3)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(146, 64, 14)
    pdf.multi_cell(0, 4, 'DISCLAIMER: This model is for educational and research purposes only. It is not a substitute for clinical judgment. Treatment decisions must always involve qualified healthcare professionals.')

    return bytes(pdf.output())

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("<div style='font-family:JetBrains Mono,monospace;font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#2563eb;margin-bottom:18px;padding-bottom:10px;border-bottom:2px solid #dbeafe;font-weight:600;'>Patient Bio-Profile</div>", unsafe_allow_html=True)
    age = st.slider("Age", 18, 100, 55)
    gender = st.selectbox("Biological Sex", ["Male", "Female"])
    st.markdown("<div style='margin-top:12px;font-size:10px;color:#64748b;letter-spacing:1px;text-transform:uppercase;font-weight:600;'>Risk Factors</div>", unsafe_allow_html=True)
    is_smoker = st.toggle("Active Smoker", value=False)
    has_cvd = st.toggle("Cardiovascular Disease", value=False)
    has_diab = st.toggle("Diabetes Mellitus", value=False)
    has_immune = st.toggle("Immunocompromised", value=False)
    st.markdown("<hr>", unsafe_allow_html=True)

    P_dyn, R_dyn, mort_mult, rec_mod = apply_risk_modifiers(BASE_P, BASE_R, age, is_smoker, has_cvd, has_diab, has_immune)
    opt_policy, state_values = policy_iteration(STATES, ACTIONS, P_dyn, R_dyn, 0.9)
    risk_score = int(min(100, ((mort_mult - 1.0) / 2.5) * 100))
    risk_color = "#22c55e" if risk_score < 30 else "#f59e0b" if risk_score < 65 else "#ef4444"
    risk_label = "LOW" if risk_score < 30 else "MODERATE" if risk_score < 65 else "HIGH"

    st.markdown(f"""
    <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center;'>
        <div style='font-size:10px;color:#64748b;font-family:JetBrains Mono;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:8px;font-weight:600;'>Patient Risk Index</div>
        <div style='font-size:38px;font-weight:700;font-family:JetBrains Mono;color:{risk_color};'>{risk_score}</div>
        <div style='font-size:11px;color:{risk_color};font-family:JetBrains Mono;letter-spacing:2px;font-weight:600;'>{risk_label} RISK</div>
        <div style='margin-top:10px;font-size:11px;color:#64748b;'>Mortality: {mort_mult:.2f}x  |  Recovery: {rec_mod:.2f}x</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    pdf_bytes = generate_pdf(P_dyn, R_dyn, opt_policy, state_values, age, gender, is_smoker, has_cvd, has_diab, has_immune, mort_mult, rec_mod, risk_score, risk_label)
    st.download_button("📄 Download Executive Report (PDF)", data=pdf_bytes, file_name="MDP_Clinical_Report.pdf", mime="application/pdf", use_container_width=True)
    st.markdown("<div style='margin-top:14px;font-size:10px;color:#94a3b8;line-height:1.7;'>⚠️ For educational & research use only.</div>", unsafe_allow_html=True)

# ─── HEADER ───
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🏥 Clinical MDP · Recommendation Engine</div>
    <div class="hero-sub">Markov Decision Process · Real-World Calibrated · Policy Iteration Solver</div>
</div>
""", unsafe_allow_html=True)

tab_sim, tab_verify, tab_explain = st.tabs(["⚡  Live Simulation", "📊  Probability Verification", "🧠  Model Explanation"])

# ═══════════════════════════════════════
# TAB 1 — LIVE SIMULATION
# ═══════════════════════════════════════
with tab_sim:
    col_l, col_r = st.columns([1, 1.4], gap="large")

    with col_l:
        st.markdown('<div class="section-title">Current Patient Status</div>', unsafe_allow_html=True)
        status_map = {'Healthy (No Active Infection)': 'S0', 'Mild Sepsis (Fever, Infection Signs)': 'S1', 'Severe Sepsis / Septic Shock': 'S2'}
        current = st.selectbox("State:", list(status_map.keys()), label_visibility="collapsed")
        s_key = status_map[current]

        sv = state_values[s_key]
        sv_color = "#22c55e" if sv > 0 else "#ef4444"
        st.markdown(f"""
        <div style='display:flex;gap:10px;margin:10px 0 14px 0;'>
            <div style='flex:1;background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.06);'>
                <div style='font-size:9px;color:#64748b;font-family:JetBrains Mono;letter-spacing:1px;text-transform:uppercase;font-weight:600;'>State Value V(s)</div>
                <div style='font-size:22px;font-weight:700;font-family:JetBrains Mono;color:{sv_color};'>{sv:.2f}</div>
            </div>
            <div style='flex:1;background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:12px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.06);'>
                <div style='font-size:9px;color:#64748b;font-family:JetBrains Mono;letter-spacing:1px;text-transform:uppercase;font-weight:600;'>Optimal Policy</div>
                <div style='font-size:12px;font-weight:600;font-family:JetBrains Mono;color:#2563eb;margin-top:4px;'>{ACTION_LABELS[opt_policy[s_key]]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        risks = []
        if is_smoker: risks.append("Smoker")
        if has_cvd: risks.append("CVD")
        if has_diab: risks.append("Diabetes")
        if has_immune: risks.append("Immunocomp")
        if age > 65: risks.append(f"Age {age}")
        if risks:
            tags = "".join(f'<span class="risk-tag">{r}</span>' for r in risks)
            st.markdown(f'<div style="margin-bottom:14px;">Active Risk Factors: {tags}</div>', unsafe_allow_html=True)

        go_btn = st.button("⚡ Generate Clinical Recommendation", use_container_width=True)

        if go_btn:
            best_a = opt_policy[s_key]
            q_vals = compute_qvalues(s_key, ACTIONS, P_dyn, R_dyn, state_values)
            q_ranked = sorted(q_vals.items(), key=lambda x: x[1], reverse=True)
            best_q = q_ranked[0][1]
            q_range = max(best_q - q_ranked[-1][1], 0.01)
            confidence = min(99, int(((best_q - q_ranked[1][1]) / q_range) * 60 + 50))

            icons = {'A0': '👁️', 'A1': '💊', 'A2': '💉', 'A3': '🔪'}
            st.markdown(f"""
            <div class="rec-box">
                <div class="rec-action">{icons[best_a]} {ACTION_LABELS[best_a]}</div>
                <div class="rec-confidence">MODEL CONFIDENCE: {confidence}% &nbsp;·&nbsp; Q-VALUE: {best_q:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-title">Clinical Reasoning</div>', unsafe_allow_html=True)
            state_ctx = {
                'S0': "The patient is in a stable healthy state with no active infection signs.",
                'S1': "Patient presents with mild sepsis: suspected/confirmed infection with systemic inflammatory response.",
                'S2': "Patient is in severe sepsis or septic shock — acute organ dysfunction. Time-critical emergency.",
            }
            action_ctx = {
                'A0': "**Watchful Waiting** is recommended. Close monitoring (vitals q2h). Reassess within 6 hours.",
                'A1': "**Antibiotic Bundle (3-hr)** is optimal. Measure lactate, draw cultures, administer broad-spectrum antibiotics within 1 hour. Reduces 30-day mortality by ~25% (NEJM 2017).",
                'A2': "**Antibiotics + Vasopressors** for septic shock. Norepinephrine to target MAP >= 65 mmHg. Reassess SOFA every 4-6h.",
                'A3': "**Surgical Intervention** for source control. Perform within 6-12h if surgically correctable.",
            }
            risk_note = ""
            if risks:
                risk_note = f" With {', '.join(risks).lower()}, this patient carries a {mort_mult:.1f}x mortality multiplier. Recovery reduced by {rec_mod:.2f}x."
            st.markdown(f'<div class="info-panel"><p>{state_ctx[s_key]} {action_ctx[best_a]}{risk_note}</p></div>', unsafe_allow_html=True)

            # Q-value ranking table
            st.markdown('<div class="section-title">Action Q-Value Ranking</div>', unsafe_allow_html=True)
            worst_q = q_ranked[-1][1]
            rows_html = ""
            for rank, (a, q) in enumerate(q_ranked):
                bar = int(((q - worst_q) / q_range) * 80)
                star = "⭐ " if rank == 0 else "&nbsp;&nbsp;&nbsp;"
                qcol = "#22c55e" if q > 0 else "#ef4444"
                rows_html += f"<tr><td>{star}{ACTION_LABELS[a]}</td><td style='color:{qcol};font-weight:600;'>{q:.3f}</td><td><div style='background:#e2e8f0;border-radius:3px;height:7px;width:100px;'><div style='background:{('#2563eb' if rank==0 else '#94a3b8')};width:{bar}%;height:7px;border-radius:3px;'></div></div></td></tr>"
            st.markdown(f"""<table class="styled-table"><thead><tr><th>Action</th><th>Q-Value</th><th>Relative</th></tr></thead><tbody>{rows_html}</tbody></table>""", unsafe_allow_html=True)
            st.markdown('<div class="interp-box">📊 <strong>Interpretation:</strong> The Q-value represents the expected long-term reward of each action. The optimal action (⭐) has the highest Q-value. Larger gaps indicate stronger confidence in the recommendation.</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-title">Outcome Distribution — Optimal Action</div>', unsafe_allow_html=True)
        best_a = opt_policy[s_key]
        outcomes = P_dyn[s_key][best_a]

        node_labels = [STATE_SHORT[s_key]]
        node_colors = [STATE_COLORS[s_key]]
        sources, targets, values_sk = [], [], []
        idx = 1
        for sn, prob in outcomes.items():
            if prob > 0:
                node_labels.append(f"{STATE_SHORT[sn]}\n({prob:.1%})")
                node_colors.append(STATE_COLORS[sn])
                sources.append(0); targets.append(idx); values_sk.append(prob)
                idx += 1

        fig_sk = go.Figure(go.Sankey(
            node=dict(pad=20, thickness=22, label=node_labels, color=node_colors, line=dict(color="#e2e8f0", width=1)),
            link=dict(source=sources, target=targets, value=values_sk, color=['rgba(37,99,235,0.2)']*len(sources))
        ))
        fig_sk.update_layout(**LIGHT_CHART, height=240, margin=dict(l=10, r=10, t=10, b=10),
                             title=dict(text="Sankey: State Transition Flow Under Optimal Action", font=dict(size=12, color='#1e293b')))
        st.plotly_chart(fig_sk, use_container_width=True)
        st.markdown('<div class="interp-box">📊 <strong>Interpretation:</strong> This Sankey diagram shows where the patient is most likely to transition under the optimal treatment. Thicker flows = higher probability pathways.</div>', unsafe_allow_html=True)

        # Stacked bar
        st.markdown('<div class="section-title">Outcome Probabilities — All Actions</div>', unsafe_allow_html=True)
        fig_bar = go.Figure()
        for sn, sname, sc in [('S0','Healthy','#22c55e'),('S1','Mild Sepsis','#f59e0b'),('S2','Critical','#ef4444'),('S3','Death','#6b7280')]:
            fig_bar.add_trace(go.Bar(name=sname, x=[ACTION_LABELS[a] for a in ACTIONS],
                y=[P_dyn[s_key][a].get(sn, 0) for a in ACTIONS], marker_color=sc))
        fig_bar.update_layout(**LIGHT_CHART, barmode='stack', height=300,
            margin=dict(l=10,r=10,t=40,b=80),
            title=dict(text="Stacked Bar: Outcome Distribution by Treatment Action", font=dict(size=12, color='#1e293b')),
            legend=dict(orientation='h', y=-0.25, font=dict(size=10)),
            xaxis=dict(gridcolor='#e2e8f0', tickangle=-15), yaxis=dict(gridcolor='#e2e8f0', title='Probability', tickformat='.0%'))
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('<div class="interp-box">📊 <strong>Interpretation:</strong> Taller green (Healthy) segments indicate treatments with higher recovery rates. Compare across actions to see why the MDP selects the optimal treatment.</div>', unsafe_allow_html=True)

        # Value function
        st.markdown('<div class="section-title">Value Function V(s) — All States</div>', unsafe_allow_html=True)
        fig_v = go.Figure(go.Bar(x=[STATE_SHORT[s] for s in STATES], y=[state_values[s] for s in STATES],
            marker_color=[STATE_COLORS[s] for s in STATES], text=[f"{state_values[s]:.2f}" for s in STATES],
            textposition='outside', textfont=dict(family='JetBrains Mono', size=11, color='#334155')))
        fig_v.update_layout(**LIGHT_CHART, height=250, margin=dict(l=10,r=10,t=40,b=10),
            title=dict(text="Value Function: Expected Long-Term Reward per State", font=dict(size=12, color='#1e293b')),
            xaxis=dict(gridcolor='#e2e8f0'), yaxis=dict(gridcolor='#e2e8f0', title='Expected Discounted Reward'))
        st.plotly_chart(fig_v, use_container_width=True)
        st.markdown('<div class="interp-box">📊 <strong>Interpretation:</strong> V(s) represents the expected cumulative reward starting from each state. Higher values = better prognosis. The steep drop from Healthy to Death shows the model correctly values patient survival.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════
# TAB 2 — PROBABILITY VERIFICATION
# ═══════════════════════════════════════
with tab_verify:
    st.markdown('<div class="section-title">Full Transition Matrix — Patient-Adjusted Probabilities</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-panel">
        <h4>How to Read This Table</h4>
        <p>Each row gives the probability distribution over next states for a given (state, action) pair.
        Probabilities are adjusted for the patient profile. Each row sums to 1.0000.
        <strong>Highlighted rows (blue)</strong> = optimal policy action. R(s,a) = immediate QALY-calibrated reward.
        Green gradient = higher recovery; Red gradient = higher mortality.</p>
    </div>
    """, unsafe_allow_html=True)

    rows = []
    for s in STATES:
        for a in ACTIONS:
            p = P_dyn[s][a]
            rows.append({'From State': STATE_SHORT[s], 'Action': ACTION_LABELS[a],
                'P→Healthy': p.get('S0', 0.0), 'P→Mild': p.get('S1', 0.0),
                'P→Critical': p.get('S2', 0.0), 'P→Death': p.get('S3', 0.0),
                'Sum': round(sum(p.values()), 4), 'R(s,a)': round(R_dyn[s][a], 2),
                'Optimal?': '⭐ YES' if opt_policy[s] == a else ''})

    df = pd.DataFrame(rows)
    def highlight_optimal(row):
        if row['Optimal?'] == '⭐ YES':
            return ['background-color:#eff6ff; color:#1e40af; font-weight:600'] * len(row)
        return [''] * len(row)

    styled = (df.style.apply(highlight_optimal, axis=1)
        .format({'P→Healthy':'{:.4f}','P→Mild':'{:.4f}','P→Critical':'{:.4f}','P→Death':'{:.4f}','Sum':'{:.4f}','R(s,a)':'{:.2f}'})
        .background_gradient(subset=['P→Healthy'], cmap='Greens', vmin=0, vmax=1)
        .background_gradient(subset=['P→Death'], cmap='Reds', vmin=0, vmax=0.7)
        .set_properties(**{'font-family':'JetBrains Mono','font-size':'12px'}))
    st.dataframe(styled, use_container_width=True, height=560)
    st.markdown('<div class="interp-box">📊 <strong>Interpretation:</strong> This matrix shows all 16 (state, action) pairs. Each row sums to 1.0. Rows with ⭐ are the optimal actions chosen by the MDP solver. Green shading highlights high recovery probabilities; red highlights high mortality risk.</div>', unsafe_allow_html=True)

    # Heatmaps
    st.markdown('<div class="section-title">Transition Probability Heatmaps — By Starting State</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    col_c, col_d = st.columns(2)
    next_labels = ['Healthy', 'Mild Sepsis', 'Critical', 'Death']
    next_keys = ['S0', 'S1', 'S2', 'S3']

    def heatmap_for(state, col):
        with col:
            z = [[P_dyn[state][a].get(sn, 0) for sn in next_keys] for a in ACTIONS]
            fig = go.Figure(go.Heatmap(z=z, x=next_labels, y=[ACTION_LABELS[a] for a in ACTIONS],
                colorscale=[[0,'#f8fafc'],[0.3,'#93c5fd'],[0.6,'#f59e0b'],[1,'#ef4444']],
                text=[[f'{v:.3f}' for v in row] for row in z], texttemplate='%{text}',
                textfont=dict(size=10, family='JetBrains Mono', color='#1e293b'), showscale=True, zmin=0, zmax=1))
            fig.update_layout(**LIGHT_CHART, height=230, margin=dict(l=5,r=5,t=34,b=5),
                title=dict(text=f"From: {STATE_SHORT[state]}", font=dict(size=12, color='#1e293b')),
                xaxis=dict(side='top'))
            st.plotly_chart(fig, use_container_width=True)

    heatmap_for('S0', col_a); heatmap_for('S1', col_b)
    heatmap_for('S2', col_c); heatmap_for('S3', col_d)
    st.markdown('<div class="interp-box">📊 <strong>Interpretation:</strong> Each heatmap shows how a starting state responds to each treatment. Darker red cells = higher probability transitions. Compare across heatmaps to see how acuity level changes treatment effectiveness.</div>', unsafe_allow_html=True)

    # Policy summary cards
    st.markdown('<div class="section-title">Optimal Policy & Value Function — Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, s in zip([c1, c2, c3, c4], STATES):
        color = STATE_COLORS[s]
        with col:
            st.markdown(f"""
            <div style='background:#ffffff;border:1px solid #e2e8f0;border-left:3px solid {color};border-radius:10px;padding:14px;box-shadow:0 1px 3px rgba(0,0,0,0.06);'>
                <div style='font-size:9px;color:#64748b;font-family:JetBrains Mono;letter-spacing:1.5px;text-transform:uppercase;font-weight:600;'>{STATE_SHORT[s]}</div>
                <div style='font-size:18px;font-weight:700;font-family:JetBrains Mono;color:{color};margin:6px 0;'>V = {state_values[s]:.2f}</div>
                <div style='font-size:11px;color:#475569;'>π*: {ACTION_LABELS[opt_policy[s]]}</div>
            </div>
            """, unsafe_allow_html=True)

    # State transition graph
    st.markdown('<div class="section-title">State Transition Graph — Optimal Policy</div>', unsafe_allow_html=True)
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR', bgcolor='white', fontname='Inter')
    dot.attr('node', fontname='Inter', fontsize='11', style='filled', shape='box', margin='0.25')
    dot.attr('edge', fontname='Inter', fontsize='9')
    hex_bg = {'S0':'#dcfce7','S1':'#fef3c7','S2':'#fee2e2','S3':'#f1f5f9'}
    hex_fg = {'S0':'#166534','S1':'#92400e','S2':'#991b1b','S3':'#475569'}
    for s in STATES:
        lbl = f"{STATE_SHORT[s]}\\nπ*: {ACTION_LABELS[opt_policy[s]][:12]}...\\nV={state_values[s]:.1f}"
        dot.node(s, lbl, fillcolor=hex_bg[s], fontcolor=hex_fg[s], color=hex_fg[s])
    for s in STATES:
        for sn, prob in P_dyn[s][opt_policy[s]].items():
            if prob > 0.02:
                dot.edge(s, sn, label=f"{prob:.2f}", color=hex_fg[sn], fontcolor=hex_fg[sn], penwidth=str(1+prob*3))
    st.graphviz_chart(dot, use_container_width=True)
    st.markdown('<div class="interp-box">📊 <strong>Interpretation:</strong> This directed graph shows state transitions under the optimal policy. Edge labels = transition probabilities. Thicker edges = more likely transitions. Self-loops indicate remaining in the same state.</div>', unsafe_allow_html=True)

    # Reward heatmap
    st.markdown('<div class="section-title">Reward R(s,a) Matrix</div>', unsafe_allow_html=True)
    r_z = [[R_dyn[s][a] for a in ACTIONS] for s in STATES]
    fig_r = go.Figure(go.Heatmap(z=r_z, x=[ACTION_LABELS[a] for a in ACTIONS], y=[STATE_SHORT[s] for s in STATES],
        colorscale=[[0,'#fee2e2'],[0.4,'#f1f5f9'],[1,'#dcfce7']],
        text=[[f'{R_dyn[s][a]:.2f}' for a in ACTIONS] for s in STATES], texttemplate='%{text}',
        textfont=dict(size=12, family='JetBrains Mono', color='#1e293b'), showscale=True))
    fig_r.update_layout(**LIGHT_CHART, height=260, margin=dict(l=10,r=10,t=40,b=10),
        title=dict(text="Immediate Reward Matrix R(s,a) — QALY-Calibrated", font=dict(size=12, color='#1e293b')))
    st.plotly_chart(fig_r, use_container_width=True)
    st.markdown('<div class="interp-box">📊 <strong>Interpretation:</strong> R(s,a) is the immediate reward for taking action a in state s. Green = positive outcomes (recovery, stability). Red = negative (mortality risk, treatment burden). Death state has -100 across all actions (absorbing).</div>', unsafe_allow_html=True)

    # Evidence table
    st.markdown('<div class="section-title">Real-World Evidence Calibration Sources</div>', unsafe_allow_html=True)
    evidence = [
        ("Mild Sepsis Mortality (No Tx)", "5.6%", "HCUP 2021"),
        ("Severe Sepsis Mortality", "14.9%", "HCUP 2021"),
        ("Septic Shock Mortality", "34.2%", "HCUP 2021"),
        ("3-hr Bundle Reduction", "~25% relative", "NEJM 2017"),
        ("SICU Rapid Recovery", "63%", "PMC 2020"),
        ("SICU Chronic Illness", "33%", "PMC 2020"),
        ("SICU Early Death (<14d)", "4%", "PMC 2020"),
        ("Annual US Sepsis Cases", "≥1.7M", "CDC 2023"),
        ("Discount Factor γ", "0.90", "PMC 6124941"),
        ("Age 65+ Mortality Mult.", "~1.4×", "HCUP 2021"),
        ("Immunocomp. Mort. Mult.", "~1.8×", "ICU meta-analysis"),
    ]
    ev_rows = "".join(f"<tr><td>{a}</td><td style='color:#2563eb;font-weight:600;'>{b}</td><td style='color:#64748b;font-size:11px;'>{c}</td></tr>" for a, b, c in evidence)
    st.markdown(f"""<table class="styled-table"><thead><tr><th>Parameter</th><th>Value</th><th>Source</th></tr></thead><tbody>{ev_rows}</tbody></table>""", unsafe_allow_html=True)

# ═══════════════════════════════════════
# TAB 3 — MODEL EXPLANATION
# ═══════════════════════════════════════
with tab_explain:
    col_x1, col_x2 = st.columns([1, 1], gap="large")

    with col_x1:
        st.markdown('<div class="section-title">What is this Model?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-panel">
            <h4>Markov Decision Process (MDP) — Overview</h4>
            <p>This engine implements a <strong>finite-state, infinite-horizon MDP</strong> solved via
            <strong>Policy Iteration</strong>. It solves for the globally optimal action across all future
            time steps, maximizing expected discounted rewards — a proxy for long-term patient survival
            and quality-adjusted life years (QALYs).<br><br>
            This mirrors methods used in the <em>AI Clinician</em> (Nature Medicine 2018) for sepsis
            ICU management and hypertension MDPs (PMC 2018).</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">How the Engine Works — Step by Step</div>', unsafe_allow_html=True)
        steps = [
            ("State Space S", "4 clinical states: Healthy (S0), Mild Sepsis (S1), Severe Sepsis/Shock (S2), Death (S3). S3 is absorbing — once entered, no transitions occur."),
            ("Action Space A", "4 treatments: Watchful Waiting (A0), Antibiotic 3-hr Bundle (A1), Antibiotics + Vasopressors (A2), Surgical Intervention (A3)."),
            ("Transition Function T(s,a,s')", "Full probability distribution over next states for each (state, action) pair. Calibrated to HCUP 2021, NEJM 2017, and SICU cohort data. Modified in real-time by patient risk factors."),
            ("Reward Function R(s,a)", "QALY-calibrated: Healthy ~ +10, Mild Sepsis ~ -2 to +3, Critical ~ -5 to -20, Death = -100. Risk penalties subtracted from baseline."),
            ("Policy Iteration Solver", "Alternates Policy Evaluation (Bellman convergence to delta < 1e-9) and Policy Improvement (greedy Q-value maximization). Guaranteed convergence for finite MDPs."),
            ("Bellman Optimality", "V*(s) = max_a Sum T(s,a,s') x [R(s,a) + gamma*V*(s')]. Discount gamma = 0.9 weights near-term survival more heavily."),
            ("Patient-Specific Adjustment", "Composite mortality multiplier (capped 3.5x) from: age (+1%/yr after 50), smoking (1.30x), CVD (1.35x), diabetes (1.20x), immunocompromised (1.80x). Logistic recovery decay."),
            ("Output: Optimal Policy pi*(s)", "For each state, the action maximizing expected reward. Q-values rank all actions — quantifying the cost of suboptimal treatment."),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(f"""
            <div class="flow-step">
                <div class="flow-num">{i}</div>
                <div class="flow-content"><h5>{title}</h5><p>{desc}</p></div>
            </div>
            """, unsafe_allow_html=True)

    with col_x2:
        st.markdown('<div class="section-title">Bellman Equation — Worked Example</div>', unsafe_allow_html=True)
        gamma = 0.9
        s_ex, a_ex = 'S1', 'A1'
        p_ex = P_dyn[s_ex][a_ex]
        r_ex = R_dyn[s_ex][a_ex]
        terms = [(sn, prob, state_values.get(sn, 0), prob * (r_ex + gamma * state_values.get(sn, 0)))
                 for sn, prob in p_ex.items() if prob > 0]
        q_total = sum(t[3] for t in terms)

        st.markdown(f"""
        <div class="info-panel">
            <h4>Q(Mild Sepsis, Antibiotic Bundle 3-hr)</h4>
            <p style='font-family:JetBrains Mono;font-size:12px;line-height:1.8;color:#334155;'>
                Q(S1, A1) = Sum P(s'|S1,A1) x [R(S1,A1) + gamma * V*(s')]<br>
                R(S1,A1) = {r_ex:.2f} &nbsp;·&nbsp; gamma = {gamma}
            </p>
        </div>
        """, unsafe_allow_html=True)

        term_rows = "".join(
            f"<tr><td>{STATE_SHORT[sn]}</td><td>{prob:.4f}</td><td>{r_ex:.2f}</td><td>{v:.2f}</td><td style='font-weight:600;'>{t:.4f}</td></tr>"
            for sn, prob, v, t in terms)
        st.markdown(f"""<table class="styled-table">
            <thead><tr><th>s'</th><th>P(s')</th><th>R(s,a)</th><th>V(s')</th><th>Term</th></tr></thead>
            <tbody>{term_rows}
                <tr style='border-top:2px solid #2563eb;'>
                    <td colspan='4' style='color:#1e40af;font-weight:700;'>Q(S1,A1) Total</td>
                    <td style='color:#1e40af;font-weight:700;'>{q_total:.4f}</td>
                </tr>
            </tbody></table>""", unsafe_allow_html=True)
        st.markdown('<div class="interp-box">📊 <strong>Interpretation:</strong> Each term contributes P(s\') x [R + gamma*V(s\')] to the total Q-value. The sum gives the expected long-term value of this action. The policy selects whichever action yields the highest Q.</div>', unsafe_allow_html=True)

        # Risk modifier table
        st.markdown('<div class="section-title">Risk Modifier Engine — Clinical Basis</div>', unsafe_allow_html=True)
        mods = [
            ("Age (per year after 50)", "+1% mortality/yr", "HCUP 2021"),
            ("Active Smoker", "1.30x mortality", "Respiratory sepsis meta-analysis"),
            ("Cardiovascular Disease", "1.35x mortality", "Framingham / SOFA"),
            ("Diabetes Mellitus", "1.20x mortality", "Diabetic sepsis cohorts"),
            ("Immunocompromised", "1.80x mortality", "Haematologic ICU data"),
            ("Recovery modifier", "Logistic decay, midpoint 72", "SICU + clinical judgment"),
            ("Max composite", "Capped at 3.5x", "Plausibility constraint"),
        ]
        mod_rows = "".join(f"<tr><td>{a}</td><td style='color:#d97706;font-weight:600;'>{b}</td><td style='color:#64748b;font-size:11px;'>{c}</td></tr>" for a, b, c in mods)
        st.markdown(f"""<table class="styled-table"><thead><tr><th>Factor</th><th>Effect</th><th>Basis</th></tr></thead><tbody>{mod_rows}</tbody></table>""", unsafe_allow_html=True)

        # Q-value radar chart
        st.markdown('<div class="section-title">Q-Value Comparison — All States & Actions</div>', unsafe_allow_html=True)
        fig_radar = go.Figure()
        theta_actions = [ACTION_LABELS[a] for a in ACTIONS] + [ACTION_LABELS[ACTIONS[0]]]
        for s in ['S0', 'S1', 'S2']:
            q_all = compute_qvalues(s, ACTIONS, P_dyn, R_dyn, state_values)
            vals = [q_all[a] for a in ACTIONS] + [q_all[ACTIONS[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=theta_actions, fill='toself', name=STATE_SHORT[s],
                line=dict(color=STATE_COLORS[s], width=2)))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, gridcolor='#e2e8f0', color='#64748b'),
                       angularaxis=dict(gridcolor='#e2e8f0', color='#334155'), bgcolor='white'),
            **LIGHT_CHART, legend=dict(orientation='h', y=-0.15, font=dict(size=10)),
            height=320, margin=dict(l=40, r=40, t=30, b=70),
            title=dict(text="Radar: Q-Value Landscape Across States", font=dict(size=12, color='#1e293b')))
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown('<div class="interp-box">📊 <strong>Interpretation:</strong> Each colored polygon shows Q-values for one patient state across all actions. Wider polygons = state with more treatment options. Overlap between states shows where treatments have similar effectiveness.</div>', unsafe_allow_html=True)

        # Limitations
        st.markdown('<div class="section-title">Model Assumptions & Limitations</div>', unsafe_allow_html=True)
        lims = [
            "Markov property: future depends only on present state, not full history.",
            "Probabilities are population-level averages. Individual variation not captured.",
            "4-state discretization simplifies continuous severity (SOFA 0-24).",
            "Death is absorbing — no palliative care trajectory differentiation.",
            "FOR EDUCATIONAL USE ONLY. Never substitute for clinical judgment.",
        ]
        for l in lims:
            st.markdown(f'<div class="warning-box">⚠️ {l}</div>', unsafe_allow_html=True)

        # Solver parameters
        st.markdown('<div class="section-title">Solver Parameters</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-panel"><p>
            <strong>Algorithm:</strong> Policy Iteration (Howard, 1960)<br>
            <strong>Discount factor:</strong> 0.90 (clinical standard)<br>
            <strong>Convergence:</strong> 1e-9<br>
            <strong>States:</strong> 4 &nbsp;·&nbsp; <strong>Actions:</strong> 4 &nbsp;·&nbsp; <strong>Pairs:</strong> 16<br>
            <strong>Guarantee:</strong> Global optimality in finite MDPs (Puterman, 1994)
        </p></div>
        """, unsafe_allow_html=True)
