import streamlit as st
import numpy as np
import graphviz
from fpdf import FPDF
import pandas as pd

# --- APP CONFIG ---
st.set_page_config(page_title="Healthcare MDP Dashboard", layout="wide", page_icon="🩺")

# --- MDP LOGIC ---
def policy_iteration(states, actions, P, R, gamma=0.9):
    policy = {s: actions[0] for s in states}
    V = {s: 0 for s in states}
    # Evaluation
    for _ in range(100):
        while True:
            delta = 0
            for s in states:
                v = V[s]
                a = policy[s]
                V[s] = sum([prob * (R[s][a] + gamma * V.get(sn, 0)) for sn, prob in P[s][a].items()])
                delta = max(delta, abs(v - V[s]))
            if delta < 1e-6: break
        # Improvement
        stable = True
        for s in states:
            old_a = policy[s]
            policy[s] = max(actions, key=lambda a: sum([prob * (R[s][a] + gamma * V.get(sn, 0)) for sn, prob in P[s][a].items()]))
            if old_a != policy[s]: stable = False
        if stable: break
    return policy, V

# --- SIDEBAR & INPUTS ---
st.sidebar.title("⚙️ Model Parameters")
gamma = st.sidebar.slider("Discount Factor (Gamma)", 0.0, 1.0, 0.9, help="Significance of future rewards.")

st.sidebar.subheader("💰 Rewards Configuration")
r_h = st.sidebar.number_input("Reward: Healthy", value=10)
r_s = st.sidebar.number_input("Reward: Sick", value=1)
r_c = st.sidebar.number_input("Penalty: Critical", value=-20)

# --- MAIN UI ---
st.title("🩺 Healthcare Chatbot MDP Decision Engine")
st.markdown("""
This dashboard implements a **Markov Decision Process** to determine the optimal treatment strategy for a healthcare chatbot. 
Adjust the probabilities and rewards to see how the 'Golden Path' shifts.
""")

# --- DATA SETUP ---
states = ['S0', 'S1', 'S2']
state_labels = {'S0': 'Healthy', 'S1': 'Sick', 'S2': 'Critical'}
actions = ['A0', 'A1', 'A2']
action_labels = {'A0': 'No Treatment', 'A1': 'Medication', 'A2': 'Surgery'}
colors = {'S0': '#c7e9c0', 'S1': '#fee391', 'S2': '#fc9272'}

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📊 Probability Tuning")
    p_s1_a1_s0 = st.slider("Sick --Medication--> Healthy", 0.0, 1.0, 0.6)
    p_s1_a1_s2 = st.slider("Sick --Medication--> Critical", 0.0, 1.0 - p_s1_a1_s0, 0.1)
    p_s1_a1_s1 = round(1.0 - p_s1_a1_s0 - p_s1_a1_s2, 2)
    st.caption(f"Calculated Sick --Medication--> Sick: {p_s1_a1_s1}")

    # Probability Matrix
    P = {
        'S0': {'A0': {'S0': 0.8, 'S1': 0.2}, 'A1': {'S0': 0.9, 'S1': 0.1}, 'A2': {'S0': 0.95, 'S2': 0.05}},
        'S1': {'A0': {'S1': 0.6, 'S2': 0.4}, 'A1': {'S0': p_s1_a1_s0, 'S1': p_s1_a1_s1, 'S2': p_s1_a1_s2}, 'A2': {'S0': 0.8, 'S2': 0.2}},
        'S2': {'A0': {'S2': 1.0}, 'A1': {'S1': 0.4, 'S2': 0.6}, 'A2': {'S0': 0.7, 'S1': 0.2, 'S2': 0.1}}
    }
    R = {
        'S0': {'A0': r_h, 'A1': r_h-2, 'A2': r_h-5},
        'S1': {'A0': r_s-3, 'A1': r_s, 'A2': r_s-6},
        'S2': {'A0': r_c, 'A1': r_c+10, 'A2': r_c+5}
    }

opt_policy, state_values = policy_iteration(states, actions, P, R, gamma)

with col2:
    st.subheader("🎯 Optimal Policy Visualization")
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR')
    for s in states:
        best_a = opt_policy[s]
        dot.node(s, f"{state_labels[s]}\nUSE: {action_labels[best_a]}\nValue: {state_values[s]:.1f}", 
                 style='filled', fillcolor=colors[s], shape='ellipse')
        for sn, prob in P[s][best_a].items():
            if prob > 0:
                dot.edge(s, sn, label=f"p={prob}")
    st.graphviz_chart(dot)

# --- PDF EXPORT ---
def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    pdf.cell(0, 10, "MDP Strategic Report", 0, 1, 'C')
    pdf.set_font("Helvetica", '', 12)
    pdf.ln(10)
    for s in states:
        pdf.cell(0, 10, f"- {state_labels[s]}: Recommended {action_labels[opt_policy[s]]}", 0, 1)
    return pdf.output(dest='S').encode('latin-1')

st.divider()
st.download_button("📥 Download Executive Summary", data=generate_pdf(), file_name="strategy_report.pdf", mime="application/pdf")