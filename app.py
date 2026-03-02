import streamlit as st
import numpy as np
import graphviz
import pandas as pd

# --- APP CONFIG ---
st.set_page_config(page_title="Realistic Health MDP", layout="wide", page_icon="⚖️")

# --- CORE MDP ENGINE ---
def policy_iteration(states, actions, P, R, gamma=0.9):
    policy = {s: actions[0] for s in states}
    V = {s: 0 for s in states}
    for _ in range(100):
        while True:
            delta = 0
            for s in states:
                v = V[s]
                a = policy[s]
                V[s] = sum([prob * (R[s][a] + gamma * V.get(sn, 0)) for sn, prob in P[s][a].items()])
                delta = max(delta, abs(v - V[s]))
            if delta < 1e-6: break
        stable = True
        for s in states:
            old_a = policy[s]
            policy[s] = max(actions, key=lambda a: sum([prob * (R[s][a] + gamma * V.get(sn, 0)) for sn, prob in P[s][a].items()]))
            if old_a != policy[s]: stable = False
        if stable: break
    return policy, V

# --- DATA DEFINITIONS ---
states = ['S0', 'S1', 'S2']
state_labels = {'S0': 'Healthy', 'S1': 'Sick', 'S2': 'Critical'}
actions = ['A0', 'A1', 'A2']
action_labels = {'A0': 'No Treatment', 'A1': 'Medication', 'A2': 'Surgery'}
colors = {'S0': '#c7e9c0', 'S1': '#fee391', 'S2': '#fc9272'}

# --- SIDEBAR & INPUTS ---
st.sidebar.title("🩺 Patient Bio-Profile")
age = st.sidebar.slider("Patient Age", 1, 100, 45)
is_smoker = st.sidebar.toggle("Active Smoker", value=False)
pre_existing = st.sidebar.toggle("Comorbidities", value=False)

# --- REALISTIC DYNAMIC CALCULATION ---

# 1. Base Probabilities
P_dyn = {
    'S0': {'A0': {'S0': 0.8, 'S1': 0.2, 'S2': 0.0}, 'A1': {'S0': 0.9, 'S1': 0.1, 'S2': 0.0}, 'A2': {'S0': 0.95, 'S1': 0.0, 'S2': 0.05}},
    'S1': {'A0': {'S1': 0.6, 'S2': 0.4}, 'A1': {'S0': 0.6, 'S1': 0.3, 'S2': 0.1}, 'A2': {'S0': 0.8, 'S2': 0.2}},
    'S2': {'A0': {'S2': 1.0}, 'A1': {'S0': 0.0, 'S1': 0.4, 'S2': 0.6}, 'A2': {'S0': 0.7, 'S1': 0.2, 'S2': 0.1}}
}

# 2. Base Rewards
R_dyn = {
    'S0': {'A0': 10, 'A1': 8, 'A2': 5},
    'S1': {'A0': -5, 'A1': 2, 'A2': -5},
    'S2': {'A0': -50, 'A1': -35, 'A2': -40}
}

# --- THE CONTINUOUS AGING ENGINE ---
# Effective Age Calculation
eff_age = age + (15 if is_smoker else 0) + (10 if pre_existing else 0)

# A. Recovery Decay (Surgery Effectiveness S2 -> S0)
# Uses a logistic-style decay: drops sharply as effective age exceeds 70
recovery_modifier = 1 / (1 + np.exp((eff_age - 75) / 10))
P_dyn['S2']['A2']['S0'] = round(0.8 * recovery_modifier, 2)
P_dyn['S2']['A2']['S2'] = round(1.0 - P_dyn['S2']['A2']['S0'] - P_dyn['S2']['A2']['S1'], 2)

# B. Surgical Penalty Scaling (R_S2_A2)
# The "cost" of surgery increases linearly with age due to frailty
frailty_penalty = eff_age * 1.5
R_dyn['S2']['A2'] -= frailty_penalty

# C. Reward for Youth (Healthy State)
# Higher reward for young patients staying healthy (Life-years value)
R_dyn['S0']['A0'] += max(0, (100 - age) / 5)

# --- SOLVE ---
opt_policy, state_values = policy_iteration(states, actions, P_dyn, R_dyn, 0.9)

# --- UI DISPLAY ---
st.title("🩺 Advanced Bio-Logical MDP Dashboard")
st.write(f"**Current Effective Biological Age:** {eff_age} years")

tab1, tab2 = st.tabs(["📊 Probability Verification", "🤖 Live Simulation"])

with tab1:
    st.subheader("State Transition Matrix (Continuous Age-Adjusted)")
    rows = []
    for s in states:
        for a in actions:
            p_vals = P_dyn[s][a]
            rows.append({
                "Context": f"{state_labels[s]} + {action_labels[a]}",
                "To Healthy": p_vals.get('S0', 0),
                "To Sick": p_vals.get('S1', 0),
                "To Critical": p_vals.get('S2', 0),
                "Row Sum": round(sum(p_vals.values()), 2)
            })
    st.table(pd.DataFrame(rows))
    
    # Digraph for Logic
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR')
    for s in states:
        dot.node(s, state_labels[s], style='filled', fillcolor=colors[s])
        for a in actions:
            for sn, prob in P_dyn[s][a].items():
                if prob > 0.1: # Only show significant transitions
                    dot.edge(s, sn, label=f"{a}({prob})", color="#95a5a6")
    st.graphviz_chart(dot)

with tab2:
    current_state = st.selectbox("Current Condition", ["Healthy", "Sick", "Critical"])
    s_key = [k for k, v in state_labels.items() if v == current_state][0]
    
    if st.button("Consult AI"):
        best_a = opt_policy[s_key]
        st.subheader(f"Chatbot Suggests: {action_labels[best_a]}")
        st.metric("State Value (Expected Outcome)", round(state_values[s_key], 2))
        
        # LOGIC EXPLANATION
        if eff_age > 85 and best_a != 'A2':
            st.info("💡 **Geriatric Strategy:** Because the patient's biological age is very high, the risk of surgical complications outweighs the benefit. The AI is prioritizing stabilization (Medication) over high-risk intervention.")
        elif age < 10:
            st.success("💡 **Pediatric Strategy:** High recovery potential and long-life value justify preventative maintenance or surgery to ensure total return to health.")
