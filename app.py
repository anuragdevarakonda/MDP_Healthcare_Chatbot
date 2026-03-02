import streamlit as st
import numpy as np
import graphviz
from fpdf import FPDF

# --- APP CONFIG ---
st.set_page_config(page_title="Advanced Health MDP", layout="wide", page_icon="⚖️")

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

# --- GLOBAL NAVIGATION & INPUTS ---
st.sidebar.title("🩺 Control Center")
page = st.sidebar.radio("Navigation", ["1. Global Model Logic", "2. Personalized Live Simulation"])
st.sidebar.divider()

st.sidebar.subheader("👤 Base Patient Profile")
age = st.sidebar.number_input("Patient Age", min_value=1, max_value=100, value=75)
is_smoker = st.sidebar.toggle("Active Smoker", value=True)
pre_existing = st.sidebar.toggle("Pre-existing Condition", value=True)

# --- DYNAMIC MATRIX MODIFICATION LOGIC ---
# This block runs for BOTH pages to ensure consistency
P_dyn = {
    'S0': {'A0': {'S0': 0.8, 'S1': 0.2}, 'A1': {'S0': 0.9, 'S1': 0.1}, 'A2': {'S0': 0.95, 'S2': 0.05}},
    'S1': {'A0': {'S1': 0.6, 'S2': 0.4}, 'A1': {'S0': 0.6, 'S1': 0.3, 'S2': 0.1}, 'A2': {'S0': 0.8, 'S2': 0.2}},
    'S2': {'A0': {'S2': 1.0}, 'A1': {'S1': 0.4, 'S2': 0.6}, 'A2': {'S0': 0.7, 'S1': 0.2, 'S2': 0.1}}
}
R_dyn = {
    'S0': {'A0': 10, 'A1': 8, 'A2': 5},
    'S1': {'A0': -5, 'A1': 2, 'A2': -5},
    'S2': {'A0': -50, 'A1': -35, 'A2': -40}
}

# Apply Modifiers
if age > 65:
    age_factor = (age - 65) / 100
    P_dyn['S2']['A2']['S0'] = round(max(0.2, P_dyn['S2']['A2']['S0'] - age_factor), 2)
    P_dyn['S2']['A2']['S2'] = round(min(0.8, P_dyn['S2']['A2']['S2'] + age_factor), 2)

if is_smoker:
    P_dyn['S1']['A0']['S2'] = round(min(0.9, P_dyn['S1']['A0']['S2'] + 0.2), 2)
    P_dyn['S1']['A0']['S1'] = round(1.0 - P_dyn['S1']['A0']['S2'], 2)

if pre_existing:
    for a in actions: R_dyn['S2'][a] -= 100

opt_policy, state_values = policy_iteration(states, actions, P_dyn, R_dyn, 0.9)

# --- PAGE 1: GLOBAL MODEL LOGIC ---
if page == "1. Global Model Logic":
    st.title("📊 Global Model Transition Logic")
    st.markdown("This page visualizes the 'Rules of the World' based on the profile you set in the sidebar.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("All State Transitions")
        dot_all = graphviz.Digraph()
        dot_all.attr(rankdir='LR', size='10')
        for s in states:
            dot_all.node(s, state_labels[s], style='filled', fillcolor=colors[s])
            for a in actions:
                for sn, prob in P_dyn[s][a].items():
                    dot_all.edge(s, sn, label=f"{a}({prob})", color="#95a5a6", fontsize="10")
        st.graphviz_chart(dot_all)

    with col2:
        st.subheader("Current Policy Weights")
        st.write(f"**Age Modifier:** Active (Surgery Effectiveness: {P_dyn['S2']['A2']['S0']*100:.0f}%)")
        st.write(f"**Smoker Modifier:** {'Active' if is_smoker else 'Inactive'}")
        st.write(f"**Critical Penalty:** {R_dyn['S2']['A0']}")
        st.info("This graph displays every possible outcome of every action under the current risk profile.")

# --- PAGE 2: LIVE SIMULATION ---
else:
    st.title("💬 Personalized Decision Simulation")
    
    current_state = st.selectbox("Select Patient Condition", ["Healthy", "Sick", "Critical"])
    s_key = [k for k, v in state_labels.items() if v == current_state][0]
    
    if st.button("Consult AI"):
        best_a = opt_policy[s_key]
        st.divider()
        
        c_txt, c_graph = st.columns([1, 1.5])
        with c_txt:
            st.chat_message("assistant").write(f"**Recommendation: {action_labels[best_a]}**")
            st.write(f"**Patient Health Value:** {state_values[s_key]:.2f}")
            st.write("---")
            if age > 75 and is_smoker and current_state == "Critical":
                st.write("**Reasoning:** Given high age and smoking history, the risk of surgery complications is significant. However, the penalty for staying critical is so high that the AI must choose between palliative care or high-risk intervention.")
        
        with c_graph:
            dot_sim = graphviz.Digraph()
            dot_sim.attr(rankdir='LR')
            for s in states:
                is_curr = (s == s_key)
                dot_sim.node(s, state_labels[s], style='filled', fillcolor=colors[s] if is_curr else "#f0f0f0")
            for sn, prob in P_dyn[s_key][best_a].items():
                dot_sim.edge(s_key, sn, label=f"{prob*100:.0f}% Outcome", color="#3498db", penwidth="3")
            st.graphviz_chart(dot_sim)
