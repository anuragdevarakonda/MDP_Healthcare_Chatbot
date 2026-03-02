import streamlit as st
import numpy as np
import graphviz
from fpdf import FPDF

# --- APP CONFIG ---
st.set_page_config(page_title="Age-Aware Health MDP", layout="wide", page_icon="⚖️")

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

# --- NAVIGATION ---
st.sidebar.title("🩺 Strategic Controls")
page = st.sidebar.radio("Navigation", ["1. Model Logic", "2. Personalized Live Simulation"])

# --- LIVE SIMULATION PAGE ---
if page == "2. Personalized Live Simulation":
    st.title("⚖️ Age-Adjusted Decision Engine")
    
    # --- DYNAMIC INPUTS ---
    st.subheader("👤 Step 1: Comprehensive Patient Profile")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Patient Age", min_value=1, max_value=100, value=75)
    with c2:
        is_smoker = st.toggle("Active Smoker", value=True)
    with c3:
        pre_existing = st.toggle("Pre-existing Condition", value=True)
    with c4:
        adherence = st.select_slider("Adherence", options=["Low", "Med", "High"], value="High")

    # --- DYNAMIC MATRIX MODIFICATION ---
    # Start with base logic
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

    # AGE LOGIC: As age > 65, surgery success (P_S2_A2_S0) drops and self-loop risk increases
    if age > 65:
        age_factor = (age - 65) / 100
        # Surgery risk: P(Critical -> Healthy) decreases
        P_dyn['S2']['A2']['S0'] = max(0.2, P_dyn['S2']['A2']['S0'] - age_factor)
        # Risk of staying Critical increases
        P_dyn['S2']['A2']['S2'] = min(0.8, P_dyn['S2']['A2']['S2'] + age_factor)
        st.warning(f"🧓 **Age Modifier:** Surgery recovery chance reduced to {P_dyn['S2']['A2']['S0']*100:.0f}%.")

    if is_smoker:
        # Increase transition from Sick to Critical
        P_dyn['S1']['A0']['S2'] = min(0.9, P_dyn['S1']['A0']['S2'] + 0.2)
        st.error("🚬 **Smoker Modifier:** Risk of Sick → Critical increased.")

    if pre_existing:
        # Heavily penalize the 'Critical' state
        for a in actions: R_dyn['S2'][a] -= 100
        st.error("❤️ **Pre-existing Modifier:** Critical state penalty tripled.")

    # Recalculate
    dyn_policy, dyn_values = policy_iteration(states, actions, P_dyn, R_dyn, 0.9)

    st.divider()
    st.subheader("🤖 Step 2: Strategic Recommendation")
    
    current_state = st.selectbox("Current Condition", ["Critical", "Sick", "Healthy"])
    s_key = [k for k, v in state_labels.items() if v == current_state][0]
    
    if st.button("Generate Decision"):
        best_a = dyn_policy[s_key]
        
        col_txt, col_graph = st.columns([1, 1.5])
        with col_txt:
            st.chat_message("assistant").write(f"**Chatbot Recommends: {action_labels[best_a]}**")
            st.write(f"**Patient Value Index:** {dyn_values[s_key]:.2f}")
            
            # THE LOGIC EXPLANATION
            if age > 75 and is_smoker and current_state == "Critical":
                if best_a == 'A1':
                    st.write("🔎 **Logic:** Due to advanced age and smoking history, Surgery is deemed too high-risk. I recommend Medication to stabilize the state rather than invasive procedures.")
                else:
                    st.write("🔎 **Logic:** Despite age risks, Surgery remains the only viable path to avoid the extreme penalties of a Critical state.")
            else:
                st.write("🔎 **Logic:** Standard risk-weighted optimization applied.")

        with col_graph:
            dot = graphviz.Digraph()
            dot.attr(rankdir='LR')
            for s in states:
                is_curr = (s == s_key)
                dot.node(s, state_labels[s], style='filled', fillcolor=colors[s] if is_curr else "#f0f0f0")
            for sn, prob in P_dyn[s_key][best_a].items():
                if prob > 0:
                    dot.edge(s_key, sn, label=f"{prob*100:.0f}%", color="#3498db", penwidth="3")
            st.graphviz_chart(dot)
