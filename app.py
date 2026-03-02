import streamlit as st
import numpy as np
import graphviz
from fpdf import FPDF
import pandas as pd

# --- APP CONFIG ---
st.set_page_config(page_title="Personalized Health MDP", layout="wide", page_icon="🩺")

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

# --- SIDEBAR NAV ---
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to", ["1. Strategy Analysis", "2. Live Chatbot Simulation"])
st.sidebar.divider()
gamma = st.sidebar.slider("Discount Factor", 0.0, 1.0, 0.9, help="Long-term vs Immediate focus.")

# --- PAGE 1: STRATEGY ANALYSIS ---
if page == "1. Strategy Analysis":
    st.title("📊 Global Strategy & All Transitions")
    st.markdown("This page shows the 'Rulebook'—every possible outcome for every action.")
    
    # Base Matrices (Static for reference)
    P_base = {
        'S0': {'A0': {'S0': 0.8, 'S1': 0.2}, 'A1': {'S0': 0.9, 'S1': 0.1}, 'A2': {'S0': 0.95, 'S2': 0.05}},
        'S1': {'A0': {'S1': 0.6, 'S2': 0.4}, 'A1': {'S0': 0.6, 'S1': 0.3, 'S2': 0.1}, 'A2': {'S0': 0.8, 'S2': 0.2}},
        'S2': {'A0': {'S2': 1.0}, 'A1': {'S1': 0.4, 'S2': 0.6}, 'A2': {'S0': 0.7, 'S1': 0.2, 'S2': 0.1}}
    }
    
    col1, col2 = st.columns([2, 1])
    with col1:
        dot_all = graphviz.Digraph()
        dot_all.attr(rankdir='LR', size='12')
        for s in states:
            dot_all.node(s, state_labels[s], style='filled', fillcolor=colors[s])
            for a in actions:
                for sn, prob in P_base[s][a].items():
                    dot_all.edge(s, sn, label=f"{a}({prob})", color="#95a5a6", fontsize="10")
        st.graphviz_chart(dot_all)
    
    with col2:
        st.info("**How to read this:**\n- **Nodes:** Current patient state.\n- **Arrows:** Potential transitions.\n- **Labels:** Probability of that outcome for a specific action (A0, A1, A2).")

# --- PAGE 2: LIVE SIMULATION (DYNAMIC) ---
else:
    st.title("💬 Personalized Decision Engine")
    
    # --- DYNAMIC INPUTS ---
    st.subheader("👤 Step 1: Define Patient Context")
    c1, c2, c3 = st.columns(3)
    with c1:
        is_smoker = st.toggle("Active Smoker", help="Increases risk of health degradation.")
    with c2:
        pre_existing = st.toggle("Pre-existing Condition", help="Increases penalty for Critical state.")
    with c3:
        adherence = st.select_slider("Medication Adherence", options=["Low", "Medium", "High"], value="High")

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

    # Apply Modifiers
    if is_smoker:
        # Increase risk of degradation from Healthy to Sick
        P_dyn['S0']['A0']['S1'] += 0.1
        P_dyn['S0']['A0']['S0'] -= 0.1
        st.warning("⚠️ **Smoker Logic Active:** Risk of falling from Healthy to Sick increased by 10%.")

    if pre_existing:
        # Increase the penalty for Critical state across all actions
        for a in actions:
            R_dyn['S2'][a] -= 50
        st.error("⚠️ **Pre-existing Condition Active:** Critical state penalty doubled (-100).")

    if adherence == "Low":
        # Medication (A1) becomes much less effective
        P_dyn['S1']['A1']['S0'] = 0.3 # Reduced from 0.6
        P_dyn['S1']['A1']['S1'] = 0.7
        st.info("ℹ️ **Low Adherence Active:** Medication effectiveness reduced from 60% to 30%.")

    # Recalculate Policy
    dyn_policy, dyn_values = policy_iteration(states, actions, P_dyn, R_dyn, gamma)

    st.divider()
    st.subheader("🤖 Step 2: Chatbot Recommendation")
    
    current_state = st.selectbox("What is the patient's current status?", ["Healthy", "Sick", "Critical"])
    s_key = [k for k, v in state_labels.items() if v == current_state][0]
    
    if st.button("Consult Chatbot"):
        best_a = dyn_policy[s_key]
        
        sc1, sc2 = st.columns([1, 1.5])
        with sc1:
            st.chat_message("assistant").write(f"**Recommendation:** I advise **{action_labels[best_a]}**.")
            st.write(f"**Personalized Value Score:** {dyn_values[s_key]:.2f}")
            st.write("---")
            st.write("**Decision Logic:**")
            if pre_existing and best_a == 'A2' and s_key == 'S1':
                st.write("Because of your pre-existing condition, I am recommending Surgery early to avoid the high risk of a Critical state.")
            elif is_smoker:
                st.write("Due to smoking status, I am prioritizing actions that aggressively return you to 'Healthy' status.")
            else:
                st.write("Standard maintenance logic applied.")

        with sc2:
            # Visualize the EFFECT of the context on the choice
            dot_sim = graphviz.Digraph()
            dot_sim.attr(rankdir='LR')
            for s in states:
                is_curr = (s == s_key)
                dot_sim.node(s, state_labels[s], style='filled', 
                             fillcolor=colors[s] if is_curr else "#f0f0f0", 
                             penwidth="4" if is_curr else "1")
            
            # Show the probabilities of the CHOSEN action for this specific patient
            for sn, prob in P_dyn[s_key][best_a].items():
                if prob > 0:
                    dot_sim.edge(s_key, sn, label=f"{prob*100}% Outcome", color="#3498db", penwidth="3")
            
            st.graphviz_chart(dot_sim)
