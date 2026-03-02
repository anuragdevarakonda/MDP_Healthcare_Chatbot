import streamlit as st
import numpy as np
import graphviz
from fpdf import FPDF
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="Healthcare MDP Engine", layout="wide", page_icon="🩺")

# --- SHARED DATA & LOGIC ---
states = ['S0', 'S1', 'S2']
state_labels = {'S0': 'Healthy', 'S1': 'Sick', 'S2': 'Critical'}
actions = ['A0', 'A1', 'A2']
action_labels = {'A0': 'No Treatment', 'A1': 'Medication', 'A2': 'Surgery'}
colors = {'S0': '#c7e9c0', 'S1': '#fee391', 'S2': '#fc9272'}

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

# --- SIDEBAR NAV ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Strategy Analysis", "Live Chatbot Simulation"])

# --- SHARED INPUTS ---
st.sidebar.divider()
st.sidebar.subheader("Model Weights")
gamma = st.sidebar.slider("Discount Factor", 0.0, 1.0, 0.9)
r_c = st.sidebar.slider("Penalty for Critical State", -100, -10, -30)

# Probability Matrix Setup
P = {
    'S0': {'A0': {'S0': 0.8, 'S1': 0.2}, 'A1': {'S0': 0.9, 'S1': 0.1}, 'A2': {'S0': 0.95, 'S2': 0.05}},
    'S1': {'A0': {'S1': 0.6, 'S2': 0.4}, 'A1': {'S0': 0.6, 'S1': 0.3, 'S2': 0.1}, 'A2': {'S0': 0.8, 'S2': 0.2}},
    'S2': {'A0': {'S2': 1.0}, 'A1': {'S1': 0.4, 'S2': 0.6}, 'A2': {'S0': 0.7, 'S1': 0.2, 'S2': 0.1}}
}
R = {
    'S0': {'A0': 10, 'A1': 8, 'A2': 5},
    'S1': {'A0': -5, 'A1': 2, 'A2': -5},
    'S2': {'A0': r_c, 'A1': r_c + 15, 'A2': r_c + 10}
}

opt_policy, state_values = policy_iteration(states, actions, P, R, gamma)

# --- PAGE 1: STRATEGY ANALYSIS ---
if page == "Strategy Analysis":
    st.title("📊 Strategy & Probability Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. All Possible Transitions")
        dot_all = graphviz.Digraph()
        dot_all.attr(rankdir='LR')
        for s in states:
            dot_all.node(s, state_labels[s], style='filled', fillcolor=colors[s])
            for a in actions:
                for sn, prob in P[s][a].items():
                    dot_all.edge(s, sn, label=f"{a}({prob})", color="#95a5a6", fontsize="10")
        st.graphviz_chart(dot_all)
        st.info("**Interpretation:** This view shows the 'Uncertainty Map'. It displays every possible outcome of every action. Use this to verify that the system is properly defined (all probabilities sum to 1.0).")

    with col2:
        st.subheader("2. The Optimized 'Golden Path'")
        dot_opt = graphviz.Digraph()
        dot_opt.attr(rankdir='LR')
        for s in states:
            best_a = opt_policy[s]
            dot_opt.node(s, f"{state_labels[s]}\nUse: {action_labels[best_a]}", style='filled', fillcolor=colors[s], penwidth="3")
            for sn, prob in P[s][best_a].items():
                dot_opt.edge(s, sn, label=f"p={prob}", color="#2ecc71" if sn=='S0' else "#e74c3c")
        st.graphviz_chart(dot_opt)
        st.success("**Interpretation:** This is the 'Intelligence Layer'. Based on the rewards, the chatbot has decided that these specific actions maximize long-term patient health.")

# --- PAGE 2: LIVE SIMULATION ---
else:
    st.title("💬 Live Chatbot Decision Simulation")
    st.markdown("Enter a patient's current status to see the chatbot's real-time reasoning.")
    
    user_state = st.selectbox("Select Patient Condition:", ["Healthy", "Sick", "Critical"])
    s_key = [k for k, v in state_labels.items() if v == user_state][0]
    
    if st.button("Generate Chatbot Recommendation"):
        st.divider()
        best_action = opt_policy[s_key]
        
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.chat_message("assistant").write(f"**Chatbot Recommendation:** Based on your current status ({user_state}), I recommend **{action_labels[best_action]}**.")
            st.write(f"**Mathematical Reasoning:** By choosing {best_action}, we achieve a state value of **{state_values[s_key]:.2f}**. This action maximizes the probability of returning to or staying in the 'Healthy' state.")
        
        with c2:
            # Highlight only the current decision visually
            sim_dot = graphviz.Digraph()
            sim_dot.attr(rankdir='LR')
            for s in states:
                is_active = (s == s_key)
                sim_dot.node(s, state_labels[s], style='filled', fillcolor=colors[s] if is_active else "#f0f0f0", penwidth="4" if is_active else "1")
            
            for sn, prob in P[s_key][best_action].items():
                sim_dot.edge(s_key, sn, label=f"Result: {prob*100}%", color="#3498db", penwidth="3")
            
            st.graphviz_chart(sim_dot)
            st.caption(f"Visualizing outcome probabilities for {action_labels[best_action]} starting from {user_state}.")
