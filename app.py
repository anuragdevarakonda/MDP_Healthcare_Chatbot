import streamlit as st
import numpy as np
import graphviz
import pandas as pd

# --- APP CONFIG ---
st.set_page_config(page_title="Personalized Health AI", layout="wide", page_icon="🩺")

# --- MDP ENGINE ---
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
age = st.sidebar.number_input("Enter Patient Age", min_value=1, max_value=110, value=45, step=1)
is_smoker = st.sidebar.toggle("Active Smoker", value=False)
pre_existing = st.sidebar.toggle("Comorbidities", value=False)

# --- DYNAMIC CALCULATION ENGINE ---
P_dyn = {
    'S0': {'A0': {'S0': 0.8, 'S1': 0.2, 'S2': 0.0}, 'A1': {'S0': 0.9, 'S1': 0.1, 'S2': 0.0}, 'A2': {'S0': 0.95, 'S1': 0.0, 'S2': 0.05}},
    'S1': {'A0': {'S1': 0.6, 'S2': 0.4}, 'A1': {'S0': 0.6, 'S1': 0.3, 'S2': 0.1}, 'A2': {'S0': 0.8, 'S2': 0.2}},
    'S2': {'A0': {'S2': 1.0}, 'A1': {'S0': 0.0, 'S1': 0.4, 'S2': 0.6}, 'A2': {'S0': 0.7, 'S1': 0.2, 'S2': 0.1}}
}
R_dyn = {
    'S0': {'A0': 10, 'A1': 8, 'A2': 5},
    'S1': {'A0': -5, 'A1': 2, 'A2': -5},
    'S2': {'A0': -50, 'A1': -35, 'A2': -40}
}

# The engine still uses 'eff_age' for the math, but we won't show the +10/15 logic to the user
eff_age = age + (15 if is_smoker else 0) + (10 if pre_existing else 0)
recovery_modifier = 1 / (1 + np.exp((eff_age - 75) / 10))
P_dyn['S2']['A2']['S0'] = round(0.8 * recovery_modifier, 2)
P_dyn['S2']['A2']['S2'] = round(1.0 - P_dyn['S2']['A2']['S0'] - P_dyn['S2']['A2']['S1'], 2)
R_dyn['S2']['A2'] -= (eff_age * 1.5)

opt_policy, state_values = policy_iteration(states, actions, P_dyn, R_dyn, 0.9)

# --- DASHBOARD ---
st.title("🩺 Real-Time Patient Recommendation Engine")

tab1, tab2 = st.tabs(["🤖 Live Simulation & Reasoning", "📊 Probability Verification"])

with tab1:
    col_input, col_viz = st.columns([1, 1.5])
    
    with col_input:
        current_status = st.selectbox("Current Patient Status:", ["Healthy", "Sick", "Critical"])
        s_key = [k for k, v in state_labels.items() if v == current_status][0]
        
        if st.button("Generate Recommendation"):
            best_a = opt_policy[s_key]
            st.divider()
            st.chat_message("assistant").write(f"**Recommendation: {action_labels[best_a]}**")
            
            # --- INSIGHTS WITHOUT CALCULATIONS ---
            insights = []
            if pre_existing: insights.append("comorbidities")
            if is_smoker: insights.append("smoking history")
            if age > 70: insights.append("advanced age")
            
            reasoning_msg = f"This recommendation prioritizes the highest statistical probability of recovery."
            if insights:
                reasoning_msg = f"Given your {', '.join(insights)}, this action provides the most stable path to a 'Healthy' state while minimizing high-risk complications."
            
            st.write(f"**Chatbot Insight:** {reasoning_msg}")
            
            # Visual Distribution
            outcomes = P_dyn[s_key][best_a]
            chart_data = pd.DataFrame({
                'Outcome': [state_labels[k] for k in outcomes.keys()],
                'Probability (%)': [v * 100 for v in outcomes.values()]
            })
            st.bar_chart(chart_data, x='Outcome', y='Probability (%)', color="#2980b9")

    with col_viz:
        st.subheader("Decision Outcome Visualization")
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')
        for s in states:
            dot.node(s, state_labels[s], style='filled', fillcolor=colors[s] if s == s_key else "#f0f0f0")
        
        for sn, prob in P_dyn[s_key][opt_policy[s_key]].items():
            if prob > 0:
                dot.edge(s_key, sn, label=f"{prob*100:.0f}% Outcome", color="#3498db", penwidth="3")
        st.graphviz_chart(dot)

with tab2:
    # Full data view for verification
    st.subheader("System Transition Logic")
    rows = []
    for s in states:
        for a in actions:
            p_vals = P_dyn[s][a]
            rows.append({
                "Context": f"{state_labels[s]} + {action_labels[a]}",
                "To Healthy": p_vals.get('S0', 0),
                "To Sick": p_vals.get('S1', 0),
                "To Critical": p_vals.get('S2', 0),
                "Sum Total": round(sum(p_vals.values()), 1)
            })
    st.table(pd.DataFrame(rows))
