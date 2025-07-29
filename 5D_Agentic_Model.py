import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import openai
import statsmodels.api as sm
import plotly.graph_objects as go

# Optional: RL imports
from collections import defaultdict

# ------------------ Configuration ------------------
# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OpenAI API key not found. Set OPENAI_API_KEY in your environment.")

# ------------------ Utility Functions ------------------

def fetch_eurostat(dataset: str, years: int = 10) -> pd.DataFrame:
    """
    Fetches Eurostat data; on error returns synthetic series.
    """
    url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/2.1/data/{dataset}"?geo=IE"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get('value', {})
            years_list = sorted({int(k.split(':')[-1]) for k in data.keys()})
            df = pd.DataFrame({"Year": years_list,
                               dataset: [data.get(f"IE:{y}", np.nan) for y in years_list]})
            if df[dataset].dropna().empty:
                raise ValueError
            return df.dropna()
    except Exception:
        st.warning(f"Failed to fetch {dataset}, using synthetic data.")
    # Fallback synthetic data
    yrs = list(range(pd.Timestamp.now().year - years + 1, pd.Timestamp.now().year + 1))
    values = np.random.uniform(0.5, 1.5, len(yrs))
    return pd.DataFrame({"Year": yrs, dataset: values})


def run_econometric_diagnostics(df: pd.DataFrame, var: str):
    """
    Runs OLS and returns the fitted model; returns None if data insufficient.
    """
    if df.shape[0] < 2:
        return None
    X = sm.add_constant(np.arange(len(df)))
    y = df[var].values
    model = sm.OLS(y, X).fit()
    return model


def agentic_ai_response(prompt: str, role: str) -> str:
    """
    Generic GPT-4o mini agent response builder.
    """
    try:
        res = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",  "content": f"You are {role}."},
                {"role": "user",    "content": prompt}
            ],
            max_tokens=400
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI response error: {e}"


def quantum_noise(df: pd.DataFrame, intensity: float = 0.05) -> pd.DataFrame:
    """
    Adds Gaussian noise to simulate quantum uncertainty.
    """
    noisy = df.copy()
    noise = np.random.normal(0, intensity, size=len(noisy))
    noisy[f"{noisy.columns[1]}_quantum"] = noisy.iloc[:, 1] * (1 + noise)
    return noisy


def train_rl_agent(df: pd.DataFrame, var: str, episodes: int = 50):
    """
    Simple Q-learning stub for optimal policy discovery.
    """
    # Discrete state: year index; actions: -1, 0, +1 subsidy change
    states = list(range(len(df)))
    actions = [-1, 0, 1]
    Q = defaultdict(lambda: {a: 0. for a in actions})
    alpha, gamma, eps = 0.1, 0.9, 0.1
    for ep in range(episodes):
        state = 0
        for t in range(len(states)-1):
            if np.random.random() < eps:
                action = np.random.choice(actions)
            else:
                action = max(Q[state], key=Q[state].get)
            reward = -abs(df[var].iloc[t+1] - df[var].iloc[t] * (1 + action*0.01))
            next_s = t+1
            best_next = max(Q[next_s].values())
            Q[state][action] += alpha * (reward + gamma*best_next - Q[state][action])
            state = next_s
    # Derive policy for first state
    best_action = max(Q[0], key=Q[0].get)
    return best_action

# ------------------ Streamlit App ------------------
st.title("5D Quantum-Agentic Econometric Simulator")

dataset = st.text_input("Eurostat Dataset Code:", "sdg_02_40")
run = st.button("Run Simulation")

if run:
    # Fetch data
    df = fetch_eurostat(dataset)
    st.subheader("Live Data")
    st.write(df)

    # Layer 1: Diagnostic AI
    model = run_econometric_diagnostics(df, dataset)
    if not model:
        st.error("Insufficient data for econometric diagnostics.")
        st.stop()
    st.subheader("Agent 1: Diagnostic AI")
    st.text(model.summary())

    # Layer 2: Interpretive AI
    st.subheader("Agent 2: Interpretive AI")
    interp = agentic_ai_response(f"Explain this regression:\n{model.summary()}", "Interpretive AI")
    st.write(interp)

    # Layer 3: Policy Simulation AI
    st.subheader("Agent 3: Policy Simulation AI")
    policy = agentic_ai_response(
        "Simulate a 10% subsidy increase and its impact based on regression.",
        "Policy AI"
    )
    st.write(policy)

    # Layer 4: Quantum Scaling AI
    st.subheader("Agent 4: Quantum Scaling AI")
    noisy_df = quantum_noise(df)
    st.write(noisy_df)
    anomaly = agentic_ai_response(
        "Data shows volatility from quantum noise. Recommend policy adjustments.",
        "Quantum AI"
    )
    st.write(anomaly)

    # Layer 5: Reinforcement-Learning AI
    st.subheader("Agent 5: Reinforcement-Learning AI")
    best_act = train_rl_agent(df, dataset)
    st.write(f"Recommended action (subsidy change): {best_act}")

    # Visualization: 3D + quantum
    st.subheader("4D Visualization with Quantum Axis")
    fig = go.Figure(data=[
        go.Scatter3d(
            x=noisy_df["Year"],
            y=noisy_df[dataset],
            z=noisy_df[f"{dataset}_quantum"],
            mode='lines+markers',
            marker=dict(size=6, color=noisy_df[f"{dataset}_quantum"], colorscale='Viridis')
        )
    ])
    fig.update_layout(
        scene=dict(xaxis_title='Year', yaxis_title='Value', zaxis_title='Quantum-Adjusted'),
        title='5D Simulation'
    )
    st.plotly_chart(fig)
