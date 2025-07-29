import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import statsmodels.api as sm
import plotly.graph_objects as go
from openai import OpenAI
from collections import defaultdict

# ------------------ Configuration ------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
    client = None
else:
    client = OpenAI(api_key=api_key)

# ------------------ Utility Functions ------------------

def fetch_eurostat(dataset: str, years: int = 10) -> pd.DataFrame:
    """
    Fetches Eurostat data for Ireland via API v2.1; falls back to synthetic if error.
    """
    url = f"https://api.europa.eu/eurostat/data/v2.1/{dataset}?geo=IE"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("value", {})
        years_list = sorted({int(k.split(":")[-1]) for k in data.keys()})
        df = pd.DataFrame({"Year": years_list,
                           dataset: [data.get(f"IE:{y}", np.nan) for y in years_list]})
        if df[dataset].dropna().empty:
            raise ValueError
        return df.dropna()
    except Exception:
        st.warning(f"Failed to fetch {dataset} from Eurostat API; using synthetic data.")
    current = pd.Timestamp.now().year
    yrs = list(range(current - years + 1, current + 1))
    values = np.random.uniform(0.5, 1.5, len(yrs))
    return pd.DataFrame({"Year": yrs, dataset: values})


def run_econometric_diagnostics(df: pd.DataFrame, var: str):
    """Runs OLS; returns None if insufficient data."""
    if df.shape[0] < 2:
        return None
    X = sm.add_constant(np.arange(len(df)))
    y = df[var].values
    return sm.OLS(y, X).fit()


def agentic_ai_response(prompt: str, role: str) -> str:
    """Queries GPT-4o-mini using latest OpenAI client API."""
    if client is None:
        return "AI client not initialized."
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are {role}."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=400
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI response error: {e}"


def quantum_noise(df: pd.DataFrame, intensity: float = 0.05) -> pd.DataFrame:
    """Adds Gaussian noise to simulate quantum uncertainty."""
    noisy = df.copy()
    noise = np.random.normal(0, intensity, len(df))
    noisy[f"{df.columns[1]}_quantum"] = df.iloc[:,1] * (1 + noise)
    return noisy


def train_rl_agent(df: pd.DataFrame, var: str, episodes: int = 50) -> str:
    """Simple Q-learning: subsidy change actions -1,0,1."""
    n = df.shape[0]
    actions = [-1, 0, 1]
    Q = defaultdict(lambda: {a: 0.0 for a in actions})
    alpha, gamma, eps = 0.1, 0.9, 0.1
    for _ in range(episodes):
        state = 0
        for t in range(n - 1):
            if np.random.rand() < eps:
                a = np.random.choice(actions)
            else:
                a = max(Q[state], key=Q[state].get)
            reward = -abs(df[var].iloc[t+1] - df[var].iloc[t] * (1 + 0.01 * a))
            next_s = t + 1
            best_next = max(Q[next_s].values())
            Q[state][a] += alpha * (reward + gamma * best_next - Q[state][a])
            state = next_s
    best_action = max(Q[0], key=Q[0].get)
    return f"Subsidy change: {best_action * 100}%"


def multi_agent_debate(topic: str, agents: list, rounds: int = 3) -> str:
    """Orchestrates a debate among multiple agents."""
    msgs = [{"role": "system", "content": f"Debate Moderator: {topic}"}]
    for ag in agents:
        msgs.append({"role": "system", "content": f"Agent {ag['name']}: {ag['system_prompt']}"})
    for _ in range(rounds):
        for ag in agents:
            msgs.append({"role": "user", "content": f"{ag['name']}, analysis:"})
            reply = agentic_ai_response(msgs[-1]["content"], ag["name"])
            msgs.append({"role": "assistant", "content": reply})
    return "\n\n".join([m["content"] for m in msgs if m["role"] in ["assistant", "user"]])

# ------------------ Streamlit App ------------------

st.title("5D Quantum-Agentic Econometric Simulator")

dataset = st.text_input("Eurostat Dataset code:", "sdg_02_40")
mode = st.selectbox("Mode:", ["National", "County Drill-Down"])
run = st.button("Run Simulation")

if run:
    df_nat = fetch_eurostat(dataset)
    if mode == "National":
        df = df_nat
        st.subheader("National Data")
        st.write(df)
    else:
        import geopandas as gpd
        gdf = gpd.read_file("ie_counties.geojson")[["NAME"]]
        county = st.selectbox("Select County:", gdf["NAME"])
        df = df_nat.copy()
        noise = np.random.normal(0, 0.02, len(df))
        df[dataset] = df[dataset] * (1 + noise)
        st.subheader(f"Synthetic Data for {county}")
        st.write(df)

    model = run_econometric_diagnostics(df, dataset)
    if not model:
        st.error("Insufficient data for regression.")
        st.stop()
    st.subheader("Agent 1: Diagnostic AI")
    st.text(model.summary())

    st.subheader("Agent 2: Interpretive AI")
    interp = agentic_ai_response(f"Explain regression:\n{model.summary()}", "Interpretive AI")
    st.write(interp)

    st.subheader("Agent 3: Policy Simulation AI")
    policy = agentic_ai_response(
        "Simulate a 10% subsidy increase based on regression.",
        "Policy AI"
    )
    st.write(policy)

    st.subheader("Agent 4: Quantum Scaling AI")
    qdf = quantum_noise(df)
    st.write(qdf)
    anomaly = agentic_ai_response(
        "Recommend adjustments for quantum-induced volatility.",
        "Quantum AI"
    )
    st.write(anomaly)

    st.subheader("Agent 5: Reinforcement Learning AI")
    rl = train_rl_agent(df, dataset)
    st.write(rl)

    st.subheader("4D Visualization")
    fig = go.Figure(data=[
        go.Scatter3d(
            x=qdf["Year"],
            y=qdf[dataset],
            z=qdf[f"{dataset}_quantum"],
            mode='lines+markers',
            marker=dict(size=6, color=qdf[f"{dataset}_quantum"], colorscale='Viridis')
        )
    ])
    fig.update_layout(
        scene=dict(xaxis_title="Year", yaxis_title="Value", zaxis_title="Quantum_Adjusted"),
        title='5D Simulation'
    )
    st.plotly_chart(fig)

    st.subheader("Multi-Agent Debate")
    agents = [
        {"name": "Diagnostic AI", "system_prompt": "Assess statistical validity."},
        {"name": "Interpretive AI", "system_prompt": "Narrate regression output."},
        {"name": "Policy AI", "system_prompt": "Argue policy merits."},
        {"name": "Quantum AI", "system_prompt": "Discuss quantum uncertainty."},
        {"name": "RL AI", "system_prompt": "Recommend actions based on rewards."}
    ]
    debate = multi_agent_debate(f"Impact of {dataset}", agents)
    st.write(debate)
