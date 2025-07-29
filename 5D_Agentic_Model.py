import streamlit as st
import pandas as pd
import numpy as np
import requests
import openai
import statsmodels.api as sm
import plotly.graph_objects as go
import geopandas as gpd
import os

# --- Attempt to import PennyLane for quantum features ---
try:
    import pennylane as qml
    from pennylane import numpy as pnp
    QML_AVAILABLE = True
except (ImportError, AttributeError):
    QML_AVAILABLE = False

# --- Configuration ---
# OpenAI API Key
api_key = None
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Set it in Streamlit secrets or OPENAI_API_KEY environment variable.")
else:
    openai.api_key = api_key

# Path to counties GeoJSON\GEOJSON_PATH = "ie_counties.geojson"

# --- Utility Functions ---

def fetch_eurostat(dataset: str, geo: str = "IE") -> pd.DataFrame:
    """Fetch national Eurostat data for given dataset."""
    url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset}?filterNonGeo=1&geo={geo}"
    resp = requests.get(url)
    if resp.status_code != 200:
        st.error(f"Eurostat fetch error: {resp.status_code}")
        return pd.DataFrame()
    values = resp.json().get('value', {})
    years = sorted({int(k.split(":")[-1]) for k in values.keys()})
    df = pd.DataFrame({"Year": years})
    df[dataset] = [values.get(f"{geo}:{y}", np.nan) for y in years]
    return df.dropna()


def run_econometric_diagnostics(df: pd.DataFrame, var: str):
    """Perform OLS regression, with safety checks for empty or insufficient data."""
    if df.empty:
        st.error("No data available for regression.")
        return None
    if len(df) < 2:
        st.error("Not enough data points for regression.")
        return None
    X = sm.add_constant(np.arange(len(df)))
    y = df[var].values
    return sm.OLS(y, X).fit()


def quantum_noise(df: pd.DataFrame, intensity: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    noise = np.random.normal(0, intensity, size=len(df))
    df["Quantum_Adjusted"] = df.iloc[:, 1] * (1 + noise)
    return df


def agentic_ai_response(messages: list, model: str = "gpt-4o-mini") -> str:
    resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=400)
    return resp.choices[0].message.content

# --- Quantum LDPC Helpers ---

def get_quantum_device(backend_name: str = "lightning.qubit", wires: int = 12):
    if not QML_AVAILABLE:
        raise RuntimeError("PennyLane unavailable.")
    return qml.device(backend_name, wires=wires)


def generate_ldpc_matrix(n: int, m: int, density: float = 0.25) -> np.ndarray:
    H = (np.random.rand(m, n) < density).astype(int)
    for i in range(m):
        if not H[i].any():
            H[i, np.random.randint(0, n)] = 1
    return H


def build_qldpc_qnode(H: np.ndarray, backend: str = "lightning.qubit"):
    dev = get_quantum_device(backend, wires=H.shape[1] + H.shape[0])
    @qml.qnode(dev)
    def qldpc_syndrome(prep_angles):
        for i, angle in enumerate(prep_angles): qml.RY(angle, wires=i)
        for row in range(H.shape[0]):
            anc = H.shape[1] + row
            for col in range(H.shape[1]):
                if H[row, col]: qml.CNOT(wires=[col, anc])
        return [qml.sample(qml.PauliZ(i)) for i in range(H.shape[1], H.shape[1]+H.shape[0])]
    return qldpc_syndrome


def decode_syndrome(syndrome: list) -> list:
    return syndrome

# --- Reinforcement Learning Agent ---
class PolicyEnv:
    def __init__(self, df: pd.DataFrame, model):
        self.df = df; self.model = model; self.actions = np.arange(0, .31, .05)
        self.state = df.iloc[-1,1]
    def step(self, idx):
        subsidy = self.actions[idx]
        X = sm.add_constant(np.arange(len(self.df)))
        y_pred = self.model.predict(X)
        reward = -abs((y_pred[-1]*(1+subsidy))-y_pred[-1])
        return self.state, reward, True, {}
    def reset(self): return self.state

def train_q_learning_agent(env: PolicyEnv, episodes: int=20, alpha: float=.1, gamma: float=.9):
    q_table = np.zeros((1,len(env.actions)))
    for _ in range(episodes):
        _ = env.reset()
        for a in range(len(env.actions)):
            _,r,_,_ = env.step(a)
            q_table[0,a] += alpha*(r+gamma*np.max(q_table[0])-q_table[0,a])
    return q_table

def select_best_action(q_table: np.ndarray) -> int:
    return int(np.argmax(q_table[0]))

# --- Multi-Agent Debate ---
def multi_agent_debate(topic: str, agents: list, rounds: int=3) -> str:
    msgs=[{"role":"system","content":f"Debate Moderator: {topic}"}]
    for ag in agents: msgs.append({"role":"system","content":f"Agent {ag['name']}: {ag['system_prompt']}"})
    for _ in range(rounds):
        for ag in agents:
            msgs.append({"role":"user","content":f"{ag['name']}, analysis:"})
            rep=agentic_ai_response(msgs); msgs.append({"role":"assistant","content":rep})
    return "\n\n".join([m['content'] for m in msgs if m['role'] in ['assistant','user']])

# --- Streamlit App ---
st.title("5D Quantum-Agentic Simulator with Synthetic Counties")
mode=st.sidebar.selectbox("Mode",["National","County Drill-Down"])
dataset=st.sidebar.text_input("Eurostat Dataset","sdg_02_40")
enable_qldpc=st.sidebar.checkbox("Enable qLDPC",False)
n=st.sidebar.slider("LDPC n",4,32,8)
m=st.sidebar.slider("LDPC m",1,16,4)
train_rl=st.sidebar.button("Train RL Agent")

# Fetch national data
national_df=fetch_eurostat(dataset,"IE")
if national_df.empty: st.error("No Eurostat data. Check dataset code."); st.stop()

# Select data
if mode=="National": df=national_df; st.write("### National Data",df)
else:
    gdf=gpd.read_file(GEOJSON_PATH)[['NAME']]
    county=st.sidebar.selectbox("County",gdf['NAME'])
    base=national_df.copy(); base['value']=base[dataset]
    noise=np.random.normal(0,0.02,len(base)); base['value']*=1+noise
    df=pd.DataFrame({'Year':base['Year'],dataset:base['value']})
    st.write(f"### Synthetic Data for {county}",df)

# Regression
diagnosis=run_econometric_diagnostics(df,dataset)
if not diagnosis: st.stop()
st.text(diagnosis.summary())

# Interpretation
i2=agentic_ai_response([
    {'role':'system','content':'You are Interpretive AI.'},
    {'role':'user','content':f'Explain regression: {diagnosis.summary()}'}
]); st.subheader('Interpretation'); st.write(i2)

# Policy
p3=agentic_ai_response([
    {'role':'system','content':'You are Policy AI.'},
    {'role':'user','content':'Simulate 10% subsidy increase.'}
]); st.subheader('Policy Simulation'); st.write(p3)

# Quantum
st.subheader('Quantum Scaling')
if enable_qldpc and QML_AVAILABLE:
    H=generate_ldpc_matrix(n,m); st.write('H:',H)
    qnode=build_qldpc_qnode(H)
    sample=qnode(np.random.uniform(0,np.pi,n)); st.write('Syndrome:',sample)
else:
    qdf=quantum_noise(df); st.write(qdf)

# Reinforcement Learning
st.subheader('Reinforcement Learning')
env=PolicyEnv(df,diagnosis)
if train_rl:
    qt=train_q_learning_agent(env); idx=select_best_action(qt)
    st.write(f'RL suggests subsidy: {env.actions[idx]*100:.1f}%')
else: st.write('Train RL agent to proceed')

# Visualization
fig=go.Figure(data=[go.Scatter3d(
    x=df['Year'],y=df[dataset],z=df.get('Quantum_Adjusted',df[dataset]),
    mode='lines+markers',marker=dict(size=5,colorscale='Viridis',color=df.get('Quantum_Adjusted',df[dataset]))
)]); fig.update_layout(scene=dict(xaxis_title='Year',yaxis_title=dataset,zaxis_title='Quantum_Adjusted'),title='5D Simulation'); st.plotly_chart(fig)

# Debate
st.subheader('Multi-Agent Debate')
ag=[
    {'name':'Diagnostic AI','system_prompt':'Assess statistical validity.'},
    {'name':'Interpretive AI','system_prompt':'Narrate regression.'},
    {'name':'Policy AI','system_prompt':'Debate policy.'},
    {'name':'Quantum AI','system_prompt':'Discuss uncertainty.'},
    {'name':'RL AI','system_prompt':'Advocate actions.'}
]
deb=multi_agent_debate(f'{dataset} impact',ag); st.text(deb)
