# =========================================================
# 🧠 AMF-OMEGA PRIME — SISTEMA ADAPTATIVO COMPLETO (VERSÃO AVANÇADA)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import math
import json
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import chisquare
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
import xgboost as xgb

# ---------------- CONFIG ----------------
TOP_N = 5
N_MC = 10000
HALF_LIFE = 45
SEQ_LEN = 10  # Para LSTM
SEED = 42
PESO_FILE = "pesos.json"

np.random.seed(SEED)

# ------------- PESOS DINÂMICOS ----------
DEFAULT_PESOS = {
    "freq_m": 0.20,
    "freq_c": 0.15,
    "rec": 0.15,
    "mc": 0.15,
    "entropia": 0.10,
    "lstm": 0.10,
    "ae_anomaly": 0.05,
    "xgb": 0.05,
    "patterns": 0.05
}

def carregar_pesos():
    if os.path.exists(PESO_FILE):
        return json.load(open(PESO_FILE))
    return DEFAULT_PESOS.copy()

def salvar_pesos(p):
    json.dump(p, open(PESO_FILE, "w"))

pesos = carregar_pesos()

# ---------------- FUNÇÕES ---------------
def decay(days):
    return np.exp(-np.log(2) * days / HALF_LIFE)

def entropia(vals):
    p = np.array(vals)
    p = p[p > 0]
    return -np.sum(p * np.log2(p)) if len(p) > 0 else 0

def train_lstm(base, seq_len=SEQ_LEN):
    if len(base) < seq_len + 1:
        return lambda x: 0  # Fallback se dados insuficientes
    
    data = base['milhar'].astype(int).values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len])
    
    X, y = np.array(X), np.array(y)
    model = Sequential([LSTM(50, input_shape=(seq_len, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0, batch_size=32)  # Reduzido para eficiência
    
    def predict(mil):
        last_seq = data_scaled[-seq_len:].reshape(1, seq_len, 1)
        pred = scaler.inverse_transform(model.predict(last_seq, verbose=0))[0][0]
        return abs(int(mil) - pred) / 9999  # Normalizado, menor distância = melhor score
    
    return predict

def train_autoencoder(base):
    if len(base) < 10:
        return lambda x: 0, 0  # Fallback
    
    data = np.array([list(map(int, m)) for m in base['milhar']])
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    input_layer = Input(shape=(4,))
    encoded = Dense(2, activation='relu')(input_layer)
    decoded = Dense(4, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data_scaled, data_scaled, epochs=20, verbose=0, batch_size=32)
    
    recon = autoencoder.predict(data_scaled, verbose=0)
    errors = np.mean((data_scaled - recon)**2, axis=1)
    threshold = np.mean(errors) + 2 * np.std(errors)
    
    def anomaly_score(mil):
        mil_vec = np.array([list(map(int, mil))]).reshape(1, -1)
        mil_scaled = scaler.transform(mil_vec)
        recon = autoencoder.predict(mil_scaled, verbose=0)
        error = np.mean((mil_scaled - recon)**2)
        return max(0, (error - threshold) / max(errors)) if max(errors) > 0 else 0
    
    return anomaly_score

def train_xgboost(base):
    if len(base) < 10:
        return lambda x: 0  # Fallback
    
    base['sum_digits'] = base['milhar'].apply(lambda m: sum(int(d) for d in m))
    base['parity'] = base['milhar'].apply(lambda m: sum(int(d) % 2 for d in m))
    base['centena'] = base['milhar'].str[-3:].astype(int)
    
    features = ['sum_digits', 'parity', 'centena']
    X = base[features].iloc[:-1]
    y = base['milhar'].astype(int).iloc[1:]
    
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
    model.fit(X, y)
    
    def predict(features_dict):
        df = pd.DataFrame([features_dict])
        return model.predict(df)[0] / 9999  # Normalizado
    
    return predict, features

def calculate_patterns(mil):
    digits = list(map(int, mil))
    sum_digits = sum(digits)
    parity = sum(d % 2 for d in digits)
    repeats = len(digits) - len(set(digits))
    return sum_digits / 36, parity / 4, repeats / 3  # Normalizados

def chi_uniformity_test(freq):
    observed = freq.values
    expected = np.full_like(observed, len(observed) / sum(observed))
    chi, p = chisquare(observed, expected)
    return p  # Baixo p-value = não-uniforme (boost para candidatos de distribuições desviadas)

# ---------------- UI --------------------
st.set_page_config(page_title="AMF-OMEGA PRIME AVANÇADO", layout="wide")
st.title("🧠 AMF-OMEGA PRIME AVANÇADO")
st.caption("Sistema adaptativo avançado com ML para análise e geração de milhares")

csv = st.file_uploader("📤 Envie o CSV", type="csv")

if not csv:
    st.stop()

df = pd.read_csv(csv)
df.columns = df["concurso","data","premio","milhar","grupo","animal"].str.lower()

df["milhar"] = df["milhar"].astype(str).str.zfill(4)
df["centena"] = df["milhar"].str[-3:]
df["data"] = pd.to_datetime(df["data"], errors="coerce")
df = df.sort_values("data")  # Ordem temporal

# ---------------- BLOQUEIOS -------------
milhares_usadas = set(df["milhar"])

# ---------------- PROCESSAMENTO ---------
resultados = []

for premio in df["premio"].unique():

    base = df[df["premio"] == premio].copy()

    freq_m = base["milhar"].value_counts(normalize=True)
    freq_c = base["centena"].value_counts(normalize=True)

    ultima_data = base["data"].max()
    rec = base.groupby("milhar")["data"].max().apply(
        lambda d: decay((ultima_data - d).days) if pd.notnull(d) else 0
    )
    atraso = base.groupby("centena")["data"].max().apply(
        lambda d: (ultima_data - d).days if pd.notnull(d) else 0
    )

    # MC ponderado por atraso inverso
    probs = freq_c.values / freq_c.values.sum()
    probs = probs * (1 / (1 + atraso[freq_c.index].values / atraso.max()))
    probs /= probs.sum()
    mc_draws = np.random.choice(freq_c.index, size=N_MC, p=probs)
    mc_score = pd.Series(mc_draws).value_counts(normalize=True)

    # Modelos avançados
    lstm_predict = train_lstm(base)
    ae_anomaly = train_autoencoder(base)
    xgb_predict, xgb_features = train_xgboost(base)

    # Teste estatístico
    chi_p = chi_uniformity_test(freq_c)

    candidatos = []

    for cent in freq_c.index:
        for d in range(10):
            mil = f"{d}{cent}"
            if mil in milhares_usadas:
                continue

            sum_norm, par_norm, rep_norm = calculate_patterns(mil)
            patterns_score = (sum_norm + par_norm + rep_norm) / 3

            # Features para XGB
            xgb_dict = {
                'sum_digits': sum(int(dd) for dd in mil),
                'parity': sum(int(dd) % 2 for dd in mil),
                'centena': int(cent)
            }
            xgb_score = abs(xgb_predict(xgb_dict) - int(mil))

            candidatos.append({
                "premio": premio,
                "milhar": mil,
                "centena": cent,
                "freq_m": freq_m.get(mil, 0),
                "freq_c": freq_c.get(cent, 0),
                "rec": rec.get(mil, 0),
                "mc": mc_score.get(cent, 0),
                "entropia": 0,  # Preencher depois
                "lstm": lstm_predict(mil),
                "ae_anomaly": ae_anomaly(mil),
                "xgb": xgb_score / 9999,  # Normalizado
                "patterns": patterns_score,
                "chi_boost": max(0, 1 - chi_p)  # Boost se não-uniforme
            })

    cand = pd.DataFrame(candidatos)

    # Entropia por centena
    ent = cand.groupby("centena")["freq_c"].apply(entropia)
    cand["entropia"] = cand["centena"].map(ent)

    # Normalização
    for c in ["freq_m", "freq_c", "rec", "mc", "entropia", "lstm", "ae_anomaly", "xgb", "patterns"]:
        m = cand[c].max()
        if m > 0:
            cand[c] = cand[c] / m

    cand["score"] = (
        pesos["freq_m"] * cand["freq_m"] +
        pesos["freq_c"] * cand["freq_c"] +
        pesos["rec"] * cand["rec"] +
        pesos["mc"] * cand["mc"] +
        pesos["entropia"] * cand["entropia"] +
        pesos["lstm"] * cand["lstm"] +
        pesos["ae_anomaly"] * cand["ae_anomaly"] +
        pesos["xgb"] * cand["xgb"] +
        pesos["patterns"] * cand["patterns"] +
        0.05 * cand["chi_boost"]  # Peso extra para chi
    )

    resultados.append(
        cand.sort_values("score", ascending=False).head(TOP_N)
    )

final = pd.concat(resultados)

st.subheader("📊 PREVISÃO FINAL")
st.dataframe(final)

# ---------------- AUTO-APRENDIZADO -------
st.subheader("🧠 Auto-aprendizado")

acertou = st.selectbox(
    "O sistema acertou alguma milhar/centena?",
    ["Nenhuma", "Algumas", "Muitas"]
)

if st.button("Atualizar aprendizado"):
    ajuste = {
        "Nenhuma": -0.05,
        "Algumas": 0.03,
        "Muitas": 0.07
    }[acertou]

    for k in pesos:
        pesos[k] = max(0.05, pesos[k] + ajuste * np.random.uniform(0.8, 1.2))  # Variação para sofisticação

    s = sum(pesos.values())
    for k in pesos:
        pesos[k] /= s

    salvar_pesos(pesos)
    st.success("Pesos ajustados automaticamente com variação ✔️")
