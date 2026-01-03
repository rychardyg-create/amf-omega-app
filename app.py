# =========================================================
# 🧠 AMF-OMEGA PRIME — SISTEMA ADAPTATIVO COMPLETO
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import math
import json
import os

# ---------------- CONFIG ----------------
TOP_N = 5
N_MC = 10000
HALF_LIFE = 45
SEED = 42
PESO_FILE = "pesos.json"

np.random.seed(SEED)

# ------------- PESOS DINÂMICOS ----------
DEFAULT_PESOS = {
    "freq_m": 0.30,
    "freq_c": 0.20,
    "rec": 0.20,
    "mc": 0.20,
    "entropia": 0.10
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
    return -np.sum(p * np.log2(p))

# ---------------- UI --------------------
st.set_page_config(page_title="AMF-OMEGA PRIME", layout="wide")
st.title("🧠 AMF-OMEGA PRIME")
st.caption("Sistema adaptativo de análise e geração de milhares")

csv = st.file_uploader("📤 Envie o CSV", type="csv")

if not csv:
    st.stop()

df = pd.read_csv(csv)
df.columns = df.columns.str.lower()

df["milhar"] = df["milhar"].astype(str).str.zfill(4)
df["centena"] = df["milhar"].str[-3:]
df["data"] = pd.to_datetime(df["data"], errors="coerce")

# ---------------- BLOQUEIOS -------------
milhares_usadas = set(df["milhar"])

# ---------------- PROCESSAMENTO ---------
resultados = []

for premio in df["premio"].unique():

    base = df[df["premio"] == premio]

    freq_m = base["milhar"].value_counts(normalize=True)
    freq_c = base["centena"].value_counts(normalize=True)

    rec = base.groupby("milhar")["data"].max().apply(
        lambda d: decay((base["data"].max() - d).days)
    )

    mc_draws = np.random.choice(
        freq_c.index,
        size=N_MC,
        p=freq_c.values / freq_c.values.sum()
    )
    mc_score = pd.Series(mc_draws).value_counts(normalize=True)

    candidatos = []

    for cent in freq_c.index:
        for d in range(10):
            mil = f"{d}{cent}"
            if mil in milhares_usadas:
                continue

            candidatos.append({
                "premio": premio,
                "milhar": mil,
                "centena": cent,
                "freq_m": freq_m.get(mil, 0),
                "freq_c": freq_c.get(cent, 0),
                "rec": rec.get(mil, 0),
                "mc": mc_score.get(cent, 0),
            })

    cand = pd.DataFrame(candidatos)

    # entropia por centena
    ent = cand.groupby("centena")["freq_c"].apply(
        lambda x: entropia(x)
    )
    cand["entropia"] = cand["centena"].map(ent)

    # normalização
    for c in ["freq_m","freq_c","rec","mc","entropia"]:
        m = cand[c].max()
        if m > 0:
            cand[c] /= m

    cand["score"] = (
        pesos["freq_m"] * cand["freq_m"] +
        pesos["freq_c"] * cand["freq_c"] +
        pesos["rec"] * cand["rec"] +
        pesos["mc"] * cand["mc"] +
        pesos["entropia"] * cand["entropia"]
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
    "O sistema acertou alguma milhar?",
    ["Nenhuma", "Algumas", "Muitas"]
)

if st.button("Atualizar aprendizado"):
    ajuste = {
        "Nenhuma": -0.05,
        "Algumas": 0.03,
        "Muitas": 0.07
    }[acertou]

    for k in pesos:
        pesos[k] = max(0.05, pesos[k] + ajuste)

    s = sum(pesos.values())
    for k in pesos:
        pesos[k] /= s

    salvar_pesos(pesos)
    st.success("Pesos ajustados automaticamente ✔️")
