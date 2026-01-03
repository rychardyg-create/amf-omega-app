        # ==========================================================
# 🧠 AMF-OMEGA PRIME v3 — APP AUTOMÁTICO STREAMLIT
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
TOP_N = 5
N_MC = 8000
DECAY_HALFLIFE = 45
SEMENTE = 42

np.random.seed(SEMENTE)

# =========================
# INTERFACE
# =========================
st.set_page_config(page_title="AMF-OMEGA PRIME", layout="wide")
st.title("🧠 AMF-OMEGA PRIME")
st.subheader("Sistema automático de análise e geração de milhares")

modo = st.selectbox(
    "⚙️ Modo de operação",
    ["Equilibrado", "Conservador", "Agressivo"]
)

uploaded = st.file_uploader("📂 Envie o CSV do Jogo do Bicho", type=["csv"])

# =========================
# FUNÇÕES
# =========================

def normalizar(s):
    if s.max() == s.min():
        return s * 0
    return (s - s.min()) / (s.max() - s.min())

def gerar_milhares_validas(df):
    todas = [f"{i:04d}" for i in range(10000)]
    usadas = set(df["milhar"].astype(str))
    return sorted(list(set(todas) - usadas))

def backtest(df, candidatos):
    ultimos = df.tail(1)
    resultado = {
        "1º": 0,
        "2º": 0,
        "3º": 0,
        "4º": 0,
        "5º": 0
    }
    for premio in resultado:
        if ultimos[premio].values[0] in candidatos:
            resultado[premio] = 1
    return resultado

# =========================
# EXECUÇÃO
# =========================
if uploaded:

    df = pd.read_csv(uploaded)

    # Esperado: colunas 1º a 5º
    premios = ["1º", "2º", "3º", "4º", "5º"]

    # Normaliza formato
    df = df[premios].astype(str)
    df = df.applymap(lambda x: x.zfill(4))

    df_long = df.melt(var_name="premio", value_name="milhar")
    df_long["centena"] = df_long["milhar"].str[-3:]

    # Frequências
    f_m = df_long["milhar"].value_counts()
    f_c = df_long["centena"].value_counts()

    # Recência
    rec = {}
    for m in f_m.index:
        idx = df_long[df_long["milhar"] == m].index.max()
        rec[m] = len(df_long) - idx

    rec = pd.Series(rec)

    # Monte Carlo
    mc = pd.Series(
        np.random.rand(len(f_m)),
        index=f_m.index
    )

    # Milhares válidas
    validas = gerar_milhares_validas(df_long)

    rows = []

    for m in validas:
        c = m[-3:]
        rows.append({
            "milhar": m,
            "centena": c,
            "f_m": f_m.get(m, 0),
            "f_c": f_c.get(c, 0),
            "rec": rec.get(m, rec.max()),
            "mc": mc.sample(1).values[0]
        })

    res = pd.DataFrame(rows)

    # Normalizações
    res["f_m"] = normalizar(res["f_m"])
    res["f_c"] = normalizar(res["f_c"])
    res["rec"] = normalizar(res["rec"])

    # Pesos por modo
    if modo == "Conservador":
        W = dict(f_m=0.25, f_c=0.20, rec=0.35, mc=0.20)
    elif modo == "Agressivo":
        W = dict(f_m=0.35, f_c=0.25, rec=0.10, mc=0.30)
    else:
        W = dict(f_m=0.30, f_c=0.20, rec=0.20, mc=0.30)

    res["score"] = (
        res["f_m"] * W["f_m"] +
        res["f_c"] * W["f_c"] +
        res["rec"] * W["rec"] +
        res["mc"] * W["mc"]
    )

    res = res.sort_values("score", ascending=False)

    final = []

    for i, premio in enumerate(premios):
        bloco = res.iloc[i*TOP_N:(i+1)*TOP_N].copy()
        bloco["premio"] = premio
        final.append(bloco)

    resultado_final = pd.concat(final)

    # =========================
    # BACKTEST
    # =========================
    candidatos = resultado_final["milhar"].tolist()
    bt = backtest(df, candidatos)

    st.subheader("📊 BACKTEST (último concurso)")
    st.json(bt)

    # =========================
    # RESULTADO FINAL
    # =========================
    st.subheader("🧠 PREVISÃO FINAL — MILHARES CANDIDATAS")
    st.dataframe(
        resultado_final[["premio", "milhar", "centena", "score"]],
        use_container_width=True
    )

else:
    st.info("⬆️ Aguardando upload do CSV...")
