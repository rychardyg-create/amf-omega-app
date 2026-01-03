# =========================================================
# 🧠 AMF-OMEGA PRIME — APP AUTOMÁTICO (STREAMLIT)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
TOP_N = 20
N_MC = 8000
DECAY_HALFLIFE = 45
SEED = 42

W_FREQ_M = 0.30
W_FREQ_C = 0.20
W_REC = 0.20
W_MC = 0.20
W_PENAL = 0.10

np.random.seed(SEED)

# =========================
# INTERFACE
# =========================
st.set_page_config(
    page_title="AMF-OMEGA PRIME",
    layout="centered"
)

st.title("🧠 AMF-OMEGA PRIME")
st.caption("Sistema automático de análise e geração de milhares")

st.markdown("---")

# =========================
# UPLOAD DO CSV
# =========================
arquivo = st.file_uploader(
    "📂 Envie o arquivo CSV (histórico do Jogo do Bicho)",
    type=["csv"]
)

if not arquivo:
    st.info("Envie um arquivo CSV para iniciar a análise.")
    st.stop()

# =========================
# CARREGAR E NORMALIZAR CSV
# =========================
df = pd.read_csv(arquivo)
df.columns = df.columns.str.lower().str.strip()

# Colunas obrigatórias
colunas_necessarias = {"milhar", "premio", "data"}
if not colunas_necessarias.issubset(df.columns):
    st.error("❌ O CSV precisa conter as colunas: milhar, premio, data")
    st.stop()

df["milhar"] = df["milhar"].astype(str).str.zfill(4)
df["centena"] = df["milhar"].str[-3:]
df["premio"] = df["premio"].astype(str)
df["data"] = pd.to_datetime(df["data"], errors="coerce")

df = df.dropna(subset=["milhar", "data"])

MILHAR_BLOQUEADA = set(df["milhar"])
CENTENA_HIST = set(df["centena"])

# =========================
# FUNÇÕES
# =========================
def decay(days, hl):
    return np.exp(-np.log(2) * days / hl)

def gerar_previsao(df, premio):
    base = df[df["premio"] == premio].copy()
    if base.empty:
        return pd.DataFrame()

    freq_m = base["milhar"].value_counts(normalize=True)
    freq_c = base["centena"].value_counts(normalize=True)

    last_date = base["data"].max()
    rec = (
        base.groupby("milhar")["data"]
        .max()
        .apply(lambda d: decay((last_date - d).days, DECAY_HALFLIFE))
    )

    cents = freq_c.index.values
    probs = freq_c.values / freq_c.values.sum()
    mc_draws = np.random.choice(cents, size=N_MC, p=probs)
    mc_score = pd.Series(mc_draws).value_counts(normalize=True)

    candidatos = []

    for cent in freq_c.index:
        for d in range(10):
            mil = f"{d}{cent}"

            if mil in MILHAR_BLOQUEADA:
                continue  # bloqueio absoluto

            penal = 1.0 if cent in CENTENA_HIST else 0.0

            candidatos.append({
                "milhar": mil,
                "centena": cent,
                "f_m": freq_m.get(mil, 0.0),
                "f_c": freq_c.get(cent, 0.0),
                "rec": rec.get(mil, 0.0),
                "mc": mc_score.get(cent, 0.0),
                "penal": penal
            })

    cand = pd.DataFrame(candidatos)
    if cand.empty:
        return pd.DataFrame()

    for c in ["f_m", "f_c", "rec", "mc"]:
        mx = cand[c].max()
        if mx > 0:
            cand[c] /= mx

    cand["score"] = (
        W_FREQ_M * cand["f_m"] +
        W_FREQ_C * cand["f_c"] +
        W_REC * cand["rec"] +
        W_MC * cand["mc"] -
        W_PENAL * cand["penal"]
    )

    return (
        cand.sort_values("score", ascending=False)
            .head(TOP_N)
            .assign(premio=premio)
            .reset_index(drop=True)
    )

# =========================
# EXECUÇÃO
# =========================
st.markdown("## 🎯 Previsão automática")

resultados = []

for p in ["1º", "2º", "3º", "4º", "5º"]:
    r = gerar_previsao(df, p)
    resultados.append(r)

resultado_final = pd.concat(resultados, ignore_index=True)

if resultado_final.empty:
    st.warning("Nenhuma previsão gerada.")
    st.stop()

st.success("✅ Previsão gerada com sucesso!")
st.dataframe(resultado_final, use_container_width=True)

# =========================
# DOWNLOAD
# =========================
csv_saida = resultado_final.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Baixar previsões em CSV",
    csv_saida,
    "previsao_amf_omega.csv",
    "text/csv"
)

st.markdown("---")
st.caption("AMF-OMEGA PRIME — análise estatística + Monte Carlo + recência")
