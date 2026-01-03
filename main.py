# =========================================================
# 🧠 AMF-OMEGA PRIME — APP AUTOMÁTICO (VERSÃO FINAL)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np

# ========================
# CONFIGURAÇÕES
# ========================
TOP_N = 5
N_MC = 8000
DECAY_HALFLIFE = 45
SEED = 42

W_FREQ_M = 0.30
W_FREQ_C = 0.20
W_REC = 0.20
W_MC = 0.20
W_PENAL = 0.10

np.random.seed(SEED)

# ========================
# FUNÇÕES
# ========================
def decay(days, hl):
    return np.exp(-np.log(2) * days / hl)

def gerar_previsao(df, premio, milhar_bloqueada, centena_hist):
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

            if mil in milhar_bloqueada:
                continue  # NÃO repetir milhar do CSV

            penal = 1.0 if cent in centena_hist else 0.0

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

# ========================
# APP
# ========================
st.set_page_config(page_title="AMF-OMEGA PRIME", layout="centered")
st.title("🧠 AMF-OMEGA PRIME")
st.subheader("Sistema automático de análise e geração de milhares")

arquivo = st.file_uploader("📂 Envie o CSV do Jogo do Bicho", type=["csv"])

if arquivo:
    df = pd.read_csv(arquivo)
    df.columns = df.columns.str.lower().str.strip()

    df["milhar"] = df["milhar"].astype(str).str.zfill(4)
    df["centena"] = df["milhar"].str[-3:]
    df["premio"] = df["premio"].astype(str)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    df = df.dropna(subset=["milhar", "centena", "premio", "data"])

    milhar_bloqueada = set(df["milhar"])
    centena_hist = set(df["centena"])

    st.success("CSV carregado com sucesso ✔️")

    if st.button("🚀 GERAR PREVISÃO"):
        resultados = []

        for p in ["1º", "2º", "3º", "4º", "5º"]:
            r = gerar_previsao(df, p, milhar_bloqueada, centena_hist)
            resultados.append(r)

        final = pd.concat(resultados, ignore_index=True)

        st.subheader("📊 MILHARES CANDIDATAS (NÃO REPETIDAS)")
        st.dataframe(final[["premio", "milhar", "centena", "score"]])

else:
    st.info("Aguardando upload do CSV…")
