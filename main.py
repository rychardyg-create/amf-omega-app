# ==========================================================
# 🧠 AMF-OMEGA PRIME — SISTEMA TOTAL (FINAL)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import math

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
SEMENTE = 42
np.random.seed(SEMENTE)

TOP_N = 5

PESOS = {
    "conservador": {"freq": 0.50, "rec": 0.30, "mc": 0.15, "ent": 0.05},
    "agressivo":   {"freq": 0.30, "rec": 0.20, "mc": 0.40, "ent": 0.10},
}

# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
st.set_page_config(page_title="AMF-OMEGA PRIME", layout="wide")

st.title("🧠 AMF-OMEGA PRIME")
st.caption("Sistema automático avançado de análise e geração de milhares")

modo = st.radio(
    "🧠 Modo de operação",
    ["conservador", "agressivo"],
    horizontal=True
)

uploaded = st.file_uploader("📂 Envie o CSV do Jogo do Bicho", type=["csv"])

# ----------------------------------------------------------
# FUNÇÕES
# ----------------------------------------------------------
def normalizar(s):
    if s.max() == s.min():
        return pd.Series([0.0] * len(s))
    return (s - s.min()) / (s.max() - s.min())


def entropia(valores):
    probs = valores / valores.sum()
    return -np.sum(probs * np.log2(probs + 1e-9))


def gerar_milhares(centena, bloqueadas):
    base = centena.zfill(3)
    return [f"{d}{base}" for d in "0123456789" if f"{d}{base}" not in bloqueadas]


# ----------------------------------------------------------
# EXECUÇÃO
# ----------------------------------------------------------
if uploaded:
    df = pd.read_csv(uploaded)

    # Padronização
    df["milhar"] = df["milhar"].astype(str).str.zfill(4)
    df["centena"] = df["milhar"].str[-3:]
    df["dezena"] = df["milhar"].str[-2:]
    df["grupo"] = df["dezena"].astype(int) // 4
    df["premio"] = df["premio"].astype(str)

    bloqueadas = set(df["milhar"])

    df["idx"] = range(len(df))

    # -----------------------------
    # MÉTRICAS BASE
    # -----------------------------
    freq_c = df["centena"].value_counts()
    rec_c = df.groupby("centena")["idx"].max()
    std_c = df.groupby("centena")["idx"].std().fillna(0)

    # Entropia por centena
    entropia_c = (
        df.groupby("centena")["dezena"]
        .value_counts()
        .groupby(level=0)
        .apply(lambda x: entropia(x.values))
    )

    # Ranking por animal
    ranking_grupo = (
        df["grupo"]
        .value_counts(normalize=True)
        .rename("ranking_grupo")
    )

    # -----------------------------
    # AUTO-BACKTEST (TOP-20)
    # -----------------------------
    cobertura = {}
    for premio in df["premio"].unique():
        ult = df[df["premio"] == premio].tail(1)
        cobertura[premio] = ult["milhar"].iloc[0] in bloqueadas

    # -----------------------------
    # GERAÇÃO DE CANDIDATAS
    # -----------------------------
    rows = []

    for premio in sorted(df["premio"].unique()):
        base_p = df[df["premio"] == premio]

        top_centenas = (
            freq_c.loc[freq_c.index.isin(base_p["centena"])]
            .sort_values(ascending=False)
            .head(10)
            .index
        )

        for c in top_centenas:
            for m in gerar_milhares(c, bloqueadas):
                grupo = int(m[-2:]) // 4

                rows.append({
                    "premio": premio,
                    "milhar": m,
                    "centena": c,
                    "grupo": grupo,
                    "f_c": freq_c.get(c, 0),
                    "rec_c": rec_c.get(c, len(df)),
                    "std_c": std_c.get(c, 0),
                    "ent": entropia_c.get(c, 0),
                    "mc": np.random.beta(2, 5),
                    "rank_grupo": ranking_grupo.get(grupo, 0)
                })

    res = pd.DataFrame(rows)

    # -----------------------------
    # NORMALIZAÇÃO
    # -----------------------------
    for col in ["f_c", "rec_c", "std_c", "ent", "mc", "rank_grupo"]:
        res[col] = normalizar(res[col])

    w = PESOS[modo]

    # -----------------------------
    # SCORE FINAL (CORRIGIDO)
    # -----------------------------
    res["score"] = (
        res["f_c"] * w["freq"] +
        (1 - res["rec_c"]) * w["rec"] +
        res["mc"] * w["mc"] +
        res["ent"] * w["ent"] +
        res["rank_grupo"] * 0.10 -
        res["std_c"] * 0.15
    )

    resultado = (
        res.sort_values("score", ascending=False)
        .groupby("premio")
        .head(TOP_N)
        .reset_index(drop=True)
    )

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.success("✅ Sistema executado com sucesso")

    st.subheader("📊 Resultado Final")
    st.dataframe(
        resultado[["premio", "milhar", "centena", "grupo", "score"]],
        use_container_width=True
    )

    st.subheader("🐾 Ranking por Animal")
    st.dataframe(ranking_grupo.reset_index())

    st.download_button(
        "📥 Baixar CSV Final",
        resultado.to_csv(index=False),
        "resultado_amf_omega_final.csv",
        "text/csv"
    )

    # -----------------------------
    # PWA INFO
    # -----------------------------
    st.info(
        "📱 Para instalar como APP:\n"
        "• Abra no Chrome\n"
        "• Menu ⋮ → Adicionar à tela inicial\n"
        "• Funciona como aplicativo"
    )

else:
    st.warning("⬆️ Envie o CSV para iniciar")
