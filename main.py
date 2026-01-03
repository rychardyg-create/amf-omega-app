# =========================================================
# 🧠 AMF-OMEGA PRIME — VERSÃO CORRIGIDA E FUNCIONAL (Streamlit Cloud)
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ---------------- CONFIG ----------------
TOP_N = 5
N_MC = 10000
HALF_LIFE = 45  # dias para decaimento de recência
SEED = 42
PESO_FILE = "pesos.json"

np.random.seed(SEED)

# ------------- PESOS DINÂMICOS ----------
DEFAULT_PESOS = {
    "freq_m": 0.25,
    "freq_c": 0.20,
    "rec": 0.15,
    "mc": 0.15,
    "atraso": 0.10,
    "entropia": 0.05,
    "patterns": 0.10
}

def carregar_pesos():
    if os.path.exists(PESO_FILE):
        try:
            return json.load(open(PESO_FILE))
        except:
            pass
    return DEFAULT_PESOS.copy()

def salvar_pesos(p):
    json.dump(p, open(PESO_FILE, "w"))

pesos = carregar_pesos()

# ---------------- FUNÇÕES ---------------
def decay(days):
    return np.exp(-np.log(2) * days / HALF_LIFE) if days >= 0 else 0

def entropia(vals):
    p = np.array(vals)
    p = p[p > 0]
    if len(p) == 0:
        return 0
    p = p / p.sum()
    return -np.sum(p * np.log2(p))

def calculate_patterns(mil):
    digits = [int(d) for d in mil]
    sum_digits = sum(digits)
    parity_count = sum(d % 2 for d in digits)
    repeats = len(digits) - len(set(digits))
    # Normalizados
    return (
        sum_digits / 36,        # máx 9+9+9+9=36
        parity_count / 4,       # máx 4 pares/ímpares
        repeats / 3             # máx 3 repetições (ex: 1111)
    )

# ---------------- UI --------------------
st.set_page_config(page_title="AMF-OMEGA PRIME", layout="wide")
st.title("🧠 AMF-OMEGA PRIME")
st.caption("Sistema adaptativo avançado para jogo do bicho — Compatível com Streamlit Cloud")

csv = st.file_uploader("📤 Envie o CSV com colunas: concurso, data, premio, milhar, grupo, animal", type="csv")

if not csv:
    st.info("👆 Faça upload do seu arquivo CSV para começar a análise.")
    st.stop()

# ---------------- CARREGAMENTO E TRATAMENTO ----------------
df = pd.read_csv(csv)

# Corrige possíveis nomes de colunas
df.columns = [col.strip().lower() for col in df.columns]

required_cols = ["concurso", "data", "premio", "milhar"]
missing = [col for col in required_cols if col not in df.columns]
if missing:
    st.error(f"Colunas obrigatórias faltando: {missing}")
    st.stop()

df["milhar"] = df["milhar"].astype(str).str.zfill(4)
df["centena"] = df["milhar"].str[-3:]
df["data"] = pd.to_datetime(df["data"], errors="coerce")
df = df.sort_values(["data", "premio"]).reset_index(drop=True)

# Remove linhas com data inválida
df = df.dropna(subset=["data"])

milhares_usadas = set(df["milhar"])

# ---------------- PROCESSAMENTO POR PRÊMIO ----------------
resultados = []

for premio in sorted(df["premio"].unique()):

    base = df[df["premio"] == premio].copy()

    if len(base) < 10:
        continue  # Poucos dados, pula

    # Frequências
    freq_m = base["milhar"].value_counts(normalize=True)
    freq_c = base["centena"].value_counts(normalize=True)

    # Recência e atraso
    ultima_data = base["data"].max()
    rec_mil = base.groupby("milhar")["data"].max().apply(lambda d: decay((ultima_data - d).days))
    atraso_cent = base.groupby("centena")["data"].max().apply(lambda d: (ultima_data - d).days)

    # Monte Carlo ponderado por atraso (inverso)
    atrasos_norm = (atraso_cent - atraso_cent.min()) / (atraso_cent.max() - atraso_cent.min() + 1)
    probs_mc = freq_c.values * (1 + atrasos_norm.values)  # mais peso para atrasadas
    probs_mc /= probs_mc.sum()
    mc_draws = np.random.choice(freq_c.index, size=N_MC, p=probs_mc)
    mc_score = pd.Series(mc_draws).value_counts(normalize=True)

    candidatos = []

    for cent in freq_c.index:
        for d in range(10):
            mil = f"{d}{cent}"
            if mil in milhares_usadas:
                continue

            sum_norm, par_norm, rep_norm = calculate_patterns(mil)
            patterns_score = (sum_norm + par_norm + rep_norm) / 3

            candidatos.append({
                "premio": premio,
                "milhar": mil,
                "centena": cent,
                "freq_m": freq_m.get(mil, 0),
                "freq_c": freq_c.get(cent, 0),
                "rec": rec_mil.get(mil, 0),
                "mc": mc_score.get(cent, 0),
                "atraso": atraso_cent.get(cent, 0),
                "patterns": patterns_score
            })

    if not candidatos:
        continue

    cand = pd.DataFrame(candidatos)

    # Entropia por centena
    ent_group = cand.groupby("centena")["freq_c"].apply(entropia)
    cand["entropia"] = cand["centena"].map(ent_group)

    # Normalização das colunas
    cols_to_norm = ["freq_m", "freq_c", "rec", "mc", "atraso", "entropia", "patterns"]
    for col in cols_to_norm:
        max_val = cand[col].max()
        if max_val > 0:
            cand[col] = cand[col] / max_val

    # Score final
    cand["score"] = (
        pesos["freq_m"] * cand["freq_m"] +
        pesos["freq_c"] * cand["freq_c"] +
        pesos["rec"] * cand["rec"] +
        pesos["mc"] * cand["mc"] +
        pesos["atraso"] * cand["atraso"] +
        pesos["entropia"] * cand["entropia"] +
        pesos["patterns"] * cand["patterns"]
    )

    top = cand.sort_values("score", ascending=False).head(TOP_N)
    resultados.append(top)

if not resultados:
    st.error("Não foi possível gerar previsões. Verifique se o CSV tem dados suficientes.")
    st.stop()

final = pd.concat(resultados).sort_values(["premio", "score"], ascending=[True, False])

st.subheader("📊 PREVISÃO FINAL")
st.dataframe(final[["premio", "milhar", "centena", "score"]].round(4), use_container_width=True)

st.download_button(
    label="📥 Baixar previsões como CSV",
    data=final.to_csv(index=False).encode(),
    file_name=f"previsao_amf_omega_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv"
)

# ---------------- AUTO-APRENDIZADO ----------------
st.subheader("🧠 Auto-aprendizado")

acertou = st.selectbox(
    "Quantas das previsões acertaram centena ou milhar?",
    ["Nenhuma", "1 ou 2", "3 ou mais"]
)

if st.button("Atualizar pesos com feedback"):
    ajuste = {"Nenhuma": -0.06, "1 ou 2": 0.04, "3 ou mais": 0.08}[acertou]

    for k in pesos:
        pesos[k] = max(0.05, pesos[k] + ajuste)

    total = sum(pesos.values())
    for k in pesos:
        pesos[k] /= total

    salvar_pesos(pesos)
    st.success("Pesos atualizados com sucesso! Sistema está aprendendo. ✔️")
    st.write("Pesos atuais:", pesos)

st.info("⚠️ Lembre-se: jogo é aleatório. Use com responsabilidade e apenas para diversão.")
