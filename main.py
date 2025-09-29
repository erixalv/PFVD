# app_enem_plotly_click_full.py
# Requisitos:
#   pip install streamlit pandas plotly geobr geopandas pyproj shapely rtree streamlit-plotly-events

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from streamlit_plotly_events import plotly_events
import geobr  # polígonos de Estados/Municípios do Brasil

st.set_page_config(page_title="ENEM — Estados e Municípios (Plotly)", layout="wide")
st.title("ENEM — Mapa interativo com drill-down (Estados → Municípios)")
st.caption("Cores vermelho → verde • Clique no estado para abrir o detalhe por município.")

# =============================================================================
# Helpers
# =============================================================================
NOTA_COLS = ["NU_NOTA_MT", "NU_NOTA_REDACAO", "NU_NOTA_LC", "NU_NOTA_CH", "NU_NOTA_CN"]

@st.cache_data(show_spinner=True)
def load_csv(path_or_file) -> pd.DataFrame:
    return pd.read_csv(path_or_file, sep=";", encoding="latin1")

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Renomeia UF
    if "SG_UF_PROVA" in df.columns and "Estado" not in df.columns:
        df = df.rename(columns={"SG_UF_PROVA": "Estado"})
    # Converter notas para numérico (suporta vírgula decimal)
    for c in [c for c in NOTA_COLS if c in df.columns]:
        if df[c].dtype == "object":
            df[c] = (
                df[c].astype(str)
                     .str.replace(".", "", regex=False)    # remove milhar '1.234,56'
                     .str.replace(",", ".", regex=False)   # vírgula → ponto
            )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # nota média
    present = [c for c in NOTA_COLS if c in df.columns]
    if present:
        df["nota_media"] = df[present].mean(axis=1, skipna=True)
    else:
        st.warning("Não encontrei nenhuma coluna de nota nas colunas esperadas.")
    return df

@st.cache_data(show_spinner=True)
def load_states_geojson():
    gdf = geobr.read_state()  # NÃO renomear aqui
    geojson = json.loads(gdf.to_json())
    valid_ufs = {f["properties"]["abbrev_state"] for f in geojson["features"]}
    aux = gdf[["abbrev_state", "name_state"]].copy()
    aux = aux.rename(columns={"abbrev_state": "Estado"})  # só para ajudar em merges/labels
    return geojson, valid_ufs, aux


@st.cache_data(show_spinner=True)
def load_municipios_geojson(uf: str):
    gdf = geobr.read_municipality(code_muni=uf).rename(columns={"abbrev_state": "Estado"})
    geojson = json.loads(gdf.to_json())
    aux = gdf[["Estado", "code_muni", "name_muni"]].copy()
    return geojson, aux

def pick_muni_code_col(df_uf: pd.DataFrame) -> str | None:
    for c in ["CO_MUNICIPIO_PROVA", "CO_MUNICIPIO_RESIDENCIA", "CO_MUNICIPIO_ESC"]:
        if c in df_uf.columns:
            return c
    return None

def pick_muni_name_col(df_uf: pd.DataFrame) -> str | None:
    for c in ["NO_MUNICIPIO_PROVA", "NO_MUNICIPIO_RESIDENCIA", "NO_MUNICIPIO_ESC"]:
        if c in df_uf.columns:
            return c
    return None

# =============================================================================
# Fonte de dados
# =============================================================================
st.sidebar.header("Fonte de dados")
default_path = "./DADOS/resultados.csv"
use_uploaded = st.sidebar.toggle("Usar arquivo enviado (upload)")
uploaded = st.sidebar.file_uploader("CSV (sep=';' | latin1)", type=["csv"])

df = None
if use_uploaded and uploaded is not None:
    df = load_csv(uploaded)
elif os.path.exists(default_path):
    df = load_csv(default_path)

if df is None:
    st.info("Não encontrei '../DADOS/resultados.csv'. Envie um arquivo na barra lateral.")
    st.stop()

df = ensure_columns(df)

# =============================================================================
# Filtros e métricas
# =============================================================================
st.sidebar.header("Filtros")
ufs_all = sorted(df["Estado"].dropna().astype(str).unique()) if "Estado" in df.columns else []
sel_ufs = st.sidebar.multiselect("Estados (UF)", ufs_all, default=ufs_all)

metrics_available = [c for c in (NOTA_COLS + ["nota_media"]) if c in df.columns]
default_idx = metrics_available.index("nota_media") if "nota_media" in metrics_available else 0
metric = st.sidebar.selectbox("Métrica", metrics_available, index=default_idx)

fdf = df[df["Estado"].isin(sel_ufs)] if sel_ufs else df.copy()

# =============================================================================
# KPIs
# =============================================================================
st.subheader("KPIs")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros", f"{len(fdf):,}".replace(",", "."))
c2.metric(f"Média ({metric})", f"{fdf[metric].mean():.2f}" if metric in fdf else "—")
c3.metric(f"Mediana ({metric})", f"{fdf[metric].median():.2f}" if metric in fdf else "—")
c4.metric(f"Desvio Padrão ({metric})", f"{fdf[metric].std():.2f}" if metric in fdf else "—")

# =============================================================================
# Agregação por Estado
# =============================================================================
if "Estado" not in fdf.columns:
    st.error("A coluna 'Estado' é obrigatória (vem de 'SG_UF_PROVA' no seu notebook).")
    st.stop()

agg_estado = (
    fdf.groupby("Estado", as_index=False)
       .agg(media_estado=(metric, "mean"), qtde=("Estado", "size"))
)

# =============================================================================
# Mapa por Estado — clique para ver municípios
# =============================================================================
st.subheader("Mapa por Estado — clique para ver municípios")

geojson_est, valid_ufs, _aux_est = load_states_geojson()

# filtra apenas UFs válidas (existentes no GeoJSON)
if not agg_estado.empty:
    missing = sorted(set(agg_estado["Estado"]) - valid_ufs)
    if missing:
        st.warning(f"UFs no CSV sem geometria no GeoJSON: {missing}")
    agg_estado = agg_estado[agg_estado["Estado"].isin(valid_ufs)]

# SEMPRE alinhar para a chave do geojson e garantir que nunca fique vazio
df_plot = (
    pd.DataFrame({"abbrev_state": sorted(valid_ufs)})  # base com todas as UFs
    .merge(agg_estado.rename(columns={"Estado": "abbrev_state"}),
           on="abbrev_state", how="left")
)

fig_est = px.choropleth(
    df_plot,
    geojson=geojson_est,
    locations="abbrev_state",
    featureidkey="properties.abbrev_state",
    color="media_estado",                          # pode ter NaN (ok)
    color_continuous_scale=[(0.0, "red"), (1.0, "green")],
    hover_data={"abbrev_state": True, "media_estado":":.2f", "qtde": True},
    labels={"media_estado": f"Média — {metric}",
            "qtde":"Registros", "abbrev_state":"UF"}
)

fig_est.update_geos(fitbounds="geojson", visible=False)
fig_est.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                      coloraxis_colorbar=dict(title="Média"))
fig_est.update_traces(hovertemplate="<b>%{location}</b><br>Média: %{z:.2f}<extra></extra>")


clicked = plotly_events(
    fig_est,
    click_event=True, hover_event=False, select_event=False,
    override_height=520, override_width="100%", key="mapa_estados"
)

st.markdown("Dica: clique em um estado no mapa acima para abrir o detalhamento por municípios.")

# Também permitir seleção manual (fallback)
uf_clicked = clicked[0].get("location") if clicked else None
with st.expander("Selecionar UF manualmente (opcional)"):
    uf_manual = st.selectbox("UF", ["—"] + ufs_all, index=0)
    if uf_manual != "—":
        uf_clicked = uf_manual

# =============================================================================
# Drill-down: Municípios do estado clicado
# =============================================================================
if uf_clicked:
    st.markdown(f"### Municípios de **{uf_clicked}** — {metric}")

    df_uf = fdf[fdf["Estado"] == uf_clicked].copy()
    if df_uf.empty:
        st.info("Não há registros para esta UF após os filtros.")
    else:
        muni_code_col = pick_muni_code_col(df_uf)
        geojson_muni, aux_muni = load_municipios_geojson(uf_clicked)

        if muni_code_col:
            # agrega por código do município (IBGE) — preferível
            df_uf = df_uf[df_uf[muni_code_col].notna()].copy()
            df_uf[muni_code_col] = pd.to_numeric(df_uf[muni_code_col], errors="coerce").astype("Int64")
            agg_muni = (
                df_uf.groupby(muni_code_col, as_index=False)
                     .agg(media_muni=(metric, "mean"), qtde=("Estado", "size"))
                     .rename(columns={muni_code_col: "code_muni"})
            )
            mapa_df = aux_muni.merge(agg_muni, on="code_muni", how="left")
            locations = "code_muni"
            feature_key = "properties.code_muni"
            hover_data = {"name_muni": True, "media_muni":":.2f", "qtde": True}
            labels = {"media_muni": f"Média — {metric}", "qtde": "Registros", "name_muni":"Município"}
        else:
            # fallback por nome (menos robusto por homônimos)
            muni_name_col = pick_muni_name_col(df_uf)
            if not muni_name_col:
                st.warning("Não encontrei colunas de município (código ou nome) no CSV para esta UF.")
                st.stop()
            agg_muni = (
                df_uf.groupby(muni_name_col, as_index=False)
                     .agg(media_muni=(metric, "mean"), qtde=("Estado", "size"))
                     .rename(columns={muni_name_col: "name_muni"})
            )
            mapa_df = aux_muni.merge(agg_muni, on="name_muni", how="left")
            locations = "name_muni"
            feature_key = "properties.name_muni"
            hover_data = {"name_muni": True, "media_muni":":.2f", "qtde": True}
            labels = {"media_muni": f"Média — {metric}", "qtde": "Registros", "name_muni":"Município"}

        # Se não houver nenhum valor numérico, ainda renderiza o contorno
        if "media_muni" not in mapa_df or mapa_df["media_muni"].notna().sum() == 0:
            mapa_df["media_muni"] = np.nan

        fig_muni = px.choropleth(
            mapa_df,
            geojson=geojson_muni,
            locations=locations,
            featureidkey=feature_key,
            color="media_muni",
            color_continuous_scale=[(0.0, "red"), (1.0, "green")],
            hover_data=hover_data,
            labels=labels
        )
        fig_muni.update_geos(fitbounds="locations", visible=False)
        fig_muni.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                               coloraxis_colorbar=dict(title="Média"))
        st.plotly_chart(fig_muni, use_container_width=True,
                        config={"scrollZoom": True, "displaylogo": False})

else:
    st.info("Nenhum estado selecionado. Clique no mapa acima ou escolha manualmente no expansor.")

