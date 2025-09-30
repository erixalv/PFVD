# app_dash_enem.py
import json, os
import pandas as pd
import numpy as np
import geopandas as gpd
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

NOTA_COLS = ["NU_NOTA_MT","NU_NOTA_REDACAO","NU_NOTA_LC","NU_NOTA_CH","NU_NOTA_CN"]

def ensure_columns(df):
    if "SG_UF_PROVA" in df.columns and "Estado" not in df.columns:
        df = df.rename(columns={"SG_UF_PROVA":"Estado"})
    df["Estado"] = (df["Estado"].astype(str).str.upper().str.strip())
    for c in [c for c in NOTA_COLS if c in df.columns]:
        if df[c].dtype == "object":
            df[c] = (df[c].astype(str)
                           .str.replace(r"\.", "", regex=True)
                           .str.replace(",", ".", regex=False))
        df[c] = pd.to_numeric(df[c], errors="coerce")
    present = [c for c in NOTA_COLS if c in df.columns]
    if present: df["nota_media"] = df[present].mean(axis=1)
    return df

# --- Dados
df = pd.read_csv("./DADOS/resultados.csv", sep=";", encoding="latin1")
df = ensure_columns(df)

gdf_est = gpd.read_file("./geodata/brazil_states.json")
geojson_est = json.loads(gdf_est.to_json())
valid_ufs = set(gdf_est["abbrev_state"])

gdf_all_munis = gpd.read_file("./geodata/brazil_municipalities.json")

metrics_available = [c for c in (NOTA_COLS+["nota_media"]) if c in df.columns]

def agg_series(s, how):
    if how=="mean": return s.mean()
    if how=="median": return s.median()
    if how=="p90": return s.quantile(0.90)
    if how=="p10": return s.quantile(0.10)
    return s.mean()

# --- App
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "ENEM — Dash Drill-Down"

# --- Layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("ENEM — Mapa interativo (Dash)", className="text-center mb-4"), width=12)
    ]),
    # Painel de filtros
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Métrica"),
                    dcc.Dropdown(metrics_available, "nota_media", id="metric")
                ], md=3),
                dbc.Col([
                    html.Label("Agregação"),
                    dcc.Dropdown(["mean","median","p90","p10"], "mean", id="agg_fun")
                ], md=3),
                dbc.Col([
                    html.Label("Mínimo de N por município"),
                    dcc.Input(id="min_n", type="number", value=10, min=1, step=1,
                              style={"width":"100%"})
                ], md=3),
                dbc.Col([
                    html.Label("Estados (UF)"),
                    dcc.Dropdown(sorted(df["Estado"].unique()), multi=True, id="ufs")
                ], md=3),
            ])
        ])
    ], className="mb-4"),
    # Mapas e tabelas
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="map_estados"))), md=12, className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(html.Div(id="uf_selecionada", className="fw-bold mb-2"), md=12)
    ]),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(dcc.Graph(id="map_municipios"))), md=12, className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Tabela por município", className="mb-3"),
            dcc.Loading(dcc.Graph(id="tbl_municipios"), type="dot")
        ])), md=12, className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H4("Tabela por estado", className="mb-3"),
            dcc.Loading(dcc.Graph(id="tbl_estados"), type="dot")
        ])), md=12, className="mb-4")
    ])
], fluid=True)

# --- Callbacks
@app.callback(
    Output("map_estados","figure"),
    Output("tbl_estados","figure"),
    Input("metric","value"),
    Input("agg_fun","value"),
    Input("ufs","value"),
)
def render_estados(metric, agg_fun, sel_ufs):
    fdf = df[df["Estado"].isin(sel_ufs)] if sel_ufs else df
    agg_estado = (
        fdf.groupby("Estado", as_index=False)
           .agg(media_estado=(metric, lambda x: agg_series(x, agg_fun)),
                qtde=("Estado","size"))
    )
    df_plot = (pd.DataFrame({"abbrev_state": sorted(list(valid_ufs))})
               .merge(agg_estado.rename(columns={"Estado":"abbrev_state"}),
                      on="abbrev_state", how="left"))
    fig_est = px.choropleth(
        df_plot,
        geojson=geojson_est,
        locations="abbrev_state",
        featureidkey="properties.abbrev_state",
        color="media_estado",
        hover_data={"abbrev_state":True,"media_estado":":.2f","qtde":":,.0f"},
        labels={"media_estado":f"Média — {metric}","qtde":"Registros","abbrev_state":"UF"},
        color_continuous_scale="Viridis"
    )
    fig_est.update_geos(fitbounds="geojson", visible=False)
    fig_est.update_layout(margin=dict(l=0,r=0,t=0,b=0), template="plotly_white")
    # Tabela por estado
    tbl = agg_estado.sort_values("media_estado", ascending=False)
    fig_tbl = px.scatter(
        tbl, x="Estado", y="media_estado", size="qtde",
        title="Estados — média vs. qtde"
    )
    fig_tbl.update_traces(mode="markers+text", text=tbl["Estado"], textposition="top center")
    fig_tbl.update_layout(template="plotly_white")
    return fig_est, fig_tbl

@app.callback(
    Output("uf_selecionada","children"),
    Output("map_municipios","figure"),
    Output("tbl_municipios","figure"),
    Input("map_estados","clickData"),
    Input("metric","value"),
    Input("agg_fun","value"),
    Input("min_n","value"),
    Input("ufs","value"),
    prevent_initial_call=True
)
def render_munis(clickData, metric, agg_fun, min_n, sel_ufs):
    uf = None
    if clickData and "points" in clickData:
        uf = clickData["points"][0].get("location")
    if not uf:
        return "Nenhuma UF selecionada.", px.scatter(title=""), px.scatter(title="")
    fdf = df[(df["Estado"]==uf) & (df["Estado"].isin(sel_ufs) if sel_ufs else True)]
    if fdf.empty:
        return f"UF selecionada: {uf}. (sem dados)", px.scatter(title=""), px.scatter(title="")

    muni_code_col = next((c for c in ["CO_MUNICIPIO_PROVA","CO_MUNICIPIO_RESIDENCIA","CO_MUNICIPIO_ESC"] if c in fdf.columns), None)
    gdf_uf = gdf_all_munis[gdf_all_munis["abbrev_state"]==uf]
    geojson_muni = json.loads(gdf_uf.to_json())

    if muni_code_col:
        tmp = fdf[fdf[muni_code_col].notna()].copy()
        tmp[muni_code_col] = pd.to_numeric(tmp[muni_code_col], errors="coerce").astype("Int64")
        agg_muni = (tmp.groupby(muni_code_col, as_index=False)
                        .agg(media_muni=(metric, lambda x: agg_series(x, agg_fun)),
                             qtde=("Estado","size"))
                        .rename(columns={muni_code_col:"code_muni"}))
        agg_muni["media_muni"] = np.where(agg_muni["qtde"]>=min_n, agg_muni["media_muni"], np.nan)
        aux = gdf_uf[["code_muni","name_muni"]]
        mapa_df = aux.merge(agg_muni, on="code_muni", how="left")
        locations = "code_muni"
        feature_key = "properties.code_muni"
    else:
        name_col = next((c for c in ["NO_MUNICIPIO_PROVA","NO_MUNICIPIO_RESIDENCIA","NO_MUNICIPIO_ESC"] if c in fdf.columns), None)
        if not name_col:
            return f"UF {uf}: sem colunas de município.", px.scatter(title=""), px.scatter(title="")
        agg_muni = (fdf.groupby(name_col, as_index=False)
                        .agg(media_muni=(metric, lambda x: agg_series(x, agg_fun)),
                             qtde=("Estado","size"))
                        .rename(columns={name_col:"name_muni"}))
        agg_muni["media_muni"] = np.where(agg_muni["qtde"]>=min_n, agg_muni["media_muni"], np.nan)
        aux = gdf_uf[["name_muni"]]
        mapa_df = aux.merge(agg_muni, on="name_muni", how="left")
        locations = "name_muni"
        feature_key = "properties.name_muni"

    fig_m = px.choropleth(
        mapa_df,
        geojson=geojson_muni,
        locations=locations,
        featureidkey=feature_key,
        color="media_muni",
        hover_data={
            "name_muni": True,
            "qtde": ":,.0f",
            "media_muni":":.2f",
            "code_muni": False   # esconde o código
        },
        labels={"media_muni":f"Média — {metric}","qtde":"Registros"}
        )

    fig_m.update_geos(fitbounds="locations", visible=False)
    fig_m.update_layout(margin=dict(l=0,r=0,t=0,b=0), template="plotly_white")

    tbl = mapa_df.sort_values("media_muni", ascending=False, na_position="last")
    fig_tbl = px.bar(tbl.head(50), x="name_muni", y="media_muni",
                     title=f"Municípios de {uf} (top 50)")
    fig_tbl.update_layout(xaxis_tickangle=45, template="plotly_white")

    return f"UF selecionada: {uf}", fig_m, fig_tbl

if __name__ == "__main__":
    app.run(debug=True)
