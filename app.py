import os
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="競馬ランクアプリ v9.6 分布ランク", layout="centered")

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
DEFAULT_FILES = {
    "style": os.path.join(BASE_DIR, "style_stats_4factor.csv"),
    "prevtrack": os.path.join(BASE_DIR, "prevtrack_stats_4factor.csv"),
    "sire": os.path.join(BASE_DIR, "sire_stats_4factor.csv"),
    "damsire": os.path.join(BASE_DIR, "damsire_stats_4factor.csv"),
    "benchmark": os.path.join(BASE_DIR, "benchmark_4factor_condition_ranks.csv"),
}

WEIGHTS = {"style": 37.5, "prevtrack": 22.5, "sire": 15.0, "damsire": 25.0}
STYLE_MAP = {"逃":"逃げ","逃げ":"逃げ","先":"先行","先行":"先行","差":"差し","差し":"差し","追":"追込","追込":"追込","追い込み":"追込"}

RACE_COLUMN_CANDIDATES = {
    "date": ["date","日付","開催日","年月日","日付S"],
    "場所": ["場所","track","競馬場","開催","場名"],
    "raceNo": ["raceNo","race_number","raceNo.","R","レース番号","race_no","レースNo","R番号"],
    "raceName": ["raceName","race_name","レース名"],
    "horseNo": ["horseNo","horse_number","馬番"],
    "horseName": ["horseName","horse_name","馬名"],
    "distance": ["distance","距離"],
    "surface": ["芝ダ","芝・ダ","surface"],
    "style": ["style","脚質","脚質タグ"],
    "prevTrack": ["prevTrack","前走場所","前走場所タグ","前開催","前走競馬場"],
    "sire": ["sire","種牡馬","血統"],
    "damsire": ["damSire","母父馬"],
    "raceLabel": ["レース","race"],
}

def read_csv_any(file_obj_or_path):
    encodings = ["utf-8-sig", "cp932", "shift_jis", "utf-8"]
    last_err = None
    for enc in encodings:
        try:
            if hasattr(file_obj_or_path, "seek"):
                file_obj_or_path.seek(0)
            return pd.read_csv(file_obj_or_path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def norm_text(v):
    if pd.isna(v):
        return ""
    s = unicodedata.normalize("NFKC", str(v)).strip()
    return " ".join(s.split())

def norm_track(v):
    s = norm_text(v)
    return {
        "東京競馬場":"東京","中山競馬場":"中山","中京競馬場":"中京","阪神競馬場":"阪神",
        "京都競馬場":"京都","新潟競馬場":"新潟","福島競馬場":"福島","小倉競馬場":"小倉",
        "札幌競馬場":"札幌","函館競馬場":"函館",
    }.get(s, s)

def norm_surface(v):
    s = norm_text(v)
    if s.startswith("芝"):
        return "芝"
    if s.startswith("ダ") or s == "ダート":
        return "ダ"
    return s

def norm_style(v):
    s = norm_text(v)
    return STYLE_MAP.get(s, s)

def rename_first_match(df, candidates_map):
    df = df.copy()
    rename_map = {}
    for target, candidates in candidates_map.items():
        for c in candidates:
            if c in df.columns:
                rename_map[c] = target
                break
    return df.rename(columns=rename_map)

def parse_race_label(v):
    s = norm_text(v)
    if not s:
        return "", ""
    m = re.search(r"(東京|中山|中京|阪神|京都|新潟|福島|小倉|札幌|函館)\s*(\d{1,2})\s*R", s)
    if m:
        return m.group(1), m.group(2)
    return "", ""

def prepare_race_df(df):
    df = rename_first_match(df, RACE_COLUMN_CANDIDATES)
    for col in ["date","場所","raceNo","raceName","horseNo","horseName","distance","surface","style","prevTrack","sire","damsire","raceLabel"]:
        if col not in df.columns:
            df[col] = ""

    parsed = df["raceLabel"].apply(parse_race_label)
    parsed_track = parsed.apply(lambda x: x[0])
    parsed_no = parsed.apply(lambda x: x[1])
    df["場所"] = np.where(df["場所"].astype(str).str.strip() != "", df["場所"], parsed_track)
    df["raceNo"] = np.where(df["raceNo"].astype(str).str.strip() != "", df["raceNo"], parsed_no)

    df["場所"] = df["場所"].apply(norm_track)
    df["surface"] = df["surface"].apply(norm_surface)
    df["style"] = df["style"].apply(norm_style)
    df["prevTrack"] = df["prevTrack"].apply(norm_track)
    df["sire"] = df["sire"].apply(norm_text)
    df["damsire"] = df["damsire"].apply(norm_text)
    df["horseName"] = df["horseName"].apply(norm_text)
    df["raceName"] = df["raceName"].apply(norm_text)
    df["date"] = df["date"].apply(norm_text)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["raceNo"] = df["raceNo"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df["horseNo"] = df["horseNo"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df["距離表示"] = np.where(
        df["surface"].astype(str).str.strip() != "",
        df["surface"].astype(str) + df["distance"].fillna("").astype(str).str.replace(".0", "", regex=False),
        df["distance"].fillna("").astype(str).str.replace(".0", "", regex=False),
    )
    df["レース"] = df["場所"].astype(str) + df["raceNo"].astype(str) + "R"
    df["レースキー"] = df["レース"].fillna("").astype(str).str.strip()
    return df

def prepare_stat_table(df, value_col, kind):
    out = df.copy()
    out["場所"] = out["場所"].apply(norm_track)
    out["芝ダ"] = out["芝ダ"].apply(norm_surface)
    out["距離"] = pd.to_numeric(out["距離"], errors="coerce")
    out[value_col] = out[value_col].apply(norm_text)
    if kind == "style":
        out[value_col] = out[value_col].apply(norm_style)
    if kind == "prevtrack":
        out[value_col] = out[value_col].apply(norm_track)
    out["母数"] = pd.to_numeric(out["母数"], errors="coerce").fillna(0)
    out["複勝数"] = pd.to_numeric(out["複勝数"], errors="coerce").fillna(0)
    out["補正複勝率"] = (out["複勝数"] + 1) / (out["母数"] + 3)
    fb = out.groupby(["場所","芝ダ","距離"], dropna=False).agg(
        母数合計=("母数","sum"), 複勝数合計=("複勝数","sum")
    ).reset_index()
    fb["条件平均"] = np.where(fb["母数合計"] > 0, fb["複勝数合計"] / fb["母数合計"], np.nan)
    return out, fb

def lookup_score(row, stats_df, fallback_df, race_col, stat_col, kind):
    value = norm_text(row.get(race_col, ""))
    if kind == "style":
        value = norm_style(value)
    if kind == "prevtrack":
        value = norm_track(value)
    cond = (
        (stats_df["場所"] == norm_track(row.get("場所", ""))) &
        (stats_df["芝ダ"] == norm_surface(row.get("surface", ""))) &
        (stats_df["距離"] == pd.to_numeric(row.get("distance", np.nan), errors="coerce")) &
        (stats_df[stat_col] == value)
    )
    hit = stats_df.loc[cond]
    if not hit.empty:
        best = hit.sort_values(["母数","複勝数"], ascending=False).iloc[0]
        return float(best["補正複勝率"]) * 100.0
    fb_cond = (
        (fallback_df["場所"] == norm_track(row.get("場所", ""))) &
        (fallback_df["芝ダ"] == norm_surface(row.get("surface", ""))) &
        (fallback_df["距離"] == pd.to_numeric(row.get("distance", np.nan), errors="coerce"))
    )
    fb = fallback_df.loc[fb_cond]
    if not fb.empty and pd.notna(fb.iloc[0]["条件平均"]):
        return float(fb.iloc[0]["条件平均"]) * 100.0
    return np.nan

def classify_distribution_rank(score, row, bench_df):
    if pd.isna(score):
        return ""
    cond = (
        (bench_df["場所"] == norm_track(row.get("場所", ""))) &
        (bench_df["芝ダ"] == norm_surface(row.get("surface", ""))) &
        (bench_df["距離"] == pd.to_numeric(row.get("distance", np.nan), errors="coerce"))
    )
    hit = bench_df.loc[cond]
    if hit.empty:
        return ""
    x = hit.iloc[0]
    if score >= x["s_cut"]:
        return "S"
    if score >= x["a_cut"]:
        return "A"
    if score >= x["b_cut"]:
        return "B"
    if score >= x["c_cut"]:
        return "C"
    return "D"

def assign_relative_ranks_df(df):
    out = df.copy()
    out["相対評価"] = "C"
    for race_key in out["レースキー"].fillna("").astype(str).unique():
        mask = out["レースキー"].fillna("").astype(str) == race_key
        idx = out.loc[mask].sort_values(["総合点","horseNo"], ascending=[False, True]).index.tolist()
        n = len(idx)
        if n == 0:
            continue
        d_count = max(1, round(n * 0.30)) if n >= 8 else max(1, round(n * 0.20))
        a_count = max(1, round(n * 0.15))
        b_count = max(1, round(n * 0.20))
        if a_count + b_count + d_count >= n:
            d_count = max(1, n - a_count - b_count - 1)
        out.loc[idx, "相対評価"] = "C"
        out.at[idx[0], "相対評価"] = "S"
        cur = 1
        for i in idx[cur:min(n - d_count, cur + a_count)]:
            out.at[i, "相対評価"] = "A"
        cur = min(n - d_count, cur + a_count)
        for i in idx[cur:min(n - d_count, cur + b_count)]:
            out.at[i, "相対評価"] = "B"
        for i in idx[max(0, n - d_count):]:
            out.at[i, "相対評価"] = "D"
    return out

def render_rank_cards(date_val, race_val, race_name_val, dist_text, card_df):
    title = " ".join([x for x in [str(date_val).strip(), str(race_val).strip()] if x]) or str(race_val).strip() or "レース"
    subtitle = " / ".join([x for x in [str(race_name_val).strip(), str(dist_text).strip()] if x])
    css = """
<style>
.kv-wrap{background:linear-gradient(180deg,#071225 0%,#0b1730 100%);border-radius:20px;padding:14px 12px;color:#f5f7fb;box-shadow:0 8px 24px rgba(0,0,0,.20);margin:8px 0 14px 0}
.kv-title{font-size:18px;font-weight:800;line-height:1.15;margin:0 0 4px 0}
.kv-subtitle{font-size:11px;color:#c8d2e8;margin:0 0 10px 0}
.horse-card{display:flex;align-items:center;justify-content:space-between;gap:8px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.04);border-radius:14px;padding:8px 10px;margin:6px 0}
.horse-left{min-width:0;flex:1}
.horse-top{display:flex;align-items:center;gap:6px;min-width:0}
.horse-no{font-size:11px;color:#9fb0d3;font-weight:700;flex:0 0 auto}
.horse-name{font-size:14px;font-weight:800;color:#ffffff;line-height:1.15;word-break:break-all}
.pill-area{display:flex;gap:6px;align-items:center;flex:0 0 auto}
.rank-box{display:flex;flex-direction:column;align-items:center;gap:3px}
.rank-label{font-size:9px;color:#9fb0d3;line-height:1}
.rank-pill{min-width:36px;text-align:center;font-size:16px;font-weight:900;border-radius:10px;padding:6px 10px;border:2px solid #334a76;color:#edf3ff;background:rgba(93,122,183,.12)}
.rank-S{border-color:#e7c65b;color:#fff2ba;background:rgba(231,198,91,.15)}
.rank-A{border-color:#c9d6ef;color:#ffffff;background:rgba(201,214,239,.10)}
.rank-B{border-color:#d9b456;color:#ffe6a3;background:rgba(217,180,86,.12)}
.rank-C{border-color:#3e547e;color:#d7e3ff;background:rgba(93,122,183,.08)}
.rank-D{border-color:#30405f;color:#b8c6e3;background:rgba(70,89,127,.06)}
</style>
"""
    html = css + f'<div class="kv-wrap"><div class="kv-title">{title}</div>'
    if subtitle:
        html += f'<div class="kv-subtitle">{subtitle}</div>'
    for _, row in card_df.iterrows():
        rel_rank = str(row.get("相対","")).strip()
        real_rank = str(row.get("実力","")).strip()
        rel_cls = f"rank-{rel_rank}" if rel_rank in {"S","A","B","C","D"} else "rank-D"
        real_cls = f"rank-{real_rank}" if real_rank in {"S","A","B","C","D"} else "rank-D"
        html += (
            f'<div class="horse-card"><div class="horse-left"><div class="horse-top">'
            f'<div class="horse-no">{str(row.get("馬番","")).strip()}</div>'
            f'<div class="horse-name">{str(row.get("馬名","")).strip()}</div>'
            f'</div></div>'
            f'<div class="pill-area">'
            f'<div class="rank-box"><div class="rank-label">相対</div><div class="rank-pill {rel_cls}">{rel_rank}</div></div>'
            f'<div class="rank-box"><div class="rank-label">実力</div><div class="rank-pill {real_cls}">{real_rank}</div></div>'
            f'</div></div>'
        )
    html += "</div>"
    return html

st.title("競馬ランクアプリ v9.6 分布ランク")
st.write("4項目で内部採点し、相対評価と、同条件の過去3年分布に対する実力評価を並べて表示します。")

with st.sidebar:
    race_file = st.file_uploader("出走馬CSV", type=["csv"], key="race")
    style_file = st.file_uploader("脚質CSV", type=["csv"], key="style")
    prev_file = st.file_uploader("前走場所CSV", type=["csv"], key="prev")
    sire_file = st.file_uploader("種牡馬CSV", type=["csv"], key="sire")
    damsire_file = st.file_uploader("母父馬CSV", type=["csv"], key="dam")
    bench_file = st.file_uploader("分布ランク基準CSV", type=["csv"], key="bench")

st.caption("比重：脚質37.5 / 前走場所22.5 / 種牡馬15 / 母父馬25")
st.caption("左が相対評価、右が実力評価です。実力評価は競馬場×芝ダ×距離ごとの過去3年分布ランクです。")

if race_file is None:
    st.info("まず出走馬CSVをアップロードしてください。")
    st.stop()

def load_default_or_upload(key, uploader):
    if uploader is not None:
        return read_csv_any(uploader)
    path = DEFAULT_FILES[key]
    if os.path.exists(path):
        return read_csv_any(path)
    return None

try:
    race_df_raw = read_csv_any(race_file)
    race_df = prepare_race_df(race_df_raw)

    style_raw = load_default_or_upload("style", style_file)
    prev_raw = load_default_or_upload("prevtrack", prev_file)
    sire_raw = load_default_or_upload("sire", sire_file)
    dam_raw = load_default_or_upload("damsire", damsire_file)
    bench_raw = load_default_or_upload("benchmark", bench_file)

    if any(x is None for x in [style_raw, prev_raw, sire_raw, dam_raw, bench_raw]):
        raise FileNotFoundError("必要CSVが不足しています。4つの過去データCSVと分布ランク基準CSVを確認してください。")

    style_df, style_fb = prepare_stat_table(style_raw, "脚質", "style")
    prev_df, prev_fb = prepare_stat_table(prev_raw, "前走場所", "prevtrack")
    sire_df, sire_fb = prepare_stat_table(sire_raw, "種牡馬", "sire")
    dam_df, dam_fb = prepare_stat_table(dam_raw, "母父馬", "damsire")

    result_df = race_df.copy()
    result_df["脚質点"] = result_df.apply(lambda r: lookup_score(r, style_df, style_fb, "style", "脚質", "style"), axis=1)
    result_df["前走場所点"] = result_df.apply(lambda r: lookup_score(r, prev_df, prev_fb, "prevTrack", "前走場所", "prevtrack"), axis=1)
    result_df["種牡馬点"] = result_df.apply(lambda r: lookup_score(r, sire_df, sire_fb, "sire", "種牡馬", "sire"), axis=1)
    result_df["母父馬点"] = result_df.apply(lambda r: lookup_score(r, dam_df, dam_fb, "damsire", "母父馬", "damsire"), axis=1)
    result_df["総合点"] = (
        result_df["脚質点"] * 0.375 +
        result_df["前走場所点"] * 0.225 +
        result_df["種牡馬点"] * 0.15 +
        result_df["母父馬点"] * 0.25
    ).round(2)

    bench = bench_raw.copy()
    bench["場所"] = bench["場所"].apply(norm_track)
    bench["芝ダ"] = bench["芝ダ"].apply(norm_surface)
    bench["距離"] = pd.to_numeric(bench["距離"], errors="coerce")
    for c in ["s_cut","a_cut","b_cut","c_cut"]:
        bench[c] = pd.to_numeric(bench[c], errors="coerce")

    result_df["実力評価"] = result_df.apply(lambda r: classify_distribution_rank(r["総合点"], r, bench), axis=1)
    result_df = assign_relative_ranks_df(result_df)
    result_df["順位"] = result_df.groupby("レースキー", dropna=False)["総合点"].rank(method="min", ascending=False)
    result_df["順位"] = result_df["順位"].fillna(999).astype(int)
    result_df = result_df.sort_values(["レースキー","順位","horseNo"], ascending=[True,True,True])

    export_df = result_df[["date","レース","raceName","horseNo","horseName","相対評価","実力評価"]].copy()
    export_df = export_df.rename(columns={"date":"日付","raceName":"レース名","horseNo":"馬番","horseName":"馬名","相対評価":"相対","実力評価":"実力"})

    st.success("自動採点が完了しました。")
    tab1, tab2 = st.tabs(["予想結果","元データ確認"])
    with tab1:
        grouped = result_df.groupby(["date","レース","raceName","距離表示"], dropna=False, sort=False)
        for (date_val, race_val, race_name_val, dist_val), g in grouped:
            show_df = g[["horseNo","horseName","相対評価","実力評価"]].copy().rename(columns={"horseNo":"馬番","horseName":"馬名","相対評価":"相対","実力評価":"実力"})
            st.markdown(render_rank_cards(date_val, race_val, race_name_val, dist_val, show_df), unsafe_allow_html=True)
    with tab2:
        st.dataframe(race_df_raw, use_container_width=True, hide_index=True)

    csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("予想結果CSVをダウンロード", data=csv_data, file_name="4factor_distribution_rank_predictions.csv", mime="text/csv")
except Exception as e:
    st.error(f"エラー: {e}")
