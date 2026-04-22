
import os
import re
import unicodedata
import itertools
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="競馬ランクアプリ v9.7 Bet Advisor", layout="centered")

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

# ---- historical ROI summaries embedded from prior 3-year calculations ----
relative_returns = pd.DataFrame([
    ["S",3417,31.34,78.58,247.12,220.97],
    ["A",6672,17.15,50.54,160.53,158.72],
    ["B",9120,7.71,25.39,81.78,87.02],
    ["C",13116,3.16,11.51,32.78,43.51],
    ["D",13647,0.76,3.08,14.89,14.53],
], columns=["ランク","頭数","単勝率","複勝率","単回率","複回率"])

real_returns = pd.DataFrame([
    ["S",2360,33.18,85.21,250.98,240.76],
    ["A",6881,19.27,54.82,177.54,167.76],
    ["B",11457,7.82,25.32,83.60,87.78],
    ["C",13818,2.71,9.91,30.62,38.48],
    ["D",11456,0.49,2.18,8.70,10.19],
], columns=["ランク","頭数","単勝率","複勝率","単回率","複回率"])

combo_returns = pd.DataFrame([
    ["S×S",1852,35.48,86.99,269.15,245.01],
    ["S×A",2139,25.30,78.92,188.43,225.24],
    ["A×S",498,25.30,78.92,188.43,225.24],
    ["A×A",4201,18.85,54.13,178.77,167.65],
    ["B×A",1153,11.54,38.68,121.90,133.53],
    ["B×B",6709,7.63,24.43,82.53,83.77],
], columns=["組み合わせ","頭数","単勝率","複勝率","単回率","複回率"])

quinella_returns = pd.DataFrame([
    ["実力 S×S",592,153,1003.55],
    ["実力 S×A",3932,577,604.04],
    ["実力 A×A",5806,486,375.78],
    ["相対 S×A",6672,1035,631.74],
    ["相対 A×A",3489,215,318.97],
    ["複合 SA×AA",1860,242,520.49],
    ["複合 AA×AA",1499,120,364.47],
], columns=["買い方","点数","的中数","馬連回収率"])

trio_returns = pd.DataFrame([
    ["実力 S×S×S",122,17,4494.84],
    ["実力 S×S×A",1204,78,1162.89],
    ["実力 S×A×A",4076,180,836.25],
    ["実力 S×A×B",13962,284,421.48],
    ["相対 S×A×A",3390,332,1846.27],
    ["相対 S×A×B",17811,585,692.46],
    ["相対 A×A×B",10055,75,203.06],
], columns=["買い方","点数","的中数","三連複回収率"])


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

def pill_class(v):
    if v in {"S","A","B","C","D"}:
        return f"rank-{v}"
    return "rank-D"

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
.rec-wrap{background:#0f1525;border:1px solid #1d2945;border-radius:16px;padding:12px 12px;margin:10px 0 14px 0;color:#eef3ff}
.rec-title{font-size:15px;font-weight:800;margin:0 0 8px 0}
.rec-sub{font-size:11px;color:#b8c6e3;margin:0 0 8px 0}
.rec-line{font-size:12px;line-height:1.5;margin:3px 0}
.small-table td,.small-table th{font-size:12px!important}
</style>
"""
    html = css + f'<div class="kv-wrap"><div class="kv-title">{title}</div>'
    if subtitle:
        html += f'<div class="kv-subtitle">{subtitle}</div>'
    for _, row in card_df.iterrows():
        rel_rank = str(row.get("相対","")).strip()
        real_rank = str(row.get("実力","")).strip()
        rel_cls = pill_class(rel_rank)
        real_cls = pill_class(real_rank)
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

def horse_list_text(df):
    if df.empty:
        return "なし"
    return " / ".join([f"{r['horseNo']} {r['horseName']}" for _, r in df.iterrows()])

def pair_text(df1, df2=None, same_group=False):
    pairs = []
    if same_group:
        rows = list(df1[["horseNo","horseName"]].itertuples(index=False, name=None))
        for a, b in itertools.combinations(rows, 2):
            pairs.append(f"{a[0]}-{b[0]} ({a[1]} / {b[1]})")
    else:
        if df2 is None:
            return "なし"
        rows1 = list(df1[["horseNo","horseName"]].itertuples(index=False, name=None))
        rows2 = list(df2[["horseNo","horseName"]].itertuples(index=False, name=None))
        used = set()
        for a in rows1:
            for b in rows2:
                if a[0] == b[0]:
                    continue
                key = tuple(sorted([str(a[0]), str(b[0])]))
                if key in used:
                    continue
                used.add(key)
                pairs.append(f"{key[0]}-{key[1]}")
    return " / ".join(pairs[:30]) if pairs else "なし"

def trio_text(groups):
    vals = []
    seen = set()
    lists = [list(g[["horseNo","horseName"]].itertuples(index=False, name=None)) for g in groups]
    for a in lists[0]:
        for b in lists[1]:
            for c in lists[2]:
                nums = [str(a[0]), str(b[0]), str(c[0])]
                if len(set(nums)) < 3:
                    continue
                key = tuple(sorted(nums))
                if key in seen:
                    continue
                seen.add(key)
                vals.append("-".join(key))
    return " / ".join(vals[:40]) if vals else "なし"

def composite_label(rel, real):
    return f"{rel}{real}"

def build_recommendations(g):
    g = g.sort_values(["順位","horseNo"], ascending=[True, True]).copy()
    rel_s = g[g["相対評価"]=="S"]
    rel_a = g[g["相対評価"]=="A"]
    rel_b = g[g["相対評価"]=="B"]
    real_s = g[g["実力評価"]=="S"]
    real_a = g[g["実力評価"]=="A"]
    real_b = g[g["実力評価"]=="B"]

    g["複合"] = [composite_label(r, t) for r, t in zip(g["相対評価"], g["実力評価"])]
    ss = g[g["複合"]=="SS"]
    sa = g[g["複合"]=="SA"]
    aa = g[g["複合"]=="AA"]

    lines = []
    lines.append(("単複おすすめ①", f"相対S×実力S/A", horse_list_text(g[(g['相対評価']=='S') & (g['実力評価'].isin(['S','A']))]), "複回率 225〜245%帯"))
    lines.append(("単複おすすめ②", f"相対A×実力A以上", horse_list_text(g[(g['相対評価']=='A') & (g['実力評価'].isin(['S','A']))]), "複回率 167%級"))

    lines.append(("馬連おすすめ①", "相対 S×A", pair_text(rel_s, rel_a), "馬連回収率 631.74%"))
    lines.append(("馬連おすすめ②", "実力 S×A", pair_text(real_s, real_a), "馬連回収率 604.04%"))
    lines.append(("馬連おすすめ③", "実力 S×S", pair_text(real_s, same_group=True), "馬連回収率 1003.55%"))
    lines.append(("馬連おすすめ④", "複合 SA×AA", pair_text(sa, aa), "馬連回収率 520.49%"))

    lines.append(("三連複おすすめ①", "相対 S×A×A", trio_text([rel_s, rel_a, rel_a]), "三連複回収率 1846.27%"))
    lines.append(("三連複おすすめ②", "実力 S×S×A", trio_text([real_s, real_s, real_a]), "三連複回収率 1162.89%"))
    lines.append(("三連複おすすめ③", "実力 S×A×A", trio_text([real_s, real_a, real_a]), "三連複回収率 836.25%"))
    lines.append(("三連複おすすめ④", "相対 S×A×B", trio_text([rel_s, rel_a, rel_b]), "三連複回収率 692.46%"))
    return lines

def load_default_or_upload(key, uploader):
    if uploader is not None:
        return read_csv_any(uploader)
    path = DEFAULT_FILES[key]
    if os.path.exists(path):
        return read_csv_any(path)
    return None

st.title("競馬ランクアプリ v9.7 Bet Advisor")
st.write("4項目で内部採点し、相対評価・実力評価に加えて、3年分の回収率集計をもとに買い目候補を表示します。")

with st.sidebar:
    race_file = st.file_uploader("出走馬CSV", type=["csv"], key="race")
    style_file = st.file_uploader("脚質CSV", type=["csv"], key="style")
    prev_file = st.file_uploader("前走場所CSV", type=["csv"], key="prev")
    sire_file = st.file_uploader("種牡馬CSV", type=["csv"], key="sire")
    damsire_file = st.file_uploader("母父馬CSV", type=["csv"], key="dam")
    bench_file = st.file_uploader("分布ランク基準CSV", type=["csv"], key="bench")

st.caption("比重：脚質37.5 / 前走場所22.5 / 種牡馬15 / 母父馬25")
st.caption("履歴サマリー：相対・実力・複合の単複回収率、馬連回収率、三連複回収率を表示。")

if race_file is None:
    st.info("まず出走馬CSVをアップロードしてください。")
    st.stop()

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

    tab1, tab2, tab3 = st.tabs(["予想結果","おすすめ買い目","回収率サマリー"])

    with tab1:
        grouped = result_df.groupby(["date","レース","raceName","距離表示"], dropna=False, sort=False)
        for (date_val, race_val, race_name_val, dist_val), g in grouped:
            show_df = g[["horseNo","horseName","相対評価","実力評価"]].copy().rename(columns={"horseNo":"馬番","horseName":"馬名","相対評価":"相対","実力評価":"実力"})
            st.markdown(render_rank_cards(date_val, race_val, race_name_val, dist_val, show_df), unsafe_allow_html=True)

    with tab2:
        grouped = result_df.groupby(["date","レース","raceName","距離表示"], dropna=False, sort=False)
        for (date_val, race_val, race_name_val, dist_val), g in grouped:
            st.markdown(f"### {date_val} {race_val} {race_name_val} {dist_val}")
            recs = build_recommendations(g)
            for title, rule, body, stat in recs:
                st.markdown(f"**{title}**")
                st.write(f"条件: {rule}")
                st.write(f"候補: {body}")
                st.caption(stat)
            st.divider()

    with tab3:
        st.markdown("### 相対ランク別 単複回収率")
        st.dataframe(relative_returns, use_container_width=True, hide_index=True)
        st.markdown("### 実力ランク別 単複回収率")
        st.dataframe(real_returns, use_container_width=True, hide_index=True)
        st.markdown("### 複合ランク別 単複回収率（主要）")
        st.dataframe(combo_returns, use_container_width=True, hide_index=True)
        st.markdown("### 馬連回収率（主要）")
        st.dataframe(quinella_returns, use_container_width=True, hide_index=True)
        st.markdown("### 三連複回収率（主要）")
        st.dataframe(trio_returns, use_container_width=True, hide_index=True)

    csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("予想結果CSVをダウンロード", data=csv_data, file_name="bet_advisor_predictions.csv", mime="text/csv")

except Exception as e:
    st.error(f"エラー: {e}")
