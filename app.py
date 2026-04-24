
import os
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="競馬ランクアプリ v12.0 Multiplier Logic", layout="centered")

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
DEFAULT_FILES = {
    "prev3c": os.path.join(BASE_DIR, "prev3c_category_stats.csv"),
    "prev4c": os.path.join(BASE_DIR, "prev4c_category_stats.csv"),
    "prevtrack": os.path.join(BASE_DIR, "prevtrack_roi_stats.csv"),
}

PACE_LABELS = ["かなり向く", "向く", "普通", "やや不向き", "不向き"]
PACE_MAP = {"かなり向く":1.25, "向く":1.10, "普通":1.00, "やや不向き":0.85, "不向き":0.70}

def norm_text(v):
    if pd.isna(v):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(v)).strip().split())

def norm_track(v):
    s = norm_text(v)
    m = {
        "東京競馬場":"東京","中山競馬場":"中山","中京競馬場":"中京","阪神競馬場":"阪神",
        "京都競馬場":"京都","新潟競馬場":"新潟","福島競馬場":"福島","小倉競馬場":"小倉",
        "札幌競馬場":"札幌","函館競馬場":"函館",
    }
    return m.get(s, s)

def norm_surface(v):
    s = norm_text(v)
    if s.startswith("芝"):
        return "芝"
    if s.startswith("ダ") or s == "ダート":
        return "ダ"
    return s

def read_csv_any(path_or_file):
    last = None
    for enc in ["utf-8-sig", "cp932", "shift_jis", "utf-8"]:
        try:
            if hasattr(path_or_file, "seek"):
                path_or_file.seek(0)
            return pd.read_csv(path_or_file, encoding=enc)
        except Exception as e:
            last = e
    raise last

def parse_race_label(v):
    s = norm_text(v)
    m = re.search(r"(東京|中山|中京|阪神|京都|新潟|福島|小倉|札幌|函館)\s*(\d+)\s*R", s)
    if not m:
        return "", np.nan
    return m.group(1), float(m.group(2))

CANDS = {
    "date":["date","日付","開催日","年月日","日付S"],
    "場所":["場所","track","競馬場","開催","場名"],
    "raceNo":["raceNo","race_number","R","レース番号","race_no","レースNo","R番号"],
    "raceName":["raceName","race_name","レース名"],
    "horseNo":["horseNo","horse_number","馬番"],
    "horseName":["horseName","horse_name","馬名"],
    "distance":["distance","距離"],
    "surface":["芝ダ","芝・ダ","surface"],
    "raceLabel":["レース","race"],
    "prevTrack":["前走競馬場","前走場所","prevTrack"],
    "prevSurface":["前走芝ダ","prevSurface"],
    "prevDistance":["前走距離数値","前走距離","prevDistance"],
    "prev3cCat":["前3角位置カテゴリ","前走3角カテゴリ","prev3cCat"],
    "prev4cCat":["前4角位置カテゴリ","前走4角カテゴリ","prev4cCat"],
    "prevStraight":["前走直線ロジック点","前走直線点","prevStraight"],
    "prev2Straight":["前々走直線ロジック点","前々走直線点","prev2Straight"],
    "paceEval":["展開予想評価","脚質展開評価","paceEval"],
    "straightEval":["直線相性評価","場所直線相性評価","straightEval"],
}

def rename_first_match(df, candidates):
    out = df.copy()
    normalized = {c: norm_text(c) for c in out.columns}
    for target, opts in candidates.items():
        if target in out.columns:
            continue
        found = None
        for o in opts:
            for c, nc in normalized.items():
                if nc == o:
                    found = c
                    break
            if found:
                break
        if found:
            out = out.rename(columns={found: target})
    return out

def prepare_race_df(df):
    df = rename_first_match(df, CANDS)
    for col in CANDS.keys():
        if col not in df.columns:
            df[col] = ""
    parsed = df["raceLabel"].apply(parse_race_label)
    df["場所"] = np.where(df["場所"].astype(str).str.strip() != "", df["場所"], parsed.apply(lambda x: x[0]))
    df["raceNo"] = np.where(df["raceNo"].astype(str).str.strip() != "", df["raceNo"], parsed.apply(lambda x: x[1]))

    for col in ["場所","prevTrack"]:
        df[col] = df[col].apply(norm_track)
    for col in ["surface","prevSurface"]:
        df[col] = df[col].apply(norm_surface)
    for col in ["horseName","raceName","date","prev3cCat","prev4cCat","paceEval","straightEval"]:
        df[col] = df[col].apply(norm_text)
    for col in ["distance","raceNo","horseNo","prevDistance","prevStraight","prev2Straight"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["prevStraight"] = df["prevStraight"].fillna(50.0).clip(0, 100)
    df["prev2Straight"] = df["prev2Straight"].fillna(50.0).clip(0, 100)
    df["距離表示"] = np.where(df["surface"].astype(str) != "", df["surface"] + df["distance"].fillna(0).astype(int).astype(str), "")
    df["レース"] = df.apply(lambda r: f"{r['場所']}{int(r['raceNo'])}R" if pd.notna(r["raceNo"]) and norm_text(r["場所"]) else norm_text(r["raceLabel"]), axis=1)
    df["レースキー"] = df["date"].astype(str) + "|" + df["レース"].astype(str)
    return df

def load_stat_defaults():
    prev3c = read_csv_any(DEFAULT_FILES["prev3c"])
    prev4c = read_csv_any(DEFAULT_FILES["prev4c"])
    prevtrack = read_csv_any(DEFAULT_FILES["prevtrack"])

    prev3c = prev3c.rename(columns={"前走競馬場":"prevTrack","前走芝ダ":"prevSurface","前走距離数値":"prevDistance","前3角位置カテゴリ":"prev3cCat","件数":"count","複勝率":"place_rate"})
    prev4c = prev4c.rename(columns={"前走競馬場":"prevTrack","前走芝ダ":"prevSurface","前走距離数値":"prevDistance","前4角位置カテゴリ":"prev4cCat","件数":"count","複勝率":"place_rate"})
    prevtrack = prevtrack.rename(columns={"競馬場":"場所","芝ダ":"surface","距離数値":"distance","前走場所":"prevTrack","件数":"count","複勝率":"place_rate"})

    for d in [prev3c, prev4c]:
        d["prevTrack"] = d["prevTrack"].apply(norm_track)
        d["prevSurface"] = d["prevSurface"].apply(norm_surface)
        d["prevDistance"] = pd.to_numeric(d["prevDistance"], errors="coerce")
        d["count"] = pd.to_numeric(d["count"], errors="coerce")
        d["place_rate"] = pd.to_numeric(d["place_rate"], errors="coerce")
    prev3c["prev3cCat"] = prev3c["prev3cCat"].apply(norm_text)
    prev4c["prev4cCat"] = prev4c["prev4cCat"].apply(norm_text)

    prevtrack["場所"] = prevtrack["場所"].apply(norm_track)
    prevtrack["surface"] = prevtrack["surface"].apply(norm_surface)
    prevtrack["distance"] = pd.to_numeric(prevtrack["distance"], errors="coerce")
    prevtrack["prevTrack"] = prevtrack["prevTrack"].apply(norm_track)
    prevtrack["count"] = pd.to_numeric(prevtrack["count"], errors="coerce")
    prevtrack["place_rate"] = pd.to_numeric(prevtrack["place_rate"], errors="coerce")
    return prev3c, prev4c, prevtrack

def map_rate_to_coef(rate, min_rate, max_rate):
    if pd.isna(rate):
        return 1.0
    if pd.isna(min_rate) or pd.isna(max_rate) or max_rate <= min_rate:
        return 1.0
    q = (rate - min_rate) / (max_rate - min_rate)
    if q >= 0.80:
        return 1.25
    if q >= 0.60:
        return 1.10
    if q >= 0.40:
        return 1.00
    if q >= 0.20:
        return 0.85
    return 0.70

def hist_coef_prev3c(row, stat):
    sub = stat[(stat["prevTrack"] == row["prevTrack"]) & (stat["prevSurface"] == row["prevSurface"]) & (stat["prevDistance"] == row["prevDistance"])]
    if sub.empty:
        return 1.0
    hit = sub[sub["prev3cCat"] == row["prev3cCat"]]
    rate = hit["place_rate"].iloc[0] if not hit.empty else np.nan
    return map_rate_to_coef(rate, sub["place_rate"].min(), sub["place_rate"].max())

def hist_coef_prev4c(row, stat):
    sub = stat[(stat["prevTrack"] == row["prevTrack"]) & (stat["prevSurface"] == row["prevSurface"]) & (stat["prevDistance"] == row["prevDistance"])]
    if sub.empty:
        return 1.0
    hit = sub[sub["prev4cCat"] == row["prev4cCat"]]
    rate = hit["place_rate"].iloc[0] if not hit.empty else np.nan
    return map_rate_to_coef(rate, sub["place_rate"].min(), sub["place_rate"].max())

def hist_coef_prevtrack(row, stat):
    sub = stat[(stat["場所"] == row["場所"]) & (stat["surface"] == row["surface"]) & (stat["distance"] == row["distance"])]
    if sub.empty:
        return 1.0
    hit = sub[sub["prevTrack"] == row["prevTrack"]]
    rate = hit["place_rate"].iloc[0] if not hit.empty else np.nan
    return map_rate_to_coef(rate, sub["place_rate"].min(), sub["place_rate"].max())

def confidence_from_score(score):
    return float(np.clip(20 + score * 1.2, 5, 95))

def absolute_rank(score):
    if pd.isna(score):
        return ""
    if score >= 65:
        return "S"
    if score >= 56:
        return "A"
    if score >= 47:
        return "B"
    if score >= 38:
        return "C"
    return "D"

def assign_relative_ranks(df):
    out = df.copy()
    out["相対評価"] = ""
    for rk in out["レースキー"].unique():
        idx = out[out["レースキー"] == rk].sort_values(["総合点","horseNo"], ascending=[False, True]).index.tolist()
        n = len(idx)
        if n == 0:
            continue
        s_n = max(1, round(n * 0.10))
        a_n = max(1, round(n * 0.20))
        b_n = max(1, round(n * 0.25))
        tail_n = max(1, round(n * 0.15))
        for i, ix in enumerate(idx):
            if i < s_n:
                out.at[ix, "相対評価"] = "S"
            elif i < s_n + a_n:
                out.at[ix, "相対評価"] = "A"
            elif i < s_n + a_n + b_n:
                out.at[ix, "相対評価"] = "B"
            elif i < n - tail_n:
                out.at[ix, "相対評価"] = "C"
            else:
                out.at[ix, "相対評価"] = "D"
    return out

def recommend_for_race(g):
    g = g.sort_values(["総合点","horseNo"], ascending=[False, True]).reset_index(drop=True)
    single = g.iloc[0]
    pair = "見送り"
    trio = "見送り"
    use = g[(g["実力評価"].isin(["S","A"])) & (g["相対評価"].isin(["S","A","B"]))]
    if len(use) >= 2:
        pair = " / ".join([f'{int(single["horseNo"])}-{int(r["horseNo"])}' for _, r in use.iloc[1:4].iterrows()])
    if len(use) >= 4:
        a, b, c = use.iloc[1], use.iloc[2], use.iloc[3]
        trio = f'{int(single["horseNo"])}-{int(a["horseNo"])}-{int(b["horseNo"])} / {int(single["horseNo"])}-{int(a["horseNo"])}-{int(c["horseNo"])} / {int(single["horseNo"])}-{int(b["horseNo"])}-{int(c["horseNo"])}'
    return single, pair, trio

def render_rank_cards(g):
    badge_map = {"S": "#d4af37", "A": "#c0c0c0", "B": "#cd7f32", "C": "#6b7280", "D": "#374151"}
    rows = []
    for i, (_, r) in enumerate(g.iterrows(), start=1):
        rel_color = badge_map.get(r["相対評価"], "#6b7280")
        pow_color = badge_map.get(r["実力評価"], "#6b7280")
        rows.append(f"""
        <div style="display:grid;grid-template-columns:28px 28px minmax(0,1fr) 44px 44px 56px;gap:6px;align-items:center;
                    padding:6px 4px;border-bottom:1px solid rgba(255,255,255,0.06);">
          <div style="font-size:12px;color:#9fb1d9;font-weight:700;text-align:center;">{i}</div>
          <div style="font-size:13px;color:#cfe0ff;font-weight:800;text-align:center;">{int(r["horseNo"])}</div>
          <div style="font-size:18px;color:white;font-weight:800;line-height:1.1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{r["horseName"]}</div>
          <div style="width:36px;height:36px;border-radius:10px;border:2px solid {rel_color};display:flex;align-items:center;justify-content:center;
                      color:white;font-weight:800;font-size:16px;margin:0 auto;">{r["相対評価"]}</div>
          <div style="width:36px;height:36px;border-radius:10px;border:2px solid {pow_color};display:flex;align-items:center;justify-content:center;
                      color:white;font-weight:800;font-size:16px;margin:0 auto;">{r["実力評価"]}</div>
          <div style="font-size:13px;color:#e8eefc;font-weight:800;text-align:right;">{r["総合点"]:.1f}</div>
        </div>
        """)
    html = f"""
    <div style="background:#081a36;border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:10px 10px 6px 10px;
                box-shadow:0 6px 16px rgba(0,0,0,0.16);">
      <div style="display:grid;grid-template-columns:28px 28px minmax(0,1fr) 44px 44px 56px;gap:6px;align-items:end;
                  padding:0 4px 8px 4px;border-bottom:1px solid rgba(255,255,255,0.10);margin-bottom:4px;">
        <div style="font-size:10px;color:#93a7d3;text-align:center;">順</div>
        <div style="font-size:10px;color:#93a7d3;text-align:center;">番</div>
        <div style="font-size:10px;color:#93a7d3;">馬名</div>
        <div style="font-size:10px;color:#93a7d3;text-align:center;">相対</div>
        <div style="font-size:10px;color:#93a7d3;text-align:center;">実力</div>
        <div style="font-size:10px;color:#93a7d3;text-align:right;">総合</div>
      </div>
      {''.join(rows)}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

st.title("競馬ランクアプリ v12.0 Multiplier Logic")
st.write("直線ロジック本体50点を、展開位置補正と前走場所直線補正の係数で評価する版です。")
st.caption("本体点 = 前走直線30 + 前々走直線20 / 最終点 = 本体点 × 展開位置補正 × 前走場所直線補正")

uploaded = st.file_uploader("予想CSVをアップロード", type=["csv"])

if uploaded is None:
    st.info("必要列: 日付,場所,芝ダ,距離,レース,レース名,馬番,馬名,前走競馬場,前走芝ダ,前走距離数値,前3角位置カテゴリ,前4角位置カテゴリ,前走場所,前走直線ロジック点,前々走直線ロジック点,展開予想評価,直線相性評価")
else:
    prev3c_stat, prev4c_stat, prevtrack_stat = load_stat_defaults()
    df = prepare_race_df(read_csv_any(uploaded))
    df["本体点"] = (df["prevStraight"] * 0.30 + df["prev2Straight"] * 0.20).round(2)
    df["3角履歴係数"] = df.apply(lambda r: hist_coef_prev3c(r, prev3c_stat), axis=1)
    df["4角履歴係数"] = df.apply(lambda r: hist_coef_prev4c(r, prev4c_stat), axis=1)
    df["展開履歴係数"] = ((df["3角履歴係数"] + df["4角履歴係数"]) / 2).round(2)
    df["展開予想係数"] = df["paceEval"].map(PACE_MAP).fillna(1.0)
    df["展開位置補正"] = ((df["展開履歴係数"] + df["展開予想係数"]) / 2).round(2)
    df["前走場所履歴係数"] = df.apply(lambda r: hist_coef_prevtrack(r, prevtrack_stat), axis=1)
    df["直線相性係数"] = df["straightEval"].map(PACE_MAP).fillna(1.0)
    df["前走場所直線補正"] = ((df["前走場所履歴係数"] + df["直線相性係数"]) / 2).round(2)
    df["総合点"] = (df["本体点"] * df["展開位置補正"] * df["前走場所直線補正"]).round(2)
    df["実力評価"] = df["総合点"].apply(absolute_rank)
    df = assign_relative_ranks(df)

    tab1, tab2 = st.tabs(["ランキング", "おすすめ買い目"])
    with tab1:
        st.header("ランキング")
        st.caption("1レースを1枚のコンパクトカードで表示します。横スクロールなし・縦幅もできるだけ圧縮しています。")
        for rk in df["レースキー"].unique():
            g = df[df["レースキー"] == rk].sort_values(["総合点","horseNo"], ascending=[False, True]).reset_index(drop=True)
            st.subheader(f'{g.iloc[0]["date"]} {g.iloc[0]["レース"]} {g.iloc[0]["raceName"]}')
            render_rank_cards(g)
            st.divider()
    with tab2:
        for rk in df["レースキー"].unique():
            g = df[df["レースキー"] == rk].sort_values(["総合点","horseNo"], ascending=[False, True]).reset_index(drop=True)
            st.subheader(f'{g.iloc[0]["date"]} {g.iloc[0]["レース"]} {g.iloc[0]["raceName"]}')
            single, pair, trio = recommend_for_race(g)
            st.markdown("### 単複おすすめ1")
            st.write(f'候補: {int(single["horseNo"])} {single["horseName"]}')
            st.caption(f'相対{single["相対評価"]} × 実力{single["実力評価"]} / 本体点 {single["本体点"]:.1f} / 展開補正 {single["展開位置補正"]:.2f} / 場所直線補正 {single["前走場所直線補正"]:.2f} / 総合点 {single["総合点"]:.2f} / 参考信頼度 {confidence_from_score(single["総合点"]):.2f}%')
            st.markdown("### 馬連おすすめ1")
            st.write(pair)
            st.markdown("### 三連複おすすめ1")
            st.write(trio)
            st.divider()

    export_cols = ["date","場所","レース","raceName","horseNo","horseName","相対評価","実力評価","本体点","展開位置補正","前走場所直線補正","総合点"]
    export_df = df[export_cols].rename(columns={"date":"日付","raceName":"レース名","horseNo":"馬番","horseName":"馬名"})
    csv = export_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("予想結果CSVをダウンロード", data=csv.encode("utf-8-sig"), file_name="keiba_rank_v120_predictions.csv", mime="text/csv")
