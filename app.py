
import os
import io
import re
import unicodedata
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="競馬ランクアプリ v11.2 Straight Core", layout="centered")

def norm_text(v):
    if pd.isna(v):
        return ""
    s = unicodedata.normalize("NFKC", str(v)).strip()
    return " ".join(s.split())

def norm_track(v):
    s = norm_text(v)
    mapping = {
        "東京競馬場":"東京","中山競馬場":"中山","中京競馬場":"中京","阪神競馬場":"阪神",
        "京都競馬場":"京都","新潟競馬場":"新潟","福島競馬場":"福島","小倉競馬場":"小倉",
        "札幌競馬場":"札幌","函館競馬場":"函館"
    }
    return mapping.get(s, s)

def norm_surface(v):
    s = norm_text(v)
    if s.startswith("芝"):
        return "芝"
    if s.startswith("ダ") or s == "ダート":
        return "ダ"
    return s

def parse_race_label(s):
    s = norm_text(s)
    m = re.search(r"([^\d]+)\s*(\d+)R", s)
    if not m:
        return "", np.nan
    return norm_track(m.group(1)), int(m.group(2))

CANDIDATES = {
    "date": ["date","日付","開催日","年月日","日付S"],
    "場所": ["場所","track","競馬場","開催","場名"],
    "raceNo": ["raceNo","race_number","raceNo.","R","レース番号","race_no","レースNo","R番号"],
    "raceName": ["raceName","race_name","レース名"],
    "horseNo": ["horseNo","horse_number","馬番"],
    "horseName": ["horseName","horse_name","馬名"],
    "distance": ["distance","距離"],
    "surface": ["芝ダ","芝・ダ","surface"],
    "raceLabel": ["レース","race"],
    "prevStraight": ["前走直線ロジック点","前走直線点","straightAdjPrev"],
    "prev2Straight": ["前々走直線ロジック点","前々走直線点","straightAdjPrev2"],
    "prev4cAdj": ["前走4角補正","4角補正","prev4cAdj"],
    "prevTrackAdj": ["前走場所補正","場所補正","prevTrackAdj"],
}

def rename_first_match(df, mapping):
    out = df.copy()
    cols_norm = {c: norm_text(c) for c in out.columns}
    for target, cands in mapping.items():
        if target in out.columns:
            continue
        for cand in cands:
            for col, nc in cols_norm.items():
                if nc == cand:
                    out = out.rename(columns={col: target})
                    break
            if target in out.columns:
                break
    return out

def prepare_df(df):
    df = rename_first_match(df, CANDIDATES)
    for col in CANDIDATES.keys():
        if col not in df.columns:
            df[col] = ""
    parsed = df["raceLabel"].apply(parse_race_label)
    df["場所"] = np.where(df["場所"].astype(str).str.strip() != "", df["場所"], parsed.apply(lambda x: x[0]))
    df["raceNo"] = np.where(df["raceNo"].astype(str).str.strip() != "", df["raceNo"], parsed.apply(lambda x: x[1]))
    df["場所"] = df["場所"].apply(norm_track)
    df["surface"] = df["surface"].apply(norm_surface)
    df["horseName"] = df["horseName"].apply(norm_text)
    df["raceName"] = df["raceName"].apply(norm_text)
    df["date"] = df["date"].apply(norm_text)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["raceNo"] = pd.to_numeric(df["raceNo"], errors="coerce")
    df["horseNo"] = pd.to_numeric(df["horseNo"], errors="coerce")
    for col in ["prevStraight","prev2Straight"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(50.0).clip(0, 100)
    for col in ["prev4cAdj","prevTrackAdj"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).clip(-5, 5)
    df["レース"] = df.apply(lambda r: f'{r["場所"]}{int(r["raceNo"])}R' if pd.notna(r["raceNo"]) and norm_text(r["場所"]) else norm_text(r["raceLabel"]), axis=1)
    df["距離表示"] = df.apply(lambda r: f'{r["surface"]}{int(r["distance"])}' if pd.notna(r["distance"]) and norm_text(r["surface"]) else "", axis=1)
    df["レースキー"] = df["date"].astype(str) + "|" + df["レース"].astype(str)
    return df

def absolute_rank(score):
    if pd.isna(score):
        return ""
    if score >= 80:
        return "S"
    if score >= 68:
        return "A"
    if score >= 56:
        return "B"
    if score >= 44:
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
        b_n = max(1, round(n * 0.30))
        for i, ix in enumerate(idx):
            if i < s_n:
                out.at[ix, "相対評価"] = "S"
            elif i < s_n + a_n:
                out.at[ix, "相対評価"] = "A"
            elif i < s_n + a_n + b_n:
                out.at[ix, "相対評価"] = "B"
            elif i < n - max(1, round(n*0.15)):
                out.at[ix, "相対評価"] = "C"
            else:
                out.at[ix, "相対評価"] = "D"
    return out

def confidence_from_score(score):
    if pd.isna(score):
        return 0.0
    # simple calibrated style, bounded
    return float(np.clip(35 + (score - 50) * 1.1, 5, 95))

def recommend_for_race(g):
    g = g.sort_values(["総合点","horseNo"], ascending=[False, True]).reset_index(drop=True)
    single = g.iloc[0]
    # pair/trio need enough support
    usable = g[(g["実力評価"].isin(["S","A"])) & (g["相対評価"].isin(["S","A","B"]))]
    pair_text = "見送り"
    trio_text = "見送り"
    if len(usable) >= 2:
        pair = usable.head(4)
        pair_text = " / ".join([f'{int(single["horseNo"])}-{int(r["horseNo"])}({single["horseName"]}/{r["horseName"]})'
                                for _, r in pair.iloc[1:].iterrows()]) if len(pair) > 1 else "見送り"
    if len(usable) >= 4:
        others = usable.iloc[1:4]
        combos = []
        for _, r1 in others.iloc[[0]].iterrows():
            pass
        names = others.to_dict("records")
        combos = [
            f'{int(single["horseNo"])}-{int(names[0]["horseNo"])}-{int(names[1]["horseNo"])}',
            f'{int(single["horseNo"])}-{int(names[0]["horseNo"])}-{int(names[2]["horseNo"])}',
            f'{int(single["horseNo"])}-{int(names[1]["horseNo"])}-{int(names[2]["horseNo"])}',
        ]
        trio_text = " / ".join(combos) + f' | 軸:{int(single["horseNo"])} {single["horseName"]}'
    return {
        "single": single,
        "pair_text": pair_text,
        "trio_text": trio_text
    }

st.title("競馬ランクアプリ v11.2 Straight Core")
st.write("直線ロジックを主軸にし、前走4角補正と前走場所補正だけを使う版です。血統要素は一旦外し、内容再現性を最優先にします。")
st.caption("主軸：直線ロジック100（前走60%＋前々走40%） / 補正：前走4角 +5/0/-5、前走場所 +5/0/-5")

uploaded = st.file_uploader("予想CSVをアップロード", type=["csv"])

if uploaded is not None:
    raw = pd.read_csv(uploaded, encoding="utf-8-sig")
    df = prepare_df(raw)
    df["直線ロジック点"] = (df["prevStraight"] * 0.60 + df["prev2Straight"] * 0.40).round(2)
    df["補正点"] = (df["prev4cAdj"] + df["prevTrackAdj"]).round(2)
    df["総合点"] = (df["直線ロジック点"] + df["補正点"]).round(2)
    df["実力評価"] = df["総合点"].apply(absolute_rank)
    df = assign_relative_ranks(df)

    st.header("予想結果")
    for rk in df["レースキー"].unique():
        g = df[df["レースキー"] == rk].sort_values(["総合点","horseNo"], ascending=[False, True]).reset_index(drop=True)
        title = f'{g.iloc[0]["date"]} {g.iloc[0]["レース"]} {g.iloc[0]["raceName"]}'
        st.subheader(title)
        rec = recommend_for_race(g)
        single = rec["single"]
        st.markdown("### 単複おすすめ1")
        st.write(f'候補: {int(single["horseNo"])} {single["horseName"]}')
        st.caption(f'直線主軸 / 相対{single["相対評価"]} × 実力{single["実力評価"]} / 総合点 {single["総合点"]:.1f} / 参考信頼度 {confidence_from_score(single["総合点"]):.2f}%')
        st.markdown("### 馬連おすすめ1")
        st.write(f'候補: {rec["pair_text"]}')
        st.markdown("### 三連複おすすめ1")
        st.write(f'候補: {rec["trio_text"]}')
        st.divider()

    export_cols = ["date","場所","レース","raceName","horseNo","horseName","距離表示","相対評価","実力評価","直線ロジック点","補正点","総合点"]
    export_df = df[export_cols].rename(columns={
        "date":"日付","raceName":"レース名","horseNo":"馬番","horseName":"馬名","距離表示":"距離",
    })
    csv = export_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button("予想結果CSVをダウンロード", data=csv.encode("utf-8-sig"), file_name="keiba_rank_v112_predictions.csv", mime="text/csv")
else:
    st.info("予想CSVをアップロードすると、v11.2ロジックで評価します。必要列は 日付,場所,芝ダ,距離,レース,レース名,馬番,馬名,前走直線ロジック点,前々走直線ロジック点,前走4角補正,前走場所補正 です。")
