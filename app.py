import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="競馬 ランクアプリ v6.8 New Logic", layout="wide")

# =========================
# 初期設定
# =========================
DEFAULT_THRESHOLDS = pd.DataFrame([
    {"rank": "S", "min_score": 28.0, "max_score": 999.0},
    {"rank": "A", "min_score": 24.0, "max_score": 28.0},
    {"rank": "B", "min_score": 22.0, "max_score": 24.0},
    {"rank": "C", "min_score": 20.0, "max_score": 22.0},
    {"rank": "D", "min_score": -999.0, "max_score": 20.0},
])

RANK_ORDER = ["S", "A", "B", "C", "D"]

if "history_df" not in st.session_state:
    st.session_state.history_df = None
if "ranked_prediction_df" not in st.session_state:
    st.session_state.ranked_prediction_df = None
if "generated_image_bytes" not in st.session_state:
    st.session_state.generated_image_bytes = None

# =========================
# CSS
# =========================
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #091426 0%, #0b1730 100%);
    color: #f6f9ff;
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

.block-container {
    max-width: 1100px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.2rem;
}

.sub-title {
    color: #d9e6ff;
    font-size: 1.1rem;
    margin-bottom: 1.1rem;
    line-height: 1.8;
}

.info-box {
    background: rgba(177, 210, 255, 0.10);
    border: 1px solid rgba(177, 210, 255, 0.22);
    border-radius: 22px;
    padding: 1.2rem 1.3rem;
    color: #f4f8ff;
    margin-bottom: 1.3rem;
    line-height: 1.9;
    font-size: 1.05rem;
}

.card-wrap {
    background: linear-gradient(180deg, rgba(10,20,40,0.96) 0%, rgba(8,16,32,0.96) 100%);
    border: 1px solid rgba(120, 150, 210, 0.22);
    border-radius: 24px;
    padding: 1.2rem 1.2rem 1rem 1.2rem;
    margin-bottom: 1.3rem;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 10px 30px rgba(0,0,0,0.18);
}

.card-title {
    color: #ffffff;
    font-size: 1.9rem;
    font-weight: 800;
    margin-bottom: 0.8rem;
}

.small-note {
    color: #e5eeff !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
}

p, label, div, span {
    color: #f4f8ff;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] summary,
label[data-testid="stWidgetLabel"] {
    color: #f5f9ff !important;
    font-weight: 600 !important;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 18px !important;
    font-weight: 800 !important;
    padding: 0.8rem 1rem !important;
    border: none !important;
}

.green-btn button {
    background: linear-gradient(90deg, #11b77d, #27c88f) !important;
    color: white !important;
}

.orange-btn button {
    background: linear-gradient(90deg, #c88710, #e7a11c) !important;
    color: white !important;
}

.red-btn button {
    background: linear-gradient(90deg, #ba4d4d, #d95a5a) !important;
    color: white !important;
}

.dark-btn button {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
}

.stButton > button:disabled,
.stDownloadButton > button:disabled {
    background: rgba(255,255,255,0.15) !important;
    color: #dfe9ff !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    opacity: 1 !important;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 0.45rem 0.55rem;
}

[data-testid="stFileUploader"] * {
    color: #f4f8ff !important;
}

[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 16px !important;
}

[data-baseweb="select"] span {
    color: #ffffff !important;
}

[data-testid="stDataFrame"] * {
    color: #f8fbff !important;
}

.metric-card {
    background: linear-gradient(180deg, rgba(12,22,42,0.98) 0%, rgba(10,18,36,0.98) 100%);
    border: 1px solid rgba(130, 160, 220, 0.22);
    border-radius: 24px;
    padding: 1rem 1rem 1.1rem 1rem;
    margin-bottom: 1rem;
}

.metric-label {
    color: #d6e4ff;
    font-size: 1rem;
    margin-bottom: 0.2rem;
}

.metric-value {
    color: white;
    font-size: 2.3rem;
    font-weight: 800;
    line-height: 1.1;
}

.rank-pill {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-weight: 800;
    font-size: 0.85rem;
    color: white;
}
.rank-S { background: #8d67ff; }
.rank-A { background: #1db273; }
.rank-B { background: #2b8dde; }
.rank-C { background: #d99716; }
.rank-D { background: #c65454; }
</style>
""", unsafe_allow_html=True)

# =========================
# 共通関数
# =========================
def normalize_text_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("\u3000", " ", regex=False).str.strip()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "date": "日付",
        "track": "開催",
        "raceNo": "R",
        "race_name": "レース名",
        "raceName": "レース名",
        "horseName": "馬名",
        "horse_name": "馬名",
        "trainer": "調教師",
        "sire": "種牡馬",
        "damSire": "母父馬",
        "surface": "芝ダ",
        "distance": "距離",
        "going": "馬場状態",
        "popularity": "人気",
        "finishPosition": "着順",
        "winOdds": "単勝",
        "verticalScore": "縦軸点",
        "horizontalScore": "横軸点",
        "totalScore": "総合点",
        "rank": "ランク",
        "prevTrack": "前開催",
        "prevDistance": "前距離",
        "prevGoing": "前走馬場状態",
        "prevDate": "前走日付",
        "intervalCategory": "間隔カテゴリ",
        "distanceChange": "距離変化",
        "trackChange": "開催変化",
        "jockeyChange": "騎手変化",
        "prevJockey": "前騎手",
        "jockey": "騎手",
        "headCount": "頭数",
    }
    df = df.rename(columns=rename_map)

    for c in ["馬名", "レース名", "開催", "調教師", "種牡馬", "母父馬", "騎手", "前騎手"]:
        if c in df.columns:
            df[c] = normalize_text_col(df[c])

    return df

def ensure_race_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["日付", "開催", "レース名", "馬名"]:
        if col not in df.columns:
            df[col] = ""
    if "R" not in df.columns:
        df["R"] = ""
    return df

def parse_surface_distance(value):
    if pd.isna(value):
        return "", np.nan
    s = str(value).strip()
    if s.startswith("芝"):
        return "芝", pd.to_numeric(s.replace("芝", ""), errors="coerce")
    if s.startswith("ダ"):
        return "ダート", pd.to_numeric(s.replace("ダ", ""), errors="coerce")
    if s.startswith("障"):
        return "障害", pd.to_numeric(s.replace("障", ""), errors="coerce")
    return "", pd.to_numeric(s, errors="coerce")

def add_surface_distance_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "距離" in df.columns and "芝ダ" not in df.columns:
        parsed = df["距離"].apply(parse_surface_distance)
        df["芝ダ"] = parsed.apply(lambda x: x[0])
        df["距離数値"] = parsed.apply(lambda x: x[1])
    else:
        if "距離" in df.columns:
            df["距離数値"] = pd.to_numeric(df["距離"], errors="coerce")
        else:
            df["距離数値"] = np.nan
    return df

def get_distance_band(distance):
    if pd.isna(distance):
        return "不明"
    d = float(distance)
    if d <= 1400:
        return "短距離"
    if d <= 1700:
        return "マイル"
    if d <= 2000:
        return "中距離"
    if d <= 2400:
        return "中長距離"
    return "長距離"

def get_going_group(going):
    if pd.isna(going):
        return "不明"
    g = str(going).strip()
    if g == "良":
        return "良"
    return "道悪"

def get_interval_category(value):
    if pd.isna(value):
        return "不明"
    s = str(value).strip()
    if s == "":
        return "不明"
    num = pd.to_numeric(s, errors="coerce")
    if pd.notna(num):
        w = float(num)
        if w <= 0:
            return "連闘"
        if w <= 2:
            return "中1〜2週"
        if w <= 5:
            return "中3〜5週"
        if w <= 9:
            return "中6〜9週"
        return "10週以上"
    if "連闘" in s:
        return "連闘"
    if "中1" in s or "中2" in s:
        return "中1〜2週"
    if "中3" in s or "中4" in s or "中5" in s:
        return "中3〜5週"
    if "中6" in s or "中7" in s or "中8" in s or "中9" in s:
        return "中6〜9週"
    return "10週以上"

def get_distance_change(curr, prev):
    curr_num = pd.to_numeric(curr, errors="coerce")
    prev_num = pd.to_numeric(prev, errors="coerce")
    if pd.isna(curr_num) or pd.isna(prev_num):
        return "不明"
    if curr_num > prev_num:
        return "延長"
    if curr_num < prev_num:
        return "短縮"
    return "同距離"

def get_track_change(curr, prev):
    if pd.isna(curr) or pd.isna(prev):
        return "不明"
    return "同場" if str(curr).strip() == str(prev).strip() else "場替わり"

def get_jockey_change(curr, prev):
    if pd.isna(curr) or pd.isna(prev) or str(curr).strip() == "" or str(prev).strip() == "":
        return "不明"
    return "継続" if str(curr).strip() == str(prev).strip() else "乗り替わり"

def calc_place_flag(series):
    s = pd.to_numeric(series, errors="coerce")
    return np.where((s >= 1) & (s <= 3), 1, 0)

def add_rank(score, thresholds):
    for _, row in thresholds.iterrows():
        if score >= row["min_score"] and score < row["max_score"]:
            return row["rank"]
    return "D"

def classify_rank(rank):
    rank = str(rank)
    return {
        "S": "本線",
        "A": "本線",
        "B": "相手",
        "C": "注意",
        "D": "軽視",
    }.get(rank, "軽視")

def rank_badge(rank: str) -> str:
    cls = f"rank-{rank}" if rank in RANK_ORDER else "rank-D"
    return f'<span class="rank-pill {cls}">{rank}</span>'

# =========================
# 集計系
# =========================
def make_axis_summary_tables(history_df: pd.DataFrame):
    df = history_df.copy()

    if "着順" in df.columns:
        df["複勝"] = calc_place_flag(df["着順"])
    else:
        df["複勝"] = 0

    if "単勝" in df.columns:
        odds = pd.to_numeric(df["単勝"], errors="coerce")
        finish = pd.to_numeric(df["着順"], errors="coerce")
        df["単回収"] = np.where(finish == 1, odds, 0)
    else:
        df["単回収"] = 0

    out = {}

    def make_group(keys):
        if not set(keys).issubset(df.columns):
            return pd.DataFrame()
        return (
            df.groupby(keys, dropna=False)
            .agg(件数=("馬名", "count"), 複勝率=("複勝", "mean"), 単回収率=("単回収", "mean"))
            .reset_index()
        )

    out["sire_track"] = make_group(["種牡馬", "開催"])
    out["sire_dist"] = make_group(["種牡馬", "距離帯"])
    out["sire_going"] = make_group(["種牡馬", "馬場区分"])

    out["trainer_track"] = make_group(["調教師", "開催"])
    out["trainer_dist"] = make_group(["調教師", "距離帯"])
    out["trainer_interval"] = make_group(["調教師", "間隔カテゴリ"])
    out["trainer_distchg"] = make_group(["調教師", "距離変化"])
    out["trainer_trackchg"] = make_group(["調教師", "開催変化"])
    out["trainer_jchg"] = make_group(["調教師", "騎手変化"])

    return out

def score_from_summary(base_df: pd.DataFrame, summary_df: pd.DataFrame, keys: list, min_count=10):
    if summary_df.empty:
        return pd.Series([np.nan] * len(base_df), index=base_df.index)
    tmp = summary_df.copy()
    tmp = tmp[tmp["件数"] >= min_count]
    if tmp.empty:
        return pd.Series([np.nan] * len(base_df), index=base_df.index)
    merged = base_df[keys].merge(tmp[keys + ["複勝率"]], on=keys, how="left")
    return merged["複勝率"]

def count_from_summary(base_df: pd.DataFrame, summary_df: pd.DataFrame, keys: list, min_count=1):
    if summary_df.empty:
        return pd.Series([np.nan] * len(base_df), index=base_df.index)
    tmp = summary_df.copy()
    tmp = tmp[tmp["件数"] >= min_count]
    if tmp.empty:
        return pd.Series([np.nan] * len(base_df), index=base_df.index)
    merged = base_df[keys].merge(tmp[keys + ["件数"]], on=keys, how="left")
    return merged["件数"]

def safe_min_count(row, cols):
    vals = []
    for c in cols:
        v = row.get(c, np.nan)
        if pd.notna(v):
            vals.append(float(v))
    if not vals:
        return np.nan
    return int(min(vals))

def build_condition_text(row):
    return (
        f"縦:{row.get('種牡馬','')}×{row.get('開催','')}×{row.get('距離帯','')}×{row.get('馬場区分','')} / "
        f"横:{row.get('調教師','')}×{row.get('間隔カテゴリ','')}×{row.get('距離変化','')}×{row.get('開催変化','')}"
    )

# =========================
# データ整形
# =========================
def prepare_history_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = ensure_race_key_columns(df)
    df = add_surface_distance_columns(df)

    if "距離帯" not in df.columns:
        df["距離帯"] = df["距離数値"].apply(get_distance_band)

    if "馬場区分" not in df.columns:
        if "馬場状態" in df.columns:
            df["馬場区分"] = df["馬場状態"].apply(get_going_group)
        else:
            df["馬場区分"] = "不明"

    if "間隔カテゴリ" not in df.columns:
        if "間隔" in df.columns:
            df["間隔カテゴリ"] = df["間隔"].apply(get_interval_category)
        else:
            df["間隔カテゴリ"] = "不明"

    if "距離変化" not in df.columns:
        if "前距離" in df.columns and "距離数値" in df.columns:
            df["距離変化"] = df.apply(lambda r: get_distance_change(r.get("距離数値"), r.get("前距離")), axis=1)
        else:
            df["距離変化"] = "不明"

    if "開催変化" not in df.columns:
        if "前開催" in df.columns:
            df["開催変化"] = df.apply(lambda r: get_track_change(r.get("開催"), r.get("前開催")), axis=1)
        else:
            df["開催変化"] = "不明"

    if "騎手変化" not in df.columns:
        if "騎手" in df.columns and "前騎手" in df.columns:
            df["騎手変化"] = df.apply(lambda r: get_jockey_change(r.get("騎手"), r.get("前騎手")), axis=1)
        else:
            df["騎手変化"] = "不明"

    return df

def prepare_prediction_df(df: pd.DataFrame, history_df: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = ensure_race_key_columns(df)
    df = add_surface_distance_columns(df)

    if "距離帯" not in df.columns:
        df["距離帯"] = df["距離数値"].apply(get_distance_band)

    if "馬場区分" not in df.columns:
        if "馬場状態" in df.columns:
            df["馬場区分"] = df["馬場状態"].apply(get_going_group)
        else:
            df["馬場区分"] = "不明"

    if "間隔カテゴリ" not in df.columns:
        if "間隔" in df.columns:
            df["間隔カテゴリ"] = df["間隔"].apply(get_interval_category)
        else:
            df["間隔カテゴリ"] = "不明"

    if "距離変化" not in df.columns:
        if "前距離" in df.columns and "距離数値" in df.columns:
            df["距離変化"] = df.apply(lambda r: get_distance_change(r.get("距離数値"), r.get("前距離")), axis=1)
        else:
            df["距離変化"] = "不明"

    if "開催変化" not in df.columns:
        if "前開催" in df.columns:
            df["開催変化"] = df.apply(lambda r: get_track_change(r.get("開催"), r.get("前開催")), axis=1)
        else:
            df["開催変化"] = "不明"

    if "騎手変化" not in df.columns:
        if "騎手" in df.columns and "前騎手" in df.columns:
            df["騎手変化"] = df.apply(lambda r: get_jockey_change(r.get("騎手"), r.get("前騎手")), axis=1)
        else:
            df["騎手変化"] = "不明"

    summaries = make_axis_summary_tables(history_df)

    # 縦軸
    df["score_sire_track"] = score_from_summary(df, summaries["sire_track"], ["種牡馬", "開催"])
    df["score_sire_dist"] = score_from_summary(df, summaries["sire_dist"], ["種牡馬", "距離帯"])
    df["score_sire_going"] = score_from_summary(df, summaries["sire_going"], ["種牡馬", "馬場区分"])

    # 横軸
    df["score_trainer_track"] = score_from_summary(df, summaries["trainer_track"], ["調教師", "開催"])
    df["score_trainer_dist"] = score_from_summary(df, summaries["trainer_dist"], ["調教師", "距離帯"])
    df["score_trainer_interval"] = score_from_summary(df, summaries["trainer_interval"], ["調教師", "間隔カテゴリ"])
    df["score_trainer_distchg"] = score_from_summary(df, summaries["trainer_distchg"], ["調教師", "距離変化"])
    df["score_trainer_trackchg"] = score_from_summary(df, summaries["trainer_trackchg"], ["調教師", "開催変化"])
    df["score_trainer_jchg"] = score_from_summary(df, summaries["trainer_jchg"], ["調教師", "騎手変化"])

    # 母数
    df["count_sire_track"] = count_from_summary(df, summaries["sire_track"], ["種牡馬", "開催"])
    df["count_sire_dist"] = count_from_summary(df, summaries["sire_dist"], ["種牡馬", "距離帯"])
    df["count_sire_going"] = count_from_summary(df, summaries["sire_going"], ["種牡馬", "馬場区分"])
    df["count_trainer_track"] = count_from_summary(df, summaries["trainer_track"], ["調教師", "開催"])
    df["count_trainer_dist"] = count_from_summary(df, summaries["trainer_dist"], ["調教師", "距離帯"])
    df["count_trainer_interval"] = count_from_summary(df, summaries["trainer_interval"], ["調教師", "間隔カテゴリ"])
    df["count_trainer_distchg"] = count_from_summary(df, summaries["trainer_distchg"], ["調教師", "距離変化"])
    df["count_trainer_trackchg"] = count_from_summary(df, summaries["trainer_trackchg"], ["調教師", "開催変化"])
    df["count_trainer_jchg"] = count_from_summary(df, summaries["trainer_jchg"], ["調教師", "騎手変化"])

    vertical_cols = ["score_sire_track", "score_sire_dist", "score_sire_going"]
    horizontal_cols = [
        "score_trainer_track", "score_trainer_dist", "score_trainer_interval",
        "score_trainer_distchg", "score_trainer_trackchg", "score_trainer_jchg"
    ]

    df["縦軸点"] = df[vertical_cols].mean(axis=1, skipna=True) * 100
    df["横軸点"] = df[horizontal_cols].mean(axis=1, skipna=True) * 100
    df["総合点"] = df[["縦軸点", "横軸点"]].mean(axis=1, skipna=True)

    df["縦軸点"] = df["縦軸点"].fillna(0).round(1)
    df["横軸点"] = df["横軸点"].fillna(0).round(1)
    df["総合点"] = df["総合点"].fillna(0).round(1)

    df["ランク"] = df["総合点"].apply(lambda x: add_rank(x, thresholds))
    df["信頼度"] = df["ランク"]
    df["分類"] = df["ランク"].apply(classify_rank)
    df["条件"] = df.apply(build_condition_text, axis=1)

    count_cols = [
        "count_sire_track", "count_sire_dist", "count_sire_going",
        "count_trainer_track", "count_trainer_dist", "count_trainer_interval",
        "count_trainer_distchg", "count_trainer_trackchg", "count_trainer_jchg"
    ]
    df["母数"] = df.apply(lambda r: safe_min_count(r, count_cols), axis=1)

    sort_cols = [c for c in ["日付", "開催", "R", "レース名"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols + ["総合点"], ascending=[True] * len(sort_cols) + [False])
    else:
        df = df.sort_values("総合点", ascending=False)

    return df

# =========================
# レース選択/画像
# =========================
def race_options_from_df(df: pd.DataFrame):
    need_cols = ["日付", "開催", "レース名"]
    if not all(c in df.columns for c in need_cols):
        return []
    cols = [c for c in ["日付", "開催", "R", "レース名"] if c in df.columns]
    temp = df[cols].drop_duplicates().fillna("")
    items = []
    for _, row in temp.iterrows():
        label = f"{row.get('日付','')} | {row.get('開催','')} | {row.get('R','')} | {row.get('レース名','')}"
        items.append((label, row.to_dict()))
    return items

def filter_race_df(df: pd.DataFrame, race_dict: dict) -> pd.DataFrame:
    out = df.copy()
    for col, val in race_dict.items():
        if col in out.columns:
            out = out[out[col].astype(str) == str(val)]
    return out.copy()

def build_race_image_bytes(race_df: pd.DataFrame, title: str) -> bytes:
    show_cols = [c for c in ["馬番", "馬名", "縦軸点", "横軸点", "総合点", "ランク"] if c in race_df.columns]
    if not show_cols:
        show_cols = [c for c in ["馬名", "縦軸点", "横軸点", "総合点", "ランク"] if c in race_df.columns]

    plot_df = race_df[show_cols].copy()

    rows = len(plot_df)
    fig_h = max(4.6, 1.0 + rows * 0.48)

    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor("#081324")
    ax.set_facecolor("#081324")
    ax.axis("off")

    ax.text(
        0.01, 1.03, title,
        fontsize=18, fontweight="bold", color="white",
        transform=ax.transAxes, ha="left", va="bottom"
    )

    table = ax.table(
        cellText=plot_df.values,
        colLabels=plot_df.columns,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.95],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#304664")
        if r == 0:
            cell.set_facecolor("#12284a")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#0c1b33")
            cell.get_text().set_color("white")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def unique_race_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    cols = [c for c in ["日付", "開催", "R", "レース名"] if c in df.columns]
    if not cols:
        return 0
    return len(df[cols].drop_duplicates())

def saved_condition_count(history_df: pd.DataFrame) -> int:
    if history_df is None or history_df.empty:
        return 0
    summaries = make_axis_summary_tables(history_df)
    return int(sum(len(v) for v in summaries.values()))

# =========================
# ヘッダー
# =========================
st.markdown('<div class="main-title">競馬 ランクアプリ<br>v6.8 New Logic</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">見た目は6.8v系の想定で、内部ロジックを新2軸版に差し替えた再構成版です。</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">
判定条件は <b>血統適性 × 厩舎ローテ適性</b> です。<br><br>
縦軸は「血統 × 競馬場・距離・馬場」、横軸は「厩舎 × 競馬場・距離・ローテ」で評価します。<br>
総合点から <b>S〜D</b> を自動判定します。
</div>
""", unsafe_allow_html=True)

with st.expander("ランク基準を見る"):
    st.dataframe(DEFAULT_THRESHOLDS, use_container_width=True)

# =========================
# 過去データ
# =========================
st.markdown('<div class="card-wrap"><div class="card-title">過去レースCSV（収集用）</div></div>', unsafe_allow_html=True)

history_file = st.file_uploader("過去レースCSV", type=["csv"], key="history_uploader", label_visibility="collapsed")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="green-btn">', unsafe_allow_html=True)
    import_history = st.button("過去レースCSVを取り込む", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="dark-btn">', unsafe_allow_html=True)
    make_backup = st.button("JSONバックアップ", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="red-btn">', unsafe_allow_html=True)
    clear_history = st.button("過去データ削除", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if import_history and history_file is not None:
    raw_history = pd.read_csv(history_file)
    prepared_history = prepare_history_df(raw_history)
    st.session_state.history_df = prepared_history
    st.success(f"過去データを取り込みました。件数: {len(prepared_history):,}")

if clear_history:
    st.session_state.history_df = None
    st.success("過去データを削除しました。")

if make_backup and st.session_state.history_df is not None:
    payload = {"records": st.session_state.history_df.to_dict(orient="records")}
    backup_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    st.download_button(
        "JSONをダウンロード",
        data=backup_bytes,
        file_name="keiba_history_backup.json",
        mime="application/json",
        use_container_width=True
    )

if st.session_state.history_df is not None:
    st.markdown(f'<div class="small-note">取り込み済み件数: {len(st.session_state.history_df):,}</div>', unsafe_allow_html=True)
    with st.expander("過去データの先頭を表示"):
        st.dataframe(st.session_state.history_df.head(50), use_container_width=True)
else:
    st.markdown('<div class="small-note">まだ取り込んでいません。</div>', unsafe_allow_html=True)

# =========================
# 予想データ
# =========================
st.markdown('<div class="card-wrap"><div class="card-title">予想レースCSV（画像化用）</div></div>', unsafe_allow_html=True)

pred_file = st.file_uploader("予想レースCSV", type=["csv"], key="pred_uploader", label_visibility="collapsed")

race_labels = []
selected_race_label = None

if pred_file is not None and st.session_state.history_df is not None:
    tmp_pred = pd.read_csv(pred_file)
    tmp_pred = normalize_columns(tmp_pred)
    tmp_pred = ensure_race_key_columns(tmp_pred)
    race_options = race_options_from_df(tmp_pred)
    race_labels = [x[0] for x in race_options]
elif st.session_state.ranked_prediction_df is not None:
    race_options = race_options_from_df(st.session_state.ranked_prediction_df)
    race_labels = [x[0] for x in race_options]
else:
    race_options = []

if race_labels:
    selected_race_label = st.selectbox("対象レース", race_labels)
else:
    st.selectbox("対象レース", ["先にCSVを読み込んでください"], disabled=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="green-btn">', unsafe_allow_html=True)
    import_pred = st.button("予想CSVを読み込む", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="orange-btn">', unsafe_allow_html=True)
    make_image = st.button("画像を作成", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c3:
    st.markdown('<div class="dark-btn">', unsafe_allow_html=True)
    save_image = st.button("画像を保存", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if import_pred:
    if st.session_state.history_df is None:
        st.error("先に過去データCSVを取り込んでください。")
    elif pred_file is None:
        st.error("予想CSVを選択してください。")
    else:
        pred_raw = pd.read_csv(pred_file)
        ranked_pred = prepare_prediction_df(pred_raw, st.session_state.history_df, DEFAULT_THRESHOLDS)
        st.session_state.ranked_prediction_df = ranked_pred
        st.success(f"予想データを読み込みました。件数: {len(ranked_pred):,}")

if st.session_state.ranked_prediction_df is not None:
    ranked_df = st.session_state.ranked_prediction_df.copy()
    current_race_options = race_options_from_df(ranked_df)
    current_race_map = dict(current_race_options)

    show_df = ranked_df.copy()
    if selected_race_label and selected_race_label in current_race_map:
        show_df = filter_race_df(ranked_df, current_race_map[selected_race_label])

    preferred_cols = [c for c in ["馬番", "馬名", "縦軸点", "横軸点", "総合点", "ランク", "種牡馬", "調教師"] if c in show_df.columns]
    st.dataframe(show_df[preferred_cols] if preferred_cols else show_df, use_container_width=True)

    csv_bytes = ranked_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "ランク付きCSVをダウンロード",
        data=csv_bytes,
        file_name="ranked_prediction.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.markdown('<div class="small-note">まだ読み込んでいません。</div>', unsafe_allow_html=True)

if make_image:
    if st.session_state.ranked_prediction_df is None:
        st.error("先に予想CSVを読み込んでください。")
    else:
        ranked_df = st.session_state.ranked_prediction_df.copy()
        current_race_options = race_options_from_df(ranked_df)
        current_race_map = dict(current_race_options)

        if selected_race_label and selected_race_label in current_race_map:
            race_df = filter_race_df(ranked_df, current_race_map[selected_race_label])
            title = selected_race_label
        else:
            race_df = ranked_df.copy()
            title = "レースランキング"

        st.session_state.generated_image_bytes = build_race_image_bytes(race_df, title)
        st.success("画像を作成しました。")

if save_image:
    if st.session_state.generated_image_bytes is None:
        st.error("先に画像を作成してください。")
    else:
        st.download_button(
            "PNGをダウンロード",
            data=st.session_state.generated_image_bytes,
            file_name="keiba_rank_image.png",
            mime="image/png",
            use_container_width=True
        )

# =========================
# 画像プレビュー
# =========================
st.markdown('<div class="card-wrap"><div class="card-title">画像プレビュー</div></div>', unsafe_allow_html=True)

if st.session_state.generated_image_bytes is not None:
    st.image(st.session_state.generated_image_bytes, caption="生成画像", use_container_width=True)
else:
    st.markdown("**予想CSVを読み込んでください**")
    st.markdown("<div style='height:320px;'></div>", unsafe_allow_html=True)

# =========================
# 画像外の条件集計
# =========================
st.markdown('<div class="card-wrap"><div class="card-title">画像外の条件集計</div></div>', unsafe_allow_html=True)

if st.session_state.ranked_prediction_df is not None:
    ranked_df = st.session_state.ranked_prediction_df.copy()
    current_race_options = race_options_from_df(ranked_df)
    current_race_map = dict(current_race_options)

    summary_df = ranked_df.copy()
    if selected_race_label and selected_race_label in current_race_map:
        summary_df = filter_race_df(ranked_df, current_race_map[selected_race_label])

    show_cols = [c for c in ["馬名", "信頼度", "分類", "条件", "母数"] if c in summary_df.columns]
    if show_cols:
        st.dataframe(summary_df[show_cols], use_container_width=True, hide_index=True)
    else:
        st.info("予想CSVを読み込むと表示されます。")
else:
    st.info("予想CSVを読み込むと表示されます。")

# =========================
# 集計状況
# =========================
history_count = len(st.session_state.history_df) if st.session_state.history_df is not None else 0
condition_count = saved_condition_count(st.session_state.history_df)
prediction_race_count = unique_race_count(st.session_state.ranked_prediction_df)
prediction_horse_count = len(st.session_state.ranked_prediction_df) if st.session_state.ranked_prediction_df is not None else 0

st.markdown('<div class="card-wrap"><div class="card-title">集計状況</div></div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">保存済み履歴件数</div>
        <div class="metric-value">{history_count}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">保存済み条件数</div>
        <div class="metric-value">{condition_count}</div>
    </div>
    """, unsafe_allow_html=True)

c3, c4 = st.columns(2)
with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">予想CSVレース数</div>
        <div class="metric-value">{prediction_race_count}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">予想CSV馬数</div>
        <div class="metric-value">{prediction_horse_count}</div>
    </div>
    """, unsafe_allow_html=True)
