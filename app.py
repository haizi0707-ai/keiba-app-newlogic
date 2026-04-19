import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties

st.set_page_config(page_title="競馬 ランクアプリ v6.8 New Logic", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
THRESHOLD_PATH = DATA_DIR / "keiba_rank_thresholds.csv"

DEFAULT_THRESHOLDS = pd.DataFrame([
    {"rank": "S", "min_score": 72.0, "max_score": 999.0},
    {"rank": "A", "min_score": 60.0, "max_score": 72.0},
    {"rank": "B", "min_score": 50.0, "max_score": 60.0},
    {"rank": "C", "min_score": 40.0, "max_score": 50.0},
    {"rank": "D", "min_score": -999.0, "max_score": 40.0},
])

REQUIRED_PRED_COLS = [
    "日付", "開催", "R", "レース名", "馬番", "馬名", "種牡馬",
    "調教師", "騎手", "距離", "馬場状態", "前開催", "前距離", "間隔"
]

if "history_df" not in st.session_state:
    st.session_state.history_df = None
if "summary_data" not in st.session_state:
    st.session_state.summary_data = None
if "ranked_prediction_df" not in st.session_state:
    st.session_state.ranked_prediction_df = None
if "generated_image_bytes" not in st.session_state:
    st.session_state.generated_image_bytes = None
if "preview_race_df" not in st.session_state:
    st.session_state.preview_race_df = None
if "preview_title" not in st.session_state:
    st.session_state.preview_title = "レースランキング"

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #071223 0%, #0a1730 100%);
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
.block-container {
    max-width: 1120px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}

.main-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.25;
    margin-bottom: 0.35rem;
}
.sub-title {
    color: #dce9ff;
    font-size: 1rem;
    line-height: 1.8;
    margin-bottom: 1rem;
}
.info-box {
    background: rgba(166, 198, 255, 0.12);
    border: 1px solid rgba(166, 198, 255, 0.24);
    border-radius: 20px;
    padding: 1.1rem 1.15rem;
    color: #ffffff;
    font-size: 1rem;
    line-height: 1.9;
    margin-bottom: 1rem;
}
.section-card {
    background: linear-gradient(180deg, rgba(10,20,40,0.97) 0%, rgba(8,16,32,0.97) 100%);
    border: 1px solid rgba(122, 154, 214, 0.22);
    border-radius: 24px;
    padding: 1rem 1rem 0.9rem 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 28px rgba(0,0,0,0.16);
}
.section-title {
    color: #ffffff;
    font-size: 1.45rem;
    font-weight: 800;
    margin: 0;
}
.small-note {
    color: #eef4ff;
    font-size: 1rem;
    font-weight: 600;
}
.metric-card {
    background: linear-gradient(180deg, rgba(12,22,42,0.98) 0%, rgba(10,18,36,0.98) 100%);
    border: 1px solid rgba(130, 160, 220, 0.20);
    border-radius: 22px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.metric-label {
    color: #dbe7ff;
    font-size: 1rem;
    margin-bottom: 0.25rem;
}
.metric-value {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.1;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] summary,
label[data-testid="stWidgetLabel"] {
    color: #f8fbff !important;
    font-weight: 600 !important;
}

[data-testid="stFileUploader"] {
    background: #13233d !important;
    border: 1px solid rgba(46, 204, 113, 0.40) !important;
    border-radius: 18px !important;
    padding: 0.55rem !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: #13233d !important;
    border: 1px solid rgba(46, 204, 113, 0.32) !important;
    border-radius: 16px !important;
    color: #ffffff !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] p {
    color: #ffffff !important;
}
[data-testid="stFileUploader"] button,
[data-testid="stBaseButton-secondary"] {
    background: linear-gradient(90deg, #14b76b, #1ed37f) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    border: none !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"] svg {
    fill: #ffffff !important;
}

[data-baseweb="select"] > div {
    background: #13233d !important;
    color: #ffffff !important;
    border: 1px solid rgba(216, 92, 92, 0.45) !important;
    border-radius: 16px !important;
    min-height: 3rem !important;
}
[data-baseweb="select"] * {
    color: #ffffff !important;
}

.stButton > button,
.stDownloadButton > button {
    border-radius: 18px !important;
    font-weight: 800 !important;
    padding: 0.82rem 1rem !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 1rem !important;
}
.green-btn button {
    background: linear-gradient(90deg, #14b76b, #1ed37f) !important;
    color: #ffffff !important;
}
.orange-btn button {
    background: linear-gradient(90deg, #cc8b16, #f0a21a) !important;
    color: #ffffff !important;
}
.red-btn button {
    background: linear-gradient(90deg, #b94e4e, #d85c5c) !important;
    color: #ffffff !important;
}
.dark-btn button {
    background: #1f3151 !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}
.stButton > button:disabled,
.stDownloadButton > button:disabled {
    background: #6b7280 !important;
    color: #f8fbff !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    opacity: 1 !important;
}

[data-testid="stAlert"] {
    border-radius: 16px !important;
}
[data-testid="stAlert"] * {
    color: #ffffff !important;
}
[data-testid="stDataFrame"] * {
    color: #f8fbff !important;
}

.preview-panel {
    background: #091426;
    border: 1px solid rgba(126, 156, 214, 0.18);
    border-radius: 20px;
    padding: 1rem;
}
.preview-title {
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.25rem;
}
.preview-sub {
    color: #cfdcff;
    font-size: 1rem;
    margin-bottom: 1rem;
}
.preview-row {
    display: grid;
    grid-template-columns: 1fr 90px;
    gap: 12px;
    align-items: center;
    padding: 10px 12px;
    border-radius: 16px;
    margin-bottom: 8px;
    background: rgba(255,255,255,0.03);
}
.preview-name {
    color: #ffffff;
    font-size: 1.35rem;
    font-weight: 800;
    line-height: 1.2;
}
.preview-class {
    color: #cdd9f4;
    font-size: 0.95rem;
    margin-top: 4px;
}
.rank-box {
    text-align: center;
    border-radius: 14px;
    padding: 8px 0;
    font-weight: 800;
    font-size: 1.35rem;
    color: #f7fbff;
    background: #1d2b46;
    border: 1px solid rgba(120,160,220,0.22);
}
.rank-S { background: #4c2fa8; }
.rank-A { background: #1f8b58; }
.rank-B { background: #2c6eb8; }
.rank-C { background: #a97115; }
.rank-D { background: #5a6578; }

.cond-table {
    width: 100%;
    border-collapse: collapse;
}
.cond-table th, .cond-table td {
    text-align: left;
    padding: 12px 10px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    color: #f8fbff;
    vertical-align: top;
}
.cond-table th {
    color: #ffffff;
    font-size: 1rem;
    font-weight: 700;
}
.cond-cond {
    font-size: 0.92rem;
    color: #d7e5ff;
    line-height: 1.45;
    word-break: break-word;
}

@media (max-width: 720px) {
    .preview-row {
        grid-template-columns: 1fr 72px;
    }
    .preview-name {
        font-size: 1.1rem;
    }
}
</style>
""", unsafe_allow_html=True)


def build_history_summary(history_df: pd.DataFrame):
    df = history_df.copy()

    if "距離帯" not in df.columns:
        if "距離数値" not in df.columns:
            df = add_surface_distance_columns(df)
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
        if "前距離" in df.columns and "距離" in df.columns:
            df["距離変化"] = df.apply(lambda r: get_distance_change(r.get("距離"), r.get("前距離")), axis=1)
        else:
            df["距離変化"] = "不明"

    if "開催変化" not in df.columns:
        if "前開催" in df.columns and "開催" in df.columns:
            df["開催変化"] = df.apply(lambda r: get_track_change(r.get("開催"), r.get("前開催")), axis=1)
        else:
            df["開催変化"] = "不明"

    if "複勝フラグ" in df.columns:
        df["placed_flag"] = pd.to_numeric(df["複勝フラグ"], errors="coerce").fillna(0).astype(int)
    elif "着順" in df.columns:
        df["placed_flag"] = calc_place_flag(df["着順"])
    else:
        raise ValueError("過去レースCSVに 複勝フラグ または 着順 列が必要です。")

    def make_group(keys):
        missing = [k for k in keys if k not in df.columns]
        if missing:
            return {}
        grouped = (
            df.groupby(keys, dropna=False)
            .agg(count=("placed_flag", "size"), placed=("placed_flag", "sum"))
            .reset_index()
        )
        out = {}
        for row in grouped.itertuples(index=False):
            key = f"{row[0]}|||{row[1]}"
            count = int(row[2])
            placed = int(row[3])
            out[key] = {
                "count": count,
                "placed": placed,
                "place_rate": placed / count if count else 0.0
            }
        return out

    return {
        "source_rows": int(len(df)),
        "sire_track": make_group(["種牡馬", "開催"]),
        "sire_dist": make_group(["種牡馬", "距離帯"]),
        "sire_going": make_group(["種牡馬", "馬場区分"]),
        "trainer_track": make_group(["調教師", "開催"]),
        "trainer_dist": make_group(["調教師", "距離帯"]),
        "trainer_interval": make_group(["調教師", "間隔カテゴリ"]),
        "trainer_distchg": make_group(["調教師", "距離変化"]),
        "trainer_trackchg": make_group(["調教師", "開催変化"]),
    }


def history_backup_payload():
    if st.session_state.history_df is None or st.session_state.summary_data is None:
        return None
    return {
        "history_rows": st.session_state.history_df.to_dict(orient="records"),
        "summary_data": st.session_state.summary_data,
    }


def restore_history_backup(uploaded_json):
    uploaded_json.seek(0)
    payload = json.load(uploaded_json)
    st.session_state.history_df = pd.DataFrame(payload.get("history_rows", []))
    st.session_state.summary_data = payload.get("summary_data", None)


st.markdown('<div class="main-title">競馬 ランクアプリ<br>v6.8 New Logic</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">見た目は6.8v系の想定で、内部ロジックを新2軸版に差し替えた再構成版です。</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
判定条件は <b>血統適性 × 厩舎ローテ適性</b> です。<br><br>
縦軸は「血統 × 競馬場・距離・馬場」、横軸は「厩舎 × 競馬場・距離・ローテ」で評価します。<br>
Sランクはかなり厳しめで、原則1頭、条件を満たす時だけ最大2頭です。
</div>
""", unsafe_allow_html=True)

thresholds = load_thresholds()
with st.expander("ランク基準を見る"):
    st.dataframe(thresholds, use_container_width=True)

st.markdown('<div class="section-card"><div class="section-title">過去レースCSV（収集用）</div></div>', unsafe_allow_html=True)
history_file = st.file_uploader("過去レースCSV", type=["csv"], key="history_uploader", label_visibility="collapsed")
history_backup_file = st.file_uploader("履歴バックアップJSON復元", type=["json"], key="history_backup_uploader", label_visibility="collapsed")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="green-btn">', unsafe_allow_html=True)
    import_history = st.button("過去レースCSVを取り込む", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="dark-btn">', unsafe_allow_html=True)
    backup_history = st.button("履歴バックアップ保存", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="red-btn">', unsafe_allow_html=True)
    clear_history = st.button("過去データ削除", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if import_history:
    if history_file is None:
        st.error("過去レースCSVを選択してください。")
    else:
        try:
            raw_history = read_uploaded_csv(history_file)
            raw_history = normalize_columns(raw_history)
            raw_history = ensure_race_key_columns(raw_history)
            raw_history = add_surface_distance_columns(raw_history)
            st.session_state.history_df = raw_history
            st.session_state.summary_data = build_history_summary(raw_history)
            st.success(f"過去データを取り込みました。件数: {len(raw_history):,}")
        except Exception as e:
            st.error(f"過去レースCSVの読み込みでエラーが出ました: {e}")

if history_backup_file is not None:
    try:
        restore_history_backup(history_backup_file)
        st.success("履歴バックアップを復元しました。")
    except Exception as e:
        st.error(f"履歴バックアップの復元でエラーが出ました: {e}")

if backup_history:
    payload = history_backup_payload()
    if payload is None:
        st.error("保存できる過去データがありません。")
    else:
        backup_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "履歴バックアップJSONをダウンロード",
            data=backup_bytes,
            file_name="keiba_history_backup.json",
            mime="application/json",
            use_container_width=True
        )

if clear_history:
    st.session_state.history_df = None
    st.session_state.summary_data = None
    st.session_state.ranked_prediction_df = None
    st.session_state.preview_race_df = None
    st.session_state.generated_image_bytes = None
    st.success("過去データを削除しました。")

if st.session_state.history_df is not None:
    st.markdown(f'<div class="small-note">取り込み済み件数: {len(st.session_state.history_df):,}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="small-note">まだ取り込んでいません。</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">予想レースCSV（画像化用）</div></div>', unsafe_allow_html=True)
pred_file = st.file_uploader("予想レースCSV", type=["csv"], key="pred_uploader", label_visibility="collapsed")

race_options = []
race_labels = []
selected_race_label = None

if pred_file is not None:
    try:
        tmp_pred = read_uploaded_csv(pred_file)
        tmp_pred = normalize_columns(tmp_pred)
        tmp_pred = ensure_race_key_columns(tmp_pred)
        race_options = race_options_from_df(tmp_pred)
        race_labels = [x[0] for x in race_options]
        pred_file.seek(0)
    except Exception:
        race_options = []
        race_labels = []
elif st.session_state.ranked_prediction_df is not None:
    race_options = race_options_from_df(st.session_state.ranked_prediction_df)
    race_labels = [x[0] for x in race_options]

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
    if st.session_state.summary_data is None:
        st.error("先に過去レースCSVを取り込んでください。")
    elif pred_file is None:
        st.error("予想CSVを選択してください。")
    else:
        try:
            pred_file.seek(0)
            pred_raw = pd.read_csv(pred_file)
            ranked = prepare_prediction_df(pred_raw, st.session_state.summary_data, thresholds)
            st.session_state.ranked_prediction_df = ranked

            current_race_map = dict(race_options_from_df(ranked))
            preview_df = ranked.copy()
            preview_title = "レースランキング"

            if selected_race_label and selected_race_label in current_race_map:
                preview_df = filter_race_df(ranked, current_race_map[selected_race_label])
                preview_title = selected_race_label
            else:
                race_list = race_options_from_df(ranked)
                if race_list:
                    first_label, first_dict = race_list[0]
                    preview_df = filter_race_df(ranked, first_dict)
                    preview_title = first_label

            st.session_state.preview_race_df = preview_df
            st.session_state.preview_title = preview_title
            st.session_state.generated_image_bytes = build_race_image_bytes(preview_df, preview_title)

            st.success(f"予想CSV読込完了: {len(ranked):,}頭 / {unique_race_count(ranked)}レース")
        except Exception as e:
            st.error(f"予想CSVの読み込みでエラーが出ました: {e}")

if st.session_state.ranked_prediction_df is not None:
    ranked_df = st.session_state.ranked_prediction_df.copy()
    current_race_map = dict(race_options_from_df(ranked_df))

    show_df = ranked_df.copy()
    current_title = st.session_state.preview_title

    if selected_race_label and selected_race_label in current_race_map:
        show_df = filter_race_df(ranked_df, current_race_map[selected_race_label])
        current_title = selected_race_label

    st.session_state.preview_race_df = show_df.copy()
    st.session_state.preview_title = current_title

    st.markdown(
        f'<div class="small-note">予想CSV読込完了: {len(ranked_df):,}頭 / {unique_race_count(ranked_df)}レース</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown('<div class="small-note">まだ読み込んでいません。</div>', unsafe_allow_html=True)

if make_image:
    if st.session_state.preview_race_df is None or st.session_state.preview_race_df.empty:
        st.error("先に予想CSVを読み込んでください。")
    else:
        st.session_state.generated_image_bytes = build_race_image_bytes(
            st.session_state.preview_race_df,
            st.session_state.preview_title
        )
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

st.markdown('<div class="section-card"><div class="section-title">画像プレビュー</div></div>', unsafe_allow_html=True)
if st.session_state.preview_race_df is not None and not st.session_state.preview_race_df.empty:
    render_preview_html(st.session_state.preview_race_df, st.session_state.preview_title)
else:
    st.markdown('<div class="preview-panel"><div class="preview-sub">予想CSVを読み込んでください</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">画像外の条件集計</div></div>', unsafe_allow_html=True)
if st.session_state.preview_race_df is not None and not st.session_state.preview_race_df.empty:
    render_condition_table(st.session_state.preview_race_df)
else:
    st.info("予想CSVを読み込むと表示されます。")

history_count = len(st.session_state.history_df) if st.session_state.history_df is not None else 0
condition_count = saved_condition_count(st.session_state.summary_data)
prediction_race_count = unique_race_count(st.session_state.ranked_prediction_df)
prediction_horse_count = len(st.session_state.ranked_prediction_df) if st.session_state.ranked_prediction_df is not None else 0

st.markdown('<div class="section-card"><div class="section-title">集計状況</div></div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">保存済み履歴件数</div><div class="metric-value">{history_count}</div></div>',
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">保存済み条件数</div><div class="metric-value">{condition_count}</div></div>',
        unsafe_allow_html=True
    )

c3, c4 = st.columns(2)
with c3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">予想CSVレース数</div><div class="metric-value">{prediction_race_count}</div></div>',
        unsafe_allow_html=True
    )
with c4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">予想CSV馬数</div><div class="metric-value">{prediction_horse_count}</div></div>',
        unsafe_allow_html=True
    )
