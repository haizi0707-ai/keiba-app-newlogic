import streamlit as st
import pandas as pd
import math

# =========================================================
# 競馬ランクアプリ v12.0 Multiplier Logic
# 全部差し替え用 app.py
#
# 修正内容:
# ・1つのCSVに京都/東京/福島など複数会場が入っていても混ざらない
# ・必ず「日付 + 場所 + レース」の3条件で抽出
# ・レース表記が 7 / 7R のどちらでも動くように正規化
# ・アップロードCSVから日付→場所→レースを選択
# ・v12.0 Multiplier Logic の簡易計算とランキング表示
# =========================================================

st.set_page_config(
    page_title="競馬ランクアプリ v12.0",
    page_icon="🏇",
    layout="centered",
)

# -----------------------------
# 共通ユーティリティ
# -----------------------------
def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_race(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith("R"):
        return s
    try:
        return f"{int(float(s))}R"
    except Exception:
        return s


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        s = str(x).replace(",", "").strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def prepare_df(df):
    df = df.copy()

    required_cols = [
        "日付", "場所", "芝ダ", "距離", "レース", "レース名",
        "馬番", "馬名",
        "前走競馬場", "前走芝ダ", "前走距離数値",
        "前3角位置カテゴリ", "前4角位置カテゴリ", "前走場所",
        "前走直線ロジック点", "前々走直線ロジック点",
        "展開予想評価", "直線相性評価",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error("CSVに必要な列がありません。")
        st.write("不足列:", missing)
        st.stop()

    # 文字列正規化
    for col in [
        "日付", "場所", "芝ダ", "距離", "レース", "レース名",
        "馬番", "馬名",
        "前走競馬場", "前走芝ダ",
        "前3角位置カテゴリ", "前4角位置カテゴリ", "前走場所",
        "展開予想評価", "直線相性評価",
    ]:
        df[col] = df[col].apply(normalize_text)

    df["レース"] = df["レース"].apply(normalize_race)
    df["前走距離数値"] = df["前走距離数値"].apply(lambda x: int(safe_float(x, 0)))
    df["前走直線ロジック点"] = df["前走直線ロジック点"].apply(lambda x: safe_float(x, 0))
    df["前々走直線ロジック点"] = df["前々走直線ロジック点"].apply(lambda x: safe_float(x, 0))

    # race_idを作成
    df["race_id"] = df["日付"] + "_" + df["場所"] + "_" + df["レース"]

    # 馬番ソート用
    df["_馬番_num"] = df["馬番"].apply(lambda x: int(float(x)) if str(x).replace(".", "", 1).isdigit() else 999)

    return df


# -----------------------------
# v12.0 Multiplier Logic
# -----------------------------
def position_multiplier(pos):
    """
    展開位置補正の基礎。
    先行有利になりやすい短距離/小回りにも対応しやすいように、
    極端な差はつけすぎない。
    """
    mapping = {
        "1番手": 1.08,
        "2-3番手": 1.06,
        "4-6番手": 1.03,
        "7-10番手": 0.98,
        "11番手以下": 0.94,
    }
    return mapping.get(str(pos).strip(), 1.00)


def evaluation_multiplier(label):
    mapping = {
        "かなり向く": 1.15,
        "向く": 1.08,
        "普通": 1.00,
        "やや不向き": 0.92,
        "不向き": 0.85,
    }
    return mapping.get(str(label).strip(), 1.00)


def place_straight_multiplier(label):
    """
    直線相性評価による補正。
    """
    return evaluation_multiplier(label)


def rank_label(score):
    if score >= 58:
        return "S"
    if score >= 52:
        return "A"
    if score >= 44:
        return "B"
    if score >= 36:
        return "C"
    return "D"


def rank_sort_value(label):
    order = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
    return order.get(label, 0)


def calculate_scores(race_df):
    df = race_df.copy()

    # 本体点 = 前走直線ロジック点×0.30 + 前々走直線ロジック点×0.20
    df["本体点"] = (
        df["前走直線ロジック点"].apply(safe_float) * 0.30
        + df["前々走直線ロジック点"].apply(safe_float) * 0.20
    )

    # 展開位置補正
    # 前4角をやや重視し、展開予想評価を掛ける
    df["位置補正"] = (
        df["前3角位置カテゴリ"].apply(position_multiplier) * 0.4
        + df["前4角位置カテゴリ"].apply(position_multiplier) * 0.6
    )
    df["展開評価補正"] = df["展開予想評価"].apply(evaluation_multiplier)
    df["展開位置補正"] = df["位置補正"] * df["展開評価補正"]

    # 前走場所直線補正
    df["前走場所直線補正"] = df["直線相性評価"].apply(place_straight_multiplier)

    # 最終点
    df["最終点"] = df["本体点"] * df["展開位置補正"] * df["前走場所直線補正"]

    # 見やすさ用に0〜100へ軽くスケール
    # 本体点が最大50点なので、2倍基準にして100点換算
    df["表示点"] = (df["最終点"] * 2).clip(lower=0, upper=100).round(1)

    df["ランク"] = df["表示点"].apply(rank_label)

    df = df.sort_values(
        by=["表示点", "前走直線ロジック点", "前々走直線ロジック点", "_馬番_num"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    df["順位"] = range(1, len(df) + 1)

    return df


# -----------------------------
# UI表示
# -----------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .main {
            background: #ffffff;
        }
        .race-card {
            background: #071a36;
            color: white;
            border-radius: 28px;
            padding: 26px 22px;
            margin: 12px 0 24px 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.18);
        }
        .race-title {
            font-size: 34px;
            font-weight: 900;
            line-height: 1.1;
            margin-bottom: 8px;
            letter-spacing: 0.5px;
        }
        .race-subtitle {
            color: #c7d3ea;
            font-size: 19px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        .horse-row {
            background: #102443;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 16px 14px;
            margin: 12px 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .horse-num {
            color: #b8c7e8;
            font-weight: 900;
            font-size: 22px;
            width: 42px;
            text-align: center;
            flex-shrink: 0;
        }
        .horse-main {
            flex-grow: 1;
            min-width: 0;
        }
        .horse-name {
            color: #ffffff;
            font-weight: 900;
            font-size: 25px;
            line-height: 1.15;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .horse-score {
            color: #c7d3ea;
            font-size: 13px;
            font-weight: 700;
            margin-top: 5px;
        }
        .rank-area {
            width: 78px;
            flex-shrink: 0;
            text-align: center;
        }
        .rank-label {
            color: #c7d3ea;
            font-size: 13px;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .rank-box {
            border: 4px solid #d8b64a;
            border-radius: 18px;
            height: 58px;
            width: 58px;
            display: inline-flex;
            justify-content: center;
            align-items: center;
            color: #ffffff;
            font-size: 30px;
            font-weight: 900;
        }
        .small-note {
            color: #6b7280;
            font-size: 13px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_cards(result_df, selected_date, selected_place, selected_race):
    if result_df.empty:
        st.warning("該当データがありません。")
        return

    first = result_df.iloc[0]
    race_name = first.get("レース名", "")
    turf_dirt = first.get("芝ダ", "")
    distance = first.get("距離", "")

    title_date = selected_date
    if str(title_date).startswith("2026."):
        title_date = title_date
    title = f"{title_date} {selected_place}{selected_race}"

    rows_html = ""
    for _, row in result_df.iterrows():
        rows_html += f"""
        <div class="horse-row">
            <div class="horse-num">{row.get("馬番", "")}</div>
            <div class="horse-main">
                <div class="horse-name">{row.get("馬名", "")}</div>
                <div class="horse-score">点数 {row.get("表示点", "")} / 展開 {row.get("展開予想評価", "")} / 直線 {row.get("直線相性評価", "")}</div>
            </div>
            <div class="rank-area">
                <div class="rank-label">ランク</div>
                <div class="rank-box">{row.get("ランク", "")}</div>
            </div>
        </div>
        """

    st.markdown(
        f"""
        <div class="race-card">
            <div class="race-title">{title}</div>
            <div class="race-subtitle">{race_name} / {turf_dirt}{distance}</div>
            {rows_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_table(result_df):
    show_cols = [
        "順位", "馬番", "馬名", "表示点", "ランク",
        "本体点", "展開位置補正", "前走場所直線補正",
        "前走競馬場", "前走芝ダ", "前走距離数値",
        "前3角位置カテゴリ", "前4角位置カテゴリ",
        "前走直線ロジック点", "前々走直線ロジック点",
        "展開予想評価", "直線相性評価",
    ]
    existing = [c for c in show_cols if c in result_df.columns]
    st.dataframe(result_df[existing], use_container_width=True, hide_index=True)


# -----------------------------
# メイン
# -----------------------------
inject_css()

st.title("競馬ランクアプリ v12.0")
st.caption("Multiplier Logic / 日付 + 場所 + レースで安全抽出")

uploaded_file = st.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file is None:
    st.info("CSVファイルをアップロードしてください。")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except UnicodeDecodeError:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, encoding="cp932")

df = prepare_df(df)

# 日付選択
dates = sorted(df["日付"].dropna().unique().tolist())
selected_date = st.selectbox("日付", dates)

# 場所選択
date_df = df[df["日付"] == selected_date].copy()
places = sorted(date_df["場所"].dropna().unique().tolist())
selected_place = st.selectbox("場所", places)

# レース選択
place_df = date_df[date_df["場所"] == selected_place].copy()
races = sorted(
    place_df["レース"].dropna().unique().tolist(),
    key=lambda x: int(str(x).replace("R", "")) if str(x).replace("R", "").isdigit() else 999
)
selected_race = st.selectbox("レース", races)

# ここが重要：日付 + 場所 + レース の3条件で抽出
race_df = df[
    (df["日付"] == selected_date)
    & (df["場所"] == selected_place)
    & (df["レース"] == selected_race)
].copy()

if race_df.empty:
    st.error(f"{selected_date} {selected_place} {selected_race} のデータがありません。")
    st.stop()

result_df = calculate_scores(race_df)

st.success(f"{selected_date} {selected_place} {selected_race}：{len(result_df)}頭を表示中")

view_mode = st.radio(
    "表示形式",
    ["カード表示", "表表示"],
    horizontal=True,
)

if view_mode == "カード表示":
    render_cards(result_df, selected_date, selected_place, selected_race)
else:
    render_table(result_df)

# ダウンロード
download_df = result_df.drop(columns=[c for c in ["_馬番_num"] if c in result_df.columns])
csv_bytes = download_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

st.download_button(
    label="このレースの計算結果CSVをダウンロード",
    data=csv_bytes,
    file_name=f"{selected_date}_{selected_place}_{selected_race}_v12_result.csv",
    mime="text/csv",
)

with st.expander("抽出確認"):
    st.write("現在の抽出条件")
    st.code(f"日付={selected_date} / 場所={selected_place} / レース={selected_race}")
    st.write("race_id")
    st.code(f"{selected_date}_{selected_place}_{selected_race}")
    st.write("CSV内の対象行数:", len(race_df))
