# v12.0 Multiplier Logic app update
# 目的:
# 1つのCSVに複数会場の同一Rが入っていても混ざらないように、
# 「日付 + 場所 + レース」の3条件で抽出するための更新用コードです。
#
# 使い方:
# - 既存のStreamlitアプリにこの内容を貼り付ける
# - 既存コード内で「日付 + レース」だけで絞り込んでいる箇所を
#   filter_race_df(...) に置き換える
#
# 例:
# race_df = filter_race_df(df, selected_date, selected_place, selected_race)

import pandas as pd
import streamlit as st


def normalize_race_value(x):
    """
    レース列を '7R' '8R' の形式に統一する。
    CSV側が 7 / 7R / ' 7R ' のどれでも対応。
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith("R"):
        return s
    try:
        return f"{int(float(s))}R"
    except Exception:
        return s


def normalize_place_value(x):
    """
    場所列の余計な空白を除去。
    """
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_date_value(x):
    """
    日付列の余計な空白を除去。
    例: 2026.4.26
    """
    if pd.isna(x):
        return ""
    return str(x).strip()


def prepare_race_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSV読み込み直後に実行する前処理。
    日付・場所・レースを正規化し、race_idを追加する。
    """
    df = df.copy()

    required_cols = ["日付", "場所", "レース"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"CSVに必須列がありません: {', '.join(missing)}")
        st.stop()

    df["日付"] = df["日付"].apply(normalize_date_value)
    df["場所"] = df["場所"].apply(normalize_place_value)
    df["レース"] = df["レース"].apply(normalize_race_value)

    df["race_id"] = (
        df["日付"].astype(str)
        + "_"
        + df["場所"].astype(str)
        + "_"
        + df["レース"].astype(str)
    )

    return df


def filter_race_df(
    df: pd.DataFrame,
    selected_date: str,
    selected_place: str,
    selected_race: str,
) -> pd.DataFrame:
    """
    1レース分を安全に抽出する関数。
    必ず「日付 + 場所 + レース」の3条件で絞り込む。
    """
    selected_date = normalize_date_value(selected_date)
    selected_place = normalize_place_value(selected_place)
    selected_race = normalize_race_value(selected_race)

    race_df = df[
        (df["日付"] == selected_date)
        & (df["場所"] == selected_place)
        & (df["レース"] == selected_race)
    ].copy()

    return race_df


def render_race_selector(df: pd.DataFrame):
    """
    Streamlit用の安全なレース選択UI。
    日付 → 場所 → レース の順に選択する。
    """
    df = prepare_race_dataframe(df)

    dates = sorted(df["日付"].dropna().unique().tolist())
    selected_date = st.selectbox("日付", dates)

    place_df = df[df["日付"] == selected_date]
    places = sorted(place_df["場所"].dropna().unique().tolist())
    selected_place = st.selectbox("場所", places)

    race_list_df = place_df[place_df["場所"] == selected_place]
    races = sorted(
        race_list_df["レース"].dropna().unique().tolist(),
        key=lambda x: int(str(x).replace("R", "")) if str(x).replace("R", "").isdigit() else 999
    )
    selected_race = st.selectbox("レース", races)

    race_df = filter_race_df(
        df=df,
        selected_date=selected_date,
        selected_place=selected_place,
        selected_race=selected_race,
    )

    if race_df.empty:
        st.warning(f"{selected_date} {selected_place} {selected_race} のデータが見つかりません。")
        st.stop()

    st.caption(f"表示中: {selected_date} {selected_place} {selected_race} / {len(race_df)}頭")

    return df, race_df, selected_date, selected_place, selected_race


# ==============================
# 既存アプリへの組み込み例
# ==============================
#
# uploaded_file = st.file_uploader("CSVをアップロード", type=["csv"])
#
# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#
#     df, race_df, selected_date, selected_place, selected_race = render_race_selector(df)
#
#     # ここから下は既存のランキング計算処理へ
#     # 必ず df 全体ではなく race_df を渡す
#     #
#     # result_df = calculate_rank(race_df)
#     # st.dataframe(result_df)
#
# ==============================
# 既存コードで置き換えるポイント
# ==============================
#
# 修正前:
# race_df = df[
#     (df["日付"] == selected_date) &
#     (df["レース"] == selected_race)
# ]
#
# 修正後:
# race_df = filter_race_df(
#     df=df,
#     selected_date=selected_date,
#     selected_place=selected_place,
#     selected_race=selected_race,
# )
#
