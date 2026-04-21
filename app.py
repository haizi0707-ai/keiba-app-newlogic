import os
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="競馬ランクアプリ v8.4 Rank View", layout="centered")

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
DEFAULT_FILES = {
    "bloodline": os.path.join(BASE_DIR, "bloodline_stats_7to12.csv"),
    "trainer": os.path.join(BASE_DIR, "trainer_stats_7to12.csv"),
    "style": os.path.join(BASE_DIR, "style_stats_7to12.csv"),
    "prevtrack": os.path.join(BASE_DIR, "prevtrack_stats_7to12.csv"),
    "winstyle": os.path.join(BASE_DIR, "winstyle_stats_7to12.csv"),
}

TABLE_SPECS = {
    "bloodline": {"label": "血統", "key_col": "血統", "jp_col": "血統", "score_col": "血統点", "count_col": "血統母数"},
    "trainer": {"label": "調教師", "key_col": "調教師", "jp_col": "調教師", "score_col": "調教師点", "count_col": "調教師母数"},
    "style": {"label": "脚質", "key_col": "脚質", "jp_col": "脚質", "score_col": "脚質点", "count_col": "脚質母数"},
    "prevtrack": {"label": "前走場所", "key_col": "前走場所", "jp_col": "前走場所", "score_col": "前走場所点", "count_col": "前走場所母数"},
    "winstyle": {"label": "勝ち方", "key_col": "勝ち方", "jp_col": "勝ち方", "score_col": "勝ち方点", "count_col": "勝ち方母数"},
}

RACE_COLUMN_CANDIDATES = {
    "date": ["date", "日付", "開催日", "年月日"],
    "場所": ["場所", "track", "競馬場", "開催", "場名"],
    "raceNo": ["raceNo", "race_number", "raceNo.", "R", "レース番号", "race_no"],
    "raceName": ["raceName", "race_name", "レース名", "レース"],
    "horseNo": ["horseNo", "horse_number", "馬番"],
    "horseName": ["horseName", "horse_name", "馬名"],
    "jockey": ["jockey", "騎手"],
    "trainer": ["trainer", "調教師"],
    "sire": ["sire", "種牡馬", "血統"],
    "distance": ["distance", "距離"],
    "going": ["going", "馬場状態", "馬場"],
    "style": ["style", "脚質"],
    "prevTrack": ["prevTrack", "前走場所", "前開催", "前走競馬場"],
    "winStyle": ["winStyle", "勝ち方", "前走勝ち方"],
    "comment": ["comment", "短評", "備考"],
}

STYLE_MAP = {
    "逃": "逃げ", "逃げ": "逃げ",
    "先": "先行", "先行": "先行",
    "差": "差し", "差し": "差し",
    "追": "追込", "追込": "追込", "追い込み": "追込",
}

WINSTYLE_MAP = {
    "逃げ切り": "逃げ切り",
    "先行押し切り": "先行押し切り",
    "好位差し": "好位差し",
    "差し": "差し",
    "追い込み": "追い込み",
    "追込": "追い込み",
    "まくり": "まくり",
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
    replace_map = {
        "東京競馬場": "東京", "中山競馬場": "中山", "中京競馬場": "中京", "阪神競馬場": "阪神",
        "京都競馬場": "京都", "新潟競馬場": "新潟", "福島競馬場": "福島", "小倉競馬場": "小倉",
        "札幌競馬場": "札幌", "函館競馬場": "函館",
    }
    return replace_map.get(s, s)


def norm_surface(v):
    s = norm_text(v)
    if s.startswith("芝"):
        return "芝"
    if s.startswith("ダ") or s.startswith("ダート"):
        return "ダ"
    if s.startswith("障"):
        return "障"
    if s in ["芝", "ダ", "障"]:
        return s
    if s == "ダート":
        return "ダ"
    return s


def parse_distance_field(v):
    s = norm_text(v)
    if not s:
        return "", np.nan
    surface = norm_surface(s)
    num = pd.to_numeric("".join(ch for ch in s if ch.isdigit()), errors="coerce")
    return surface, num


def norm_style(v):
    s = norm_text(v)
    return STYLE_MAP.get(s, s)


def norm_winstyle(v):
    s = norm_text(v)
    return WINSTYLE_MAP.get(s, s)


def rename_first_match(df, candidates_map):
    df = df.copy()
    rename_map = {}
    for target, candidates in candidates_map.items():
        for c in candidates:
            if c in df.columns:
                rename_map[c] = target
                break
    return df.rename(columns=rename_map)


def prepare_race_df(df):
    df = rename_first_match(df, RACE_COLUMN_CANDIDATES)

    # v8.1入力CSVは「芝ダ」「距離」を別列で持つ前提。
    # 旧形式の distance=芝1200 / ダ1800 にも対応する。
    for col in ["date", "場所", "raceNo", "raceName", "horseNo", "horseName", "jockey", "trainer", "sire",
                "distance", "芝ダ", "距離", "going", "style", "prevTrack", "winStyle", "comment"]:
        if col not in df.columns:
            df[col] = ""

    df["場所"] = df["場所"].apply(norm_track)
    df["trainer"] = df["trainer"].apply(norm_text)
    df["sire"] = df["sire"].apply(norm_text)
    df["horseName"] = df["horseName"].apply(norm_text)
    df["prevTrack"] = df["prevTrack"].apply(norm_track)
    df["style"] = df["style"].apply(norm_style)
    df["winStyle"] = df["winStyle"].apply(norm_winstyle)
    df["going"] = df["going"].apply(norm_text)

    # まずは別列の芝ダ / 距離を優先
    df["芝ダ"] = df["芝ダ"].apply(norm_surface)
    df["距離"] = pd.to_numeric(df["距離"], errors="coerce")

    # 別列が空欄のときだけ旧形式distance列から補完
    parsed = df["distance"].apply(parse_distance_field)
    parsed_surface = parsed.apply(lambda x: x[0])
    parsed_dist = parsed.apply(lambda x: x[1])

    df["芝ダ"] = np.where(
        df["芝ダ"].astype(str).str.strip() != "",
        df["芝ダ"],
        parsed_surface
    )
    df["距離"] = df["距離"].where(df["距離"].notna(), parsed_dist)

    return df


def prepare_stat_table(df, kind):
    spec = TABLE_SPECS[kind]
    df = df.copy()
    df.columns = [norm_text(c) for c in df.columns]
    required = ["場所", "芝ダ", "距離", spec["jp_col"], "母数", "複勝数"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{spec['label']}テーブルに必須列が不足: {', '.join(missing)}")

    df = df[required + [c for c in ["勝利数", "単勝率", "複勝率"] if c in df.columns]].copy()
    df["場所"] = df["場所"].apply(norm_track)
    df["芝ダ"] = df["芝ダ"].apply(norm_surface)
    df[spec["jp_col"]] = df[spec["jp_col"]].apply(norm_text)
    if kind == "style":
        df[spec["jp_col"]] = df[spec["jp_col"]].apply(norm_style)
    if kind == "winstyle":
        df[spec["jp_col"]] = df[spec["jp_col"]].apply(norm_winstyle)
    if kind == "prevtrack":
        df[spec["jp_col"]] = df[spec["jp_col"]].apply(norm_track)
    df["距離"] = pd.to_numeric(df["距離"], errors="coerce")
    df["母数"] = pd.to_numeric(df["母数"], errors="coerce").fillna(0)
    df["複勝数"] = pd.to_numeric(df["複勝数"], errors="coerce").fillna(0)
    df["補正複勝率"] = (df["複勝数"] + 1) / (df["母数"] + 3)

    fallback = df.groupby(["場所", "芝ダ", "距離"], dropna=False).agg(
        母数合計=("母数", "sum"),
        複勝数合計=("複勝数", "sum")
    ).reset_index()
    fallback["全体平均複勝率"] = np.where(
        fallback["母数合計"] > 0,
        fallback["複勝数合計"] / fallback["母数合計"],
        np.nan,
    )
    return df, fallback


def lookup_component(row, table_df, fallback_df, kind):
    spec = TABLE_SPECS[kind]
    value_col_race = {
        "bloodline": "sire",
        "trainer": "trainer",
        "style": "style",
        "prevtrack": "prevTrack",
        "winstyle": "winStyle",
    }[kind]
    jp_col = spec["jp_col"]
    target_value = norm_text(row.get(value_col_race, ""))
    if kind == "style":
        target_value = norm_style(target_value)
    elif kind == "winstyle":
        target_value = norm_winstyle(target_value)
    elif kind == "prevtrack":
        target_value = norm_track(target_value)

    cond = (
        (table_df["場所"] == norm_track(row.get("場所", ""))) &
        (table_df["芝ダ"] == norm_surface(row.get("芝ダ", ""))) &
        (table_df["距離"] == pd.to_numeric(row.get("距離", np.nan), errors="coerce")) &
        (table_df[jp_col] == target_value)
    )
    hit = table_df.loc[cond]
    if not hit.empty:
        best = hit.sort_values(["母数", "複勝数"], ascending=False).iloc[0]
        return {
            "score": round(float(best["補正複勝率"]) * 100, 2),
            "count": int(best["母数"]),
            "rate": round(float(best["補正複勝率"]) * 100, 2),
            "matched": target_value,
            "source": "完全一致",
        }

    fb_cond = (
        (fallback_df["場所"] == norm_track(row.get("場所", ""))) &
        (fallback_df["芝ダ"] == norm_surface(row.get("芝ダ", ""))) &
        (fallback_df["距離"] == pd.to_numeric(row.get("距離", np.nan), errors="coerce"))
    )
    fb = fallback_df.loc[fb_cond]
    if not fb.empty and pd.notna(fb.iloc[0]["全体平均複勝率"]):
        item = fb.iloc[0]
        return {
            "score": round(float(item["全体平均複勝率"]) * 100, 2),
            "count": int(item["母数合計"]),
            "rate": round(float(item["全体平均複勝率"]) * 100, 2),
            "matched": target_value,
            "source": "条件平均",
        }

    return {"score": np.nan, "count": np.nan, "rate": np.nan, "matched": target_value, "source": "該当なし"}


def score_race_df(race_df, prepared_tables):
    out = race_df.copy()
    weights = {
        "bloodline": 8,
        "trainer": 12,
        "style": 35,
        "prevtrack": 20,
        "winstyle": 25,
    }
    weighted_parts = []
    for kind, payload in prepared_tables.items():
        table_df, fallback_df = payload
        spec = TABLE_SPECS[kind]
        result = out.apply(lambda row: lookup_component(row, table_df, fallback_df, kind), axis=1)
        out[spec["score_col"]] = result.apply(lambda x: x["score"])
        out[spec["count_col"]] = result.apply(lambda x: x["count"])
        out[f"{spec['label']}参照"] = result.apply(lambda x: x["source"])
        weighted_parts.append(out[spec["score_col"]] * (weights[kind] / 100.0))

    out["総合点"] = sum(weighted_parts).round(2)
    out["順位"] = out.groupby(["場所", "raceNo"], dropna=False)["総合点"].rank(method="min", ascending=False)
    out["順位"] = out["順位"].fillna(999).astype(int)
    out = out.sort_values(["場所", "raceNo", "順位", "horseNo"], ascending=[True, True, True, True])
    return out


def classify_score(score):
    if pd.isna(score):
        return ""
    if score >= 36:
        return "S"
    if score >= 31:
        return "A"
    if score >= 26:
        return "B"
    if score >= 21:
        return "C"
    return "D"


def render_rank_cards(date_val, race_val, race_name_val, dist_text, field_size_val, card_df):
    title_parts = []
    if str(date_val).strip():
        title_parts.append(str(date_val).strip())
    if str(race_val).strip():
        title_parts.append(str(race_val).strip())
    title = " ".join(title_parts).strip()

    subtitle_parts = []
    if str(race_name_val).strip():
        subtitle_parts.append(str(race_name_val).strip())
    if str(dist_text).strip():
        subtitle_parts.append(str(dist_text).strip())
    if str(field_size_val).strip():
        subtitle_parts.append(f"{field_size_val}頭")
    subtitle = " / ".join(subtitle_parts)

    css = """
<style>
.kv-wrap{background:linear-gradient(180deg,#071225 0%,#0b1730 100%);border-radius:22px;padding:18px 14px;color:#f5f7fb;box-shadow:0 10px 30px rgba(0,0,0,.22);margin:10px 0 18px 0}
.kv-title{font-size:22px;font-weight:800;line-height:1.2;margin:0 0 6px 0;letter-spacing:.2px}
.kv-subtitle{font-size:13px;color:#c8d2e8;margin:0 0 14px 0}
.horse-card{display:flex;align-items:center;justify-content:space-between;gap:10px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.04);border-radius:16px;padding:11px 12px;margin:8px 0}
.horse-left{min-width:0;flex:1}
.horse-top{display:flex;align-items:center;gap:8px;min-width:0}
.horse-no{font-size:12px;color:#9fb0d3;font-weight:700;flex:0 0 auto}
.horse-name{font-size:17px;font-weight:800;color:#ffffff;line-height:1.2;word-break:break-all}
.rank-pill{min-width:56px;text-align:center;font-size:22px;font-weight:900;border-radius:14px;padding:9px 18px;flex:0 0 auto;border:2px solid #334a76;color:#edf3ff;background:rgba(93,122,183,.12)}
.rank-S{border-color:#e7c65b;color:#fff2ba;background:rgba(231,198,91,.15);box-shadow:inset 0 0 0 1px rgba(231,198,91,.25)}
.rank-A{border-color:#c9d6ef;color:#ffffff;background:rgba(201,214,239,.10)}
.rank-B{border-color:#d9b456;color:#ffe6a3;background:rgba(217,180,86,.12)}
.rank-C{border-color:#3e547e;color:#d7e3ff;background:rgba(93,122,183,.08)}
.rank-D{border-color:#30405f;color:#b8c6e3;background:rgba(70,89,127,.06)}
</style>
"""
    html = css + f'<div class="kv-wrap"><div class="kv-title">{title}</div><div class="kv-subtitle">{subtitle}</div>'
    for _, row in card_df.iterrows():
        horse_no = str(row.get("馬番", "")).strip()
        horse_name = str(row.get("馬名", "")).strip()
        rank = str(row.get("ランク", "")).strip()
        rank_cls = f"rank-{rank}" if rank in {"S", "A", "B", "C", "D"} else "rank-D"
        html += (
            f'<div class="horse-card">'
            f'<div class="horse-left"><div class="horse-top">'
            f'<div class="horse-no">{horse_no}</div>'
            f'<div class="horse-name">{horse_name}</div>'
            f'</div></div>'
            f'<div class="rank-pill {rank_cls}">{rank}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


st.title("競馬ランクアプリ v8.5 Card View")
st.write("内部で自動採点し、スマホで見やすいカード形式で表示します。")

with st.sidebar:
    st.subheader("入力ファイル")
    race_file = st.file_uploader("出走馬CSV", type=["csv"], key="race")
    uploaded_stats = {}
    for kind, spec in TABLE_SPECS.items():
        uploaded_stats[kind] = st.file_uploader(f"{spec['label']}テーブルCSV（任意）", type=["csv"], key=f"stat_{kind}")
    st.caption("集計テーブルを未指定にした場合は、同じフォルダの既定CSVを使います。")

if race_file is None:
    st.info("まず出走馬CSVをアップロードしてください。")
    st.stop()

try:
    race_df_raw = read_csv_any(race_file)
    race_df = prepare_race_df(race_df_raw)

    prepared_tables = {}
    for kind in TABLE_SPECS:
        src = uploaded_stats[kind] if uploaded_stats[kind] is not None else DEFAULT_FILES[kind]
        if isinstance(src, str) and not os.path.exists(src):
            raise FileNotFoundError(f"既定ファイルが見つかりません: {src}")
        stat_df = read_csv_any(src)
        prepared_tables[kind] = prepare_stat_table(stat_df, kind)

    result_df = score_race_df(race_df, prepared_tables)
    result_df["信頼度"] = result_df["総合点"].apply(classify_score)

    # 表示と出力は最小限に整理
    if "date" not in result_df.columns:
        result_df["date"] = ""
    result_df["raceNo"] = result_df["raceNo"].fillna("").astype(str).str.strip()
    result_df["horseNo"] = result_df["horseNo"].fillna("").astype(str).str.strip()
    result_df["レース"] = result_df["場所"].astype(str) + result_df["raceNo"].astype(str) + "R"
    simple_cols = ["date", "レース", "raceName", "distance", "going", "fieldSize", "horseNo", "horseName", "信頼度"]
    for c in simple_cols:
        if c not in result_df.columns:
            result_df[c] = ""
    result_df["date"] = result_df["date"].fillna("").astype(str)
    result_df["raceName"] = result_df["raceName"].fillna("").astype(str)
    result_df["distance"] = result_df["distance"].fillna("").astype(str)
    result_df["going"] = result_df["going"].fillna("").astype(str)
    result_df["fieldSize"] = result_df["fieldSize"].fillna("").astype(str)
    result_df["horseNo"] = result_df["horseNo"].fillna("").astype(str)
    result_df["horseName"] = result_df["horseName"].fillna("").astype(str)
    result_df["信頼度"] = result_df["信頼度"].fillna("").astype(str)

    export_df = result_df[simple_cols].copy()
    export_df = export_df.rename(columns={
        "date": "日付",
        "raceName": "レース名",
        "distance": "距離",
        "going": "馬場",
        "fieldSize": "頭数",
        "horseNo": "馬番",
        "horseName": "馬名",
        "信頼度": "ランク",
    })

    st.success("自動採点が完了しました。")
    st.caption("現在の比重：脚質35 / 勝ち方25 / 前走場所20 / 調教師12 / 血統8")

    tab1, tab2 = st.tabs(["予想結果", "元データ確認"])
    with tab1:
        grouped = export_df.groupby(["日付", "レース", "レース名", "距離", "馬場", "頭数"], dropna=False, sort=False)
        for (date_val, race_val, race_name_val, dist_val, going_val, field_size_val), g in grouped:
            dist_text = ""
            if str(dist_val).strip() and str(going_val).strip():
                dist_text = f"{going_val}{dist_val}"
            elif str(dist_val).strip():
                dist_text = str(dist_val).strip()
            show_df = g[["馬番", "馬名", "ランク"]].reset_index(drop=True)
            st.markdown(render_rank_cards(date_val, race_val, race_name_val, dist_text, field_size_val, show_df), unsafe_allow_html=True)
    with tab2:
        st.write("出走馬CSV")
        if "date" not in race_df_raw.columns:
            race_df_raw["date"] = ""
        st.dataframe(race_df_raw, use_container_width=True, hide_index=True)

    csv_data = export_df[["日付", "レース", "レース名", "馬番", "馬名", "ランク"]].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="予想結果CSVをダウンロード",
        data=csv_data,
        file_name="rank_card_predictions.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"エラーが発生しました: {e}")
