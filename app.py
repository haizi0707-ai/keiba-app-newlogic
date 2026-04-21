
import os
import re
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="競馬ランクアプリ v9.0 4Factor", layout="centered")

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
DEFAULT_FILES = {
    "style": os.path.join(BASE_DIR, "style_stats_4factor.csv"),
    "prevtrack": os.path.join(BASE_DIR, "prevtrack_stats_4factor.csv"),
    "sire": os.path.join(BASE_DIR, "sire_stats_4factor.csv"),
    "damsire": os.path.join(BASE_DIR, "damsire_stats_4factor.csv"),
}

WEIGHTS = {
    "style": 37.5,
    "prevtrack": 22.5,
    "sire": 15.0,
    "damsire": 25.0,
}

TABLE_SPECS = {
    "style": {"label": "脚質", "race_col": "脚質", "stat_col": "脚質"},
    "prevtrack": {"label": "前走場所", "race_col": "前走場所", "stat_col": "前走場所"},
    "sire": {"label": "種牡馬", "race_col": "種牡馬", "stat_col": "種牡馬"},
    "damsire": {"label": "母父馬", "race_col": "母父馬", "stat_col": "母父馬"},
}

STYLE_MAP = {
    "逃": "逃げ", "逃げ": "逃げ",
    "先": "先行", "先行": "先行",
    "差": "差し", "差し": "差し",
    "追": "追込", "追込": "追込", "追い込み": "追込",
}

RACE_COLUMN_CANDIDATES = {
    "date": ["date", "日付", "開催日", "年月日", "日付S"],
    "場所": ["場所", "track", "競馬場", "開催", "場名"],
    "raceNo": ["raceNo", "race_number", "raceNo.", "R", "レース番号", "race_no", "レースNo", "R番号"],
    "raceName": ["raceName", "race_name", "レース名"],
    "horseNo": ["horseNo", "horse_number", "馬番"],
    "horseName": ["horseName", "horse_name", "馬名"],
    "distance": ["distance", "距離"],
    "surface": ["芝ダ", "芝・ダ", "surface"],
    "going": ["going", "馬場状態", "馬場"],
    "style": ["style", "脚質", "脚質タグ"],
    "prevTrack": ["prevTrack", "前走場所", "前走場所タグ", "前開催", "前走競馬場"],
    "sire": ["sire", "種牡馬", "血統"],
    "damsire": ["damSire", "母父馬"],
    "raceLabel": ["レース", "race"],
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

    for col in ["date","場所","raceNo","raceName","horseNo","horseName","distance","surface","going","style","prevTrack","sire","damsire","raceLabel"]:
        if col not in df.columns:
            df[col] = ""

    if "raceLabel" in df.columns:
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
    df["going"] = df["going"].apply(norm_text)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["raceNo"] = df["raceNo"].astype(str).str.replace(".0", "", regex=False).str.strip()
    df["horseNo"] = df["horseNo"].astype(str).str.replace(".0", "", regex=False).str.strip()

    df["距離表示"] = np.where(
        df["surface"].astype(str).str.strip() != "",
        df["surface"].astype(str) + df["distance"].fillna("").astype(str).str.replace(".0", "", regex=False),
        df["distance"].fillna("").astype(str).str.replace(".0", "", regex=False),
    )
    return df

def prepare_stat_table(df, kind):
    spec = TABLE_SPECS[kind]
    cols = ["場所", "芝ダ", "距離", spec["stat_col"], "母数", "勝利数", "複勝数", "単勝率", "複勝率"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{spec['label']}CSVに必須列不足: {', '.join(missing)}")

    out = df[cols].copy()
    out["場所"] = out["場所"].apply(norm_track)
    out["芝ダ"] = out["芝ダ"].apply(norm_surface)
    out["距離"] = pd.to_numeric(out["距離"], errors="coerce")
    out[spec["stat_col"]] = out[spec["stat_col"]].apply(norm_text)
    if kind == "style":
        out[spec["stat_col"]] = out[spec["stat_col"]].apply(norm_style)
    if kind == "prevtrack":
        out[spec["stat_col"]] = out[spec["stat_col"]].apply(norm_track)
    out["母数"] = pd.to_numeric(out["母数"], errors="coerce").fillna(0)
    out["勝利数"] = pd.to_numeric(out["勝利数"], errors="coerce").fillna(0)
    out["複勝数"] = pd.to_numeric(out["複勝数"], errors="coerce").fillna(0)
    out["補正複勝率"] = (out["複勝数"] + 1) / (out["母数"] + 3)

    fallback = out.groupby(["場所", "芝ダ", "距離"], dropna=False).agg(
        母数合計=("母数", "sum"),
        複勝数合計=("複勝数", "sum")
    ).reset_index()
    fallback["全体平均複勝率"] = np.where(
        fallback["母数合計"] > 0,
        fallback["複勝数合計"] / fallback["母数合計"],
        np.nan,
    )
    return out, fallback

def load_default_or_upload(kind, uploader):
    if uploader is not None:
        return read_csv_any(uploader)
    path = DEFAULT_FILES[kind]
    if os.path.exists(path):
        return read_csv_any(path)
    return None

def lookup_component(row, table_df, fallback_df, kind):
    spec = TABLE_SPECS[kind]
    race_col = spec["race_col"]
    stat_col = spec["stat_col"]
    value = row.get(race_col, "")
    value = norm_text(value)
    if kind == "style":
        value = norm_style(value)
    if kind == "prevtrack":
        value = norm_track(value)

    cond = (
        (table_df["場所"] == norm_track(row.get("場所", ""))) &
        (table_df["芝ダ"] == norm_surface(row.get("surface", ""))) &
        (table_df["距離"] == pd.to_numeric(row.get("distance", np.nan), errors="coerce")) &
        (table_df[stat_col] == value)
    )
    hit = table_df.loc[cond]
    if not hit.empty:
        best = hit.sort_values(["母数", "複勝数"], ascending=False).iloc[0]
        return {"score": float(best["補正複勝率"]) * 100, "source": "完全一致"}

    fb_cond = (
        (fallback_df["場所"] == norm_track(row.get("場所", ""))) &
        (fallback_df["芝ダ"] == norm_surface(row.get("surface", ""))) &
        (fallback_df["距離"] == pd.to_numeric(row.get("distance", np.nan), errors="coerce"))
    )
    fb = fallback_df.loc[fb_cond]
    if not fb.empty and pd.notna(fb.iloc[0]["全体平均複勝率"]):
        return {"score": float(fb.iloc[0]["全体平均複勝率"]) * 100, "source": "条件平均"}

    return {"score": np.nan, "source": "該当なし"}

def classify_rank(score):
    if pd.isna(score):
        return ""
    # 4項目版の3年分検証に合わせたランク帯
    # S: 複勝率 81%以上帯
    # A: 複勝率 70%以上 81%未満帯
    if score >= 40:
        return "S"
    if score >= 34:
        return "A"
    if score >= 28:
        return "B"
    if score >= 22:
        return "C"
    return "D"

def score_race_df(race_df, prepared_tables):
    out = race_df.copy()
    weighted_parts = []
    for kind, payload in prepared_tables.items():
        table_df, fallback_df = payload
        result = out.apply(lambda row: lookup_component(row, table_df, fallback_df, kind), axis=1)
        score_col = f"{TABLE_SPECS[kind]['label']}点"
        out[score_col] = result.apply(lambda x: x["score"])
        weighted_parts.append(out[score_col] * (WEIGHTS[kind] / 100.0))

    out["総合点"] = sum(weighted_parts).round(2)
    out["順位"] = out.groupby(["場所", "raceNo"], dropna=False)["総合点"].rank(method="min", ascending=False)
    out["順位"] = out["順位"].fillna(999).astype(int)
    out["ランク"] = out["総合点"].apply(classify_rank)

    # Sは各レース1頭まで
    top_idx = out.groupby(["場所", "raceNo"], dropna=False)["総合点"].idxmax()
    out.loc[out["ランク"] == "S", "ランク"] = "A"
    if len(top_idx) > 0:
        top_mask = out.index.isin(top_idx)
        out.loc[top_mask & (out["総合点"] >= 40), "ランク"] = "S"

    out["レース"] = out["場所"].astype(str) + out["raceNo"].astype(str) + "R"
    out = out.sort_values(["場所", "raceNo", "順位", "horseNo"], ascending=[True, True, True, True])
    return out

def render_rank_cards(date_val, race_val, race_name_val, dist_text, card_df):
    title_parts = []
    if str(date_val).strip():
        title_parts.append(str(date_val).strip())
    if str(race_val).strip():
        title_parts.append(str(race_val).strip())
    title = " ".join(title_parts).strip() or str(race_val).strip() or "レース"

    subtitle_parts = []
    if str(race_name_val).strip():
        subtitle_parts.append(str(race_name_val).strip())
    if str(dist_text).strip():
        subtitle_parts.append(str(dist_text).strip())
    subtitle = " / ".join(subtitle_parts)

    css = """
<style>
.kv-wrap{background:linear-gradient(180deg,#071225 0%,#0b1730 100%);border-radius:20px;padding:14px 12px;color:#f5f7fb;box-shadow:0 8px 24px rgba(0,0,0,.20);margin:8px 0 14px 0}
.kv-title{font-size:18px;font-weight:800;line-height:1.15;margin:0 0 4px 0;letter-spacing:.15px}
.kv-subtitle{font-size:11px;color:#c8d2e8;margin:0 0 10px 0}
.horse-card{display:flex;align-items:center;justify-content:space-between;gap:8px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.04);border-radius:14px;padding:8px 10px;margin:6px 0}
.horse-left{min-width:0;flex:1}
.horse-top{display:flex;align-items:center;gap:6px;min-width:0}
.horse-no{font-size:11px;color:#9fb0d3;font-weight:700;flex:0 0 auto}
.horse-name{font-size:14px;font-weight:800;color:#ffffff;line-height:1.15;word-break:break-all}
.rank-pill{min-width:44px;text-align:center;font-size:18px;font-weight:900;border-radius:12px;padding:7px 12px;flex:0 0 auto;border:2px solid #334a76;color:#edf3ff;background:rgba(93,122,183,.12)}
.rank-S{border-color:#e7c65b;color:#fff2ba;background:rgba(231,198,91,.15);box-shadow:inset 0 0 0 1px rgba(231,198,91,.25)}
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

st.title("競馬ランクアプリ v9.0 4Factor")
st.write("脚質×前走場所×種牡馬×母父馬 の4項目で内部採点し、表示はシンプルにしています。")

with st.sidebar:
    st.subheader("出走馬CSV")
    race_file = st.file_uploader("出走馬CSV", type=["csv"], key="race")
    st.subheader("過去データCSV（未指定なら同フォルダの既定ファイルを使用）")
    style_file = st.file_uploader("脚質CSV", type=["csv"], key="style")
    prev_file = st.file_uploader("前走場所CSV", type=["csv"], key="prevtrack")
    sire_file = st.file_uploader("種牡馬CSV", type=["csv"], key="sire")
    damsire_file = st.file_uploader("母父馬CSV", type=["csv"], key="damsire")

st.caption("現在の比重：脚質37.5 / 前走場所22.5 / 種牡馬15 / 母父馬25")
st.caption("ランク目安：S=複勝率81%以上帯 / A=複勝率70%以上81%未満帯。Sは各レース1頭まで。")

if race_file is None:
    st.info("まず出走馬CSVをアップロードしてください。")
    st.stop()

try:
    race_df_raw = read_csv_any(race_file)
    race_df = prepare_race_df(race_df_raw)

    loaded = {}
    for kind, uploader in [("style", style_file), ("prevtrack", prev_file), ("sire", sire_file), ("damsire", damsire_file)]:
        raw = load_default_or_upload(kind, uploader)
        if raw is None:
            raise FileNotFoundError(f"{kind} のCSVが見つかりません。アップロードするか app.py と同じフォルダへ配置してください。")
        loaded[kind] = prepare_stat_table(raw, kind)

    result_df = score_race_df(race_df, loaded)

    export_df = result_df[["date", "レース", "raceName", "horseNo", "horseName", "ランク"]].copy()
    export_df = export_df.rename(columns={
        "date": "日付",
        "raceName": "レース名",
        "horseNo": "馬番",
        "horseName": "馬名",
    })

    st.success("自動採点が完了しました。")

    tab1, tab2 = st.tabs(["予想結果", "元データ確認"])
    with tab1:
        grouped = result_df.groupby(["date", "レース", "raceName", "距離表示"], dropna=False, sort=False)
        for (date_val, race_val, race_name_val, dist_val), g in grouped:
            show_df = g[["horseNo", "horseName", "ランク"]].copy().rename(columns={"horseNo": "馬番", "horseName": "馬名"})
            st.markdown(render_rank_cards(date_val, race_val, race_name_val, dist_val, show_df), unsafe_allow_html=True)

    with tab2:
        st.write("出走馬CSV")
        st.dataframe(race_df_raw, use_container_width=True, hide_index=True)

    csv_data = export_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="予想結果CSVをダウンロード",
        data=csv_data,
        file_name="4factor_rank_predictions.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"エラー: {e}")
