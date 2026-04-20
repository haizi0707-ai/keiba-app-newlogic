
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="競馬 ランクアプリ v7.2 Persist", layout="wide")

APP_DIR = Path(__file__).resolve().parent
STORE_PATH = APP_DIR / "saved_history_store.json"

DEFAULT_STATE = {
    "history_df": None,
    "summary_data": None,
    "ranked_prediction_df": None,
    "preview_race_df": None,
    "preview_title": "レースランキング",
    "generated_svg_text": None,
    "_boot_loaded": False,
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #071223 0%, #0a1730 100%); }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.block-container { max-width: 1120px; padding-top: 1rem; padding-bottom: 3rem; }
.main-title { font-size: 2.1rem; font-weight: 800; color: #ffffff; line-height: 1.25; margin-bottom: 0.35rem; }
.sub-title { color: #dce9ff; font-size: 1rem; line-height: 1.8; margin-bottom: 1rem; }
.info-box { background: rgba(166,198,255,.12); border: 1px solid rgba(166,198,255,.24); border-radius: 20px; padding: 1.1rem 1.15rem; color:#fff; font-size:1rem; line-height:1.9; margin-bottom:1rem; }
.section-card { background: linear-gradient(180deg, rgba(10,20,40,.97) 0%, rgba(8,16,32,.97) 100%); border:1px solid rgba(122,154,214,.22); border-radius:24px; padding:1rem 1rem .9rem 1rem; margin-bottom:1rem; box-shadow:0 8px 28px rgba(0,0,0,.16); }
.section-title { color:#fff; font-size:1.45rem; font-weight:800; margin:0; }
.small-note { color:#eef4ff; font-size:1rem; font-weight:600; }
.metric-card { background: linear-gradient(180deg, rgba(12,22,42,.98) 0%, rgba(10,18,36,.98) 100%); border:1px solid rgba(130,160,220,.20); border-radius:22px; padding:1rem; margin-bottom:1rem; }
.metric-label { color:#dbe7ff; font-size:1rem; margin-bottom:.25rem; }
.metric-value { color:#fff; font-size:2.2rem; font-weight:800; line-height:1.1; }
[data-testid="stMarkdownContainer"] p, [data-testid="stExpander"] summary, label[data-testid="stWidgetLabel"] { color:#f8fbff !important; font-weight:600 !important; }
[data-testid="stFileUploader"] { background:#13233d !important; border:1px solid rgba(46,204,113,.40) !important; border-radius:18px !important; padding:.55rem !important; }
[data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"] { background:#13233d !important; border:1px solid rgba(46,204,113,.32) !important; border-radius:16px !important; color:#fff !important; }
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] p { color:#fff !important; }
[data-testid="stFileUploader"] button, [data-testid="stBaseButton-secondary"] { background:linear-gradient(90deg,#14b76b,#1ed37f) !important; color:#fff !important; font-weight:800 !important; border:none !important; border-radius:14px !important; }
[data-testid="stFileUploader"] svg { fill:#fff !important; }
[data-baseweb="select"] > div { background:#13233d !important; color:#fff !important; border:1px solid rgba(216,92,92,.45) !important; border-radius:16px !important; min-height:3rem !important; }
[data-baseweb="select"] * { color:#fff !important; }
.stButton > button, .stDownloadButton > button { border-radius:18px !important; font-weight:800 !important; padding:.82rem 1rem !important; border:none !important; box-shadow:none !important; font-size:1rem !important; }
.green-btn button { background:linear-gradient(90deg,#14b76b,#1ed37f) !important; color:#fff !important; }
.orange-btn button { background:linear-gradient(90deg,#cc8b16,#f0a21a) !important; color:#fff !important; }
.red-btn button { background:linear-gradient(90deg,#b94e4e,#d85c5c) !important; color:#fff !important; }
.dark-btn button { background:#1f3151 !important; color:#fff !important; border:1px solid rgba(255,255,255,.12) !important; }
.stButton > button:disabled, .stDownloadButton > button:disabled { background:#6b7280 !important; color:#f8fbff !important; border:1px solid rgba(255,255,255,.18) !important; opacity:1 !important; }
[data-testid="stAlert"] { border-radius:16px !important; }
[data-testid="stAlert"] * { color:#fff !important; }
.preview-panel { background:#091426; border:1px solid rgba(126,156,214,.18); border-radius:20px; padding:1rem; }
.preview-title { font-size:2rem; font-weight:800; color:#fff; margin-bottom:.25rem; }
.preview-sub { color:#cfdcff; font-size:1rem; margin-bottom:1rem; }
.preview-row { display:grid; grid-template-columns:1fr 90px; gap:12px; align-items:center; padding:10px 12px; border-radius:16px; margin-bottom:8px; background:rgba(255,255,255,.03); }
.preview-name { color:#fff; font-size:1.35rem; font-weight:800; line-height:1.2; }
.preview-class { color:#cdd9f4; font-size:.95rem; margin-top:4px; }
.rank-box { text-align:center; border-radius:14px; padding:8px 0; font-weight:800; font-size:1.35rem; color:#f7fbff; background:#1d2b46; border:1px solid rgba(120,160,220,.22); }
.rank-S { background:#4c2fa8; } .rank-A { background:#1f8b58; } .rank-B { background:#2c6eb8; } .rank-C { background:#a97115; } .rank-D { background:#5a6578; }
.cond-table { width:100%; border-collapse:collapse; }
.cond-table th, .cond-table td { text-align:left; padding:12px 10px; border-bottom:1px solid rgba(255,255,255,.08); color:#f8fbff; vertical-align:top; }
.cond-table th { color:#fff; font-size:1rem; font-weight:700; }
.cond-cond { font-size:.92rem; color:#d7e5ff; line-height:1.45; word-break:break-word; }
@media (max-width: 720px) { .preview-row { grid-template-columns:1fr 72px; } .preview-name { font-size:1.1rem; } }
</style>
""", unsafe_allow_html=True)

REQUIRED_PRED_COLS = ["日付","開催","R","レース名","馬番","馬名","種牡馬","調教師","騎手","距離","馬場状態"]
HISTORY_REQUIRED = ["種牡馬","調教師","開催","距離"]

def save_store_to_disk():
    if st.session_state.history_df is None or st.session_state.summary_data is None:
        return False, "保存できる過去データがありません。"
    payload = {
        "history_rows": st.session_state.history_df.to_dict(orient="records"),
        "summary_data": st.session_state.summary_data,
    }
    STORE_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return True, f"アプリ内に保存しました。件数: {len(st.session_state.history_df):,}"

def load_store_from_disk():
    if not STORE_PATH.exists():
        return False, "保存データがありません。"
    payload = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    st.session_state.history_df = pd.DataFrame(payload.get("history_rows", []))
    st.session_state.summary_data = payload.get("summary_data", None)
    return True, f"保存データを読み込みました。件数: {len(st.session_state.history_df):,}"

def delete_store_from_disk():
    if STORE_PATH.exists():
        STORE_PATH.unlink()
        return True, "保存データを削除しました。"
    return False, "保存データはありません。"

def boot_load_saved_data():
    if st.session_state.get("_boot_loaded"):
        return
    st.session_state["_boot_loaded"] = True
    if STORE_PATH.exists():
        try:
            payload = json.loads(STORE_PATH.read_text(encoding="utf-8"))
            st.session_state.history_df = pd.DataFrame(payload.get("history_rows", []))
            st.session_state.summary_data = payload.get("summary_data", None)
        except Exception:
            pass

boot_load_saved_data()

def read_uploaded_csv(uploaded_file):
    if uploaded_file is None:
        return None
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file)

def normalize_text_series(series):
    return series.astype(str).str.replace("\u3000", " ", regex=False).str.strip()

def normalize_columns(df):
    df = df.copy()
    rename_map = {
        "date":"日付","track":"開催","raceNo":"R","race_name":"レース名","raceName":"レース名",
        "horseNo":"馬番","horse_no":"馬番",
        "horseName":"馬名","horse_name":"馬名","trainer":"調教師","sire":"種牡馬","damSire":"母父馬",
        "surface":"芝ダ","distance":"距離","going":"馬場状態","popularity":"人気","finishPosition":"着順",
        "winOdds":"単勝","prevTrack":"前開催","prevDistance":"前距離","prevGoing":"前走馬場状態",
        "prevDate":"前走日付","intervalCategory":"間隔カテゴリ","distanceChange":"距離変化",
        "trackChange":"開催変化","prevJockey":"前騎手","jockey":"騎手","placed":"複勝フラグ",
        "category":"分類","trust":"信頼度",
    }
    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    for c in ["馬名","レース名","開催","調教師","種牡馬","母父馬","騎手","前騎手","馬番"]:
        if c in df.columns:
            df[c] = normalize_text_series(df[c])
    return df

def ensure_race_key_columns(df):
    df = df.copy()
    for col in ["日付","開催","レース名","馬名"]:
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
        return "芝", pd.to_numeric(s.replace("芝",""), errors="coerce")
    if s.startswith("ダ"):
        return "ダート", pd.to_numeric(s.replace("ダ",""), errors="coerce")
    if s.startswith("障"):
        return "障害", pd.to_numeric(s.replace("障",""), errors="coerce")
    return "", pd.to_numeric(s, errors="coerce")

def add_surface_distance_columns(df):
    df = df.copy()
    if "距離" in df.columns:
        parsed = df["距離"].apply(parse_surface_distance)
        if "芝ダ" not in df.columns:
            df["芝ダ"] = parsed.apply(lambda x: x[0])
        df["距離数値"] = parsed.apply(lambda x: x[1])
    else:
        df["距離数値"] = np.nan
    return df

def get_distance_band(distance):
    if pd.isna(distance):
        return "不明"
    d = float(distance)
    if d <= 1400: return "短距離"
    if d <= 1700: return "マイル"
    if d <= 2000: return "中距離"
    if d <= 2400: return "中長距離"
    return "長距離"

def get_going_group(going):
    if pd.isna(going):
        return "不明"
    return "良" if str(going).strip() == "良" else "道悪"

def calc_place_flag(series):
    s = pd.to_numeric(series, errors="coerce")
    return np.where((s >= 1) & (s <= 3), 1, 0)

def build_history_summary(history_df):
    df = history_df.copy()
    missing = [c for c in HISTORY_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("過去レースCSVの必須列不足: " + ", ".join(missing))
    df = add_surface_distance_columns(df)
    df["距離帯"] = df["距離数値"].apply(get_distance_band)
    df["馬場区分"] = df["馬場状態"].apply(get_going_group) if "馬場状態" in df.columns else "不明"
    if "複勝フラグ" in df.columns:
        df["placed_flag"] = pd.to_numeric(df["複勝フラグ"], errors="coerce").fillna(0).astype(int)
    elif "着順" in df.columns:
        df["placed_flag"] = calc_place_flag(df["着順"])
    else:
        raise ValueError("過去レースCSVに 複勝フラグ または 着順 列が必要です。")

    def make_group(keys):
        grouped = df.groupby(keys, dropna=False).agg(count=("placed_flag","size"), placed=("placed_flag","sum")).reset_index()
        out = {}
        for row in grouped.itertuples(index=False):
            key = "|||".join(str(x) for x in row[:-2])
            count = int(row[-2]); placed = int(row[-1])
            out[key] = {"count": count, "placed": placed, "place_rate": placed / count if count else 0.0}
        return out

    return {
        "source_rows": int(len(df)),
        "sire_track": make_group(["種牡馬","開催"]),
        "sire_dist": make_group(["種牡馬","距離帯"]),
        "sire_going": make_group(["種牡馬","馬場区分"]),
        "trainer_track": make_group(["調教師","開催"]),
        "trainer_dist": make_group(["調教師","距離帯"]),
        "trainer_jockey": make_group(["調教師","騎手"]) if "騎手" in df.columns else {},
    }

def history_backup_payload():
    if st.session_state.history_df is None or st.session_state.summary_data is None:
        return None
    return {"history_rows": st.session_state.history_df.to_dict(orient="records"), "summary_data": st.session_state.summary_data}

def restore_history_backup(uploaded_json):
    uploaded_json.seek(0)
    payload = json.load(uploaded_json)
    st.session_state.history_df = pd.DataFrame(payload.get("history_rows", []))
    st.session_state.summary_data = payload.get("summary_data", None)

def lookup_score(summary_data, map_name, key, min_count=20, min_rate=50):
    data = summary_data.get(map_name, {}).get(key)
    if not data:
        return {"score": None, "count": None, "rate": None}
    count = data.get("count", 0)
    rate = float(data.get("place_rate", 0)) * 100
    if count < min_count or rate < min_rate:
        return {"score": None, "count": count, "rate": rate}
    return {"score": rate, "count": count, "rate": rate}

def lookup_adjust(summary_data, map_name, key, min_count=15, min_rate=50):
    data = summary_data.get(map_name, {}).get(key)
    if not data:
        return {"adj": 0.0, "count": None, "rate": None}
    count = data.get("count", 0)
    rate = float(data.get("place_rate", 0)) * 100
    if count < min_count or rate < min_rate:
        return {"adj": 0.0, "count": count, "rate": rate}
    if rate >= 62:
        adj = 3.0
    elif rate >= 56:
        adj = 2.0
    else:
        adj = 1.0
    return {"adj": adj, "count": count, "rate": rate}

def classify_rank(rank):
    return {"S":"本命候補","A":"相手本線","B":"強穴","C":"穴候補","D":"軽視候補"}.get(str(rank), "軽視候補")

def score_to_rank_in_race(group_df):
    g = group_df.sort_values(["強条件数","補正点","総合点","母数最小"], ascending=[False,False,False,False]).copy()
    g["信頼度"] = "D"

    if len(g) >= 1 and int(g.iloc[0]["強条件数"]) >= 2:
        g.iloc[0, g.columns.get_loc("信頼度")] = "S"
    elif len(g) >= 1 and int(g.iloc[0]["強条件数"]) >= 1:
        g.iloc[0, g.columns.get_loc("信頼度")] = "A"

    if len(g) >= 2:
        t1 = g.iloc[0]; t2 = g.iloc[1]
        if int(t2["強条件数"]) >= 2 and ((t1["総合点"] + t1["補正点"]) - (t2["総合点"] + t2["補正点"]) <= 2.5):
            if g.iloc[0]["信頼度"] == "S":
                g.iloc[1, g.columns.get_loc("信頼度")] = "S"
            elif g.iloc[0]["信頼度"] == "A":
                g.iloc[1, g.columns.get_loc("信頼度")] = "A"

    for idx in g.index:
        if g.loc[idx, "信頼度"] in ["S", "A"]:
            continue
        cnt = int(g.loc[idx, "強条件数"])
        score = float(g.loc[idx, "総合点"] + g.loc[idx, "補正点"])
        if cnt >= 2 and score >= 55:
            g.loc[idx, "信頼度"] = "B"
        elif cnt >= 1 and score >= 52:
            g.loc[idx, "信頼度"] = "C"
        else:
            g.loc[idx, "信頼度"] = "D"

    g["ランク"] = g["信頼度"]
    g["分類"] = g["信頼度"].apply(classify_rank)
    return g

def prepare_prediction_df(df, summary_data):
    df = normalize_columns(df)
    df = ensure_race_key_columns(df)
    df = add_surface_distance_columns(df)

    for col in REQUIRED_PRED_COLS:
        if col not in df.columns:
            raise ValueError(f"必須列不足: {col}")

    df["距離帯"] = df["距離数値"].apply(get_distance_band)
    df["馬場区分"] = df["馬場状態"].apply(get_going_group)

    rows = []
    for _, r in df.iterrows():
        sire = r.get("種牡馬","")
        trainer = r.get("調教師","")
        jockey = r.get("騎手","")
        track = r.get("開催","")
        dist_band = r.get("距離帯","")
        going = r.get("馬場区分","")

        base_conds = [
            ("種牡馬×競馬場", lookup_score(summary_data, "sire_track", f"{sire}|||{track}")),
            ("種牡馬×距離帯", lookup_score(summary_data, "sire_dist", f"{sire}|||{dist_band}")),
            ("調教師×競馬場", lookup_score(summary_data, "trainer_track", f"{trainer}|||{track}")),
            ("調教師×距離帯", lookup_score(summary_data, "trainer_dist", f"{trainer}|||{dist_band}")),
        ]
        strong = [c for c in base_conds if c[1]["score"] is not None]
        strong_scores = [c[1]["score"] for c in strong]
        strong_counts = [c[1]["count"] for c in strong if c[1]["count"] is not None]
        total = np.mean(strong_scores) if strong_scores else 0.0
        min_count = int(min(strong_counts)) if strong_counts else 0

        adj1 = lookup_adjust(summary_data, "sire_going", f"{sire}|||{going}")
        adj2 = lookup_adjust(summary_data, "trainer_jockey", f"{trainer}|||{jockey}")

        adj_total = adj1["adj"] + adj2["adj"]
        adj_tags = []
        if adj1["adj"] > 0:
            adj_tags.append(f'種牡馬×馬場:+{adj1["adj"]} ({round(adj1["rate"],1)}%)')
        if adj2["adj"] > 0:
            adj_tags.append(f'調教師×騎手:+{adj2["adj"]} ({round(adj2["rate"],1)}%)')
        if not adj_tags:
            adj_tags.append("補正なし")

        cond_names = " / ".join([f'{name}:{round(item["rate"],1)}%' for name, item in strong]) if strong else "強条件なし"

        row = r.to_dict()
        row["総合点"] = round(total, 1)
        row["強条件数"] = len(strong)
        row["母数最小"] = min_count
        row["採用条件"] = cond_names
        row["補正点"] = round(adj_total, 1)
        row["補正内容"] = " / ".join(adj_tags)
        rows.append(row)

    out = pd.DataFrame(rows)
    group_cols = [c for c in ["日付","開催","R","レース名"] if c in out.columns]
    pieces = [score_to_rank_in_race(g) for _, g in out.groupby(group_cols, dropna=False)]
    out = pd.concat(pieces).sort_index()
    if group_cols:
        out = out.sort_values(group_cols + ["強条件数","補正点","総合点"], ascending=[True]*len(group_cols)+[False,False,False])
    return out

def race_options_from_df(df):
    need_cols = ["日付","開催","レース名"]
    if not all(c in df.columns for c in need_cols):
        return []
    cols = [c for c in ["日付","開催","R","レース名"] if c in df.columns]
    temp = df[cols].drop_duplicates().fillna("")
    return [(f"{row.get('日付','')} {row.get('開催','')} {row.get('R','')} {row.get('レース名','')}", row.to_dict()) for _, row in temp.iterrows()]

def filter_race_df(df, race_dict):
    out = df.copy()
    for col, val in race_dict.items():
        if col in out.columns:
            out = out[out[col].astype(str) == str(val)]
    return out.copy()

def build_race_svg_text(race_df, title):
    width = 1200
    row_h = 78
    height = 220 + len(race_df) * row_h
    subtitle = ""
    if not race_df.empty:
        first = race_df.iloc[0]
        subtitle = f"{first.get('レース名','')} / {first.get('距離','')} / {len(race_df)}頭"
    rank_colors = {"S":"#5B34D6","A":"#1F8B58","B":"#2C6EB8","C":"#B97E16","D":"#5A6578"}
    def esc(x):
        return str(x).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#081324"/>',
        f'<text x="40" y="68" fill="#FFFFFF" font-size="34" font-weight="800" font-family="system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">{esc(title)}</text>',
        f'<text x="40" y="114" fill="#CFDCFF" font-size="22" font-weight="500" font-family="system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">{esc(subtitle)}</text>',
    ]
    y = 150
    for _, row in race_df.iterrows():
        horse = esc(row.get("馬名",""))
        category = esc(row.get("分類",""))
        rank = str(row.get("信頼度","D"))
        color = rank_colors.get(rank, "#5A6578")
        lines.extend([
            f'<rect x="40" y="{y}" rx="18" ry="18" width="1120" height="60" fill="rgba(255,255,255,0.03)" stroke="rgba(126,156,214,0.10)"/>',
            f'<text x="68" y="{y+28}" fill="#FFFFFF" font-size="22" font-weight="800" font-family="system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">{horse}</text>',
            f'<text x="68" y="{y+50}" fill="#CDD9F4" font-size="16" font-weight="500" font-family="system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">{category}</text>',
            f'<rect x="980" y="{y+9}" rx="14" ry="14" width="120" height="42" fill="{color}" />',
            f'<text x="1040" y="{y+36}" text-anchor="middle" fill="#FFFFFF" font-size="22" font-weight="800" font-family="system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif">{rank}</text>',
        ])
        y += row_h
    lines.append("</svg>")
    return "\n".join(lines)

def unique_race_count(df):
    if df is None or df.empty:
        return 0
    cols = [c for c in ["日付","開催","R","レース名"] if c in df.columns]
    return len(df[cols].drop_duplicates()) if cols else 0

def saved_condition_count(summary_data):
    if not summary_data:
        return 0
    keys = ["sire_track","sire_dist","sire_going","trainer_track","trainer_dist","trainer_jockey"]
    return sum(len(summary_data.get(k, {})) for k in keys)

def render_preview_html(race_df, title):
    subtitle = ""
    if not race_df.empty:
        first = race_df.iloc[0]
        subtitle = f"{first.get('レース名','')} / {first.get('距離','')} / {len(race_df)}頭"
    html = [f'<div class="preview-panel"><div class="preview-title">{title}</div>']
    if subtitle:
        html.append(f'<div class="preview-sub">{subtitle}</div>')
    for _, row in race_df.iterrows():
        rank = str(row.get("信頼度","D"))
        category = row.get("分類","")
        name = row.get("馬名","")
        html.append(
            f'<div class="preview-row">'
            f'<div><div class="preview-name">{name}</div><div class="preview-class">{category}</div></div>'
            f'<div class="rank-box rank-{rank}">{rank}</div>'
            f'</div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

def render_condition_table(race_df):
    if race_df.empty:
        st.info("予想CSVを読み込むと表示されます。")
        return
    rows = []
    for _, row in race_df.iterrows():
        rows.append(
            f'<tr>'
            f'<td>{row.get("馬名","")}</td>'
            f'<td>{row.get("信頼度","")}</td>'
            f'<td>{row.get("分類","")}</td>'
            f'<td>{row.get("強条件数","")}</td>'
            f'<td class="cond-cond">{row.get("採用条件","")}</td>'
            f'<td class="cond-cond">{row.get("補正内容","")}</td>'
            f'</tr>'
        )
    html = (
        '<table class="cond-table">'
        '<thead><tr><th>馬名</th><th>信頼度</th><th>分類</th><th>強条件数</th><th>採用条件</th><th>補正内容</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
    )
    st.markdown(html, unsafe_allow_html=True)

st.markdown('<div class="main-title">競馬 ランクアプリ<br>v7.2 Persist + Adjust</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">メイン4条件に補正2条件を足し、保存機能も付けた版です。</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
<b>メイン4条件</b><br>
種牡馬×競馬場 / 種牡馬×距離帯 / 調教師×競馬場 / 調教師×距離帯<br>
→ 採用基準: <b>母数20以上・複勝率50%以上</b><br><br>
<b>補正2条件</b><br>
種牡馬×馬場 / 調教師×騎手<br>
→ 採用基準: <b>母数15以上・複勝率50%以上</b><br><br>
<b>保存機能</b><br>
アプリ内保存・再読込に対応しています。<br>
ただし Streamlit Cloud の再起動や再デプロイ時は消えることがあるため、履歴バックアップJSONの保存も推奨です。
</div>
""", unsafe_allow_html=True)

if STORE_PATH.exists():
    st.markdown('<div class="small-note">保存データあり: アプリを閉じても再読込できます（ただしサーバー再起動時は消える場合あり）</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="small-note">保存データなし</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">過去レースCSV（収集用）</div></div>', unsafe_allow_html=True)
history_file = st.file_uploader("過去レースCSV", type=["csv"], key="history_uploader", label_visibility="collapsed")
history_backup_file = st.file_uploader("履歴バックアップJSON復元", type=["json"], key="history_backup_uploader", label_visibility="collapsed")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown('<div class="green-btn">', unsafe_allow_html=True)
    import_history = st.button("過去CSV取込", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="dark-btn">', unsafe_allow_html=True)
    save_history_local = st.button("アプリ内保存", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="dark-btn">', unsafe_allow_html=True)
    load_history_local = st.button("保存データ読込", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="dark-btn">', unsafe_allow_html=True)
    backup_history = st.button("JSON保存", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c5:
    st.markdown('<div class="red-btn">', unsafe_allow_html=True)
    clear_history = st.button("保存削除", use_container_width=True)
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
            ok, msg = save_store_to_disk()
            st.success(f"過去データを取り込みました。件数: {len(raw_history):,}")
            if ok:
                st.info(msg)
        except Exception as e:
            st.error(f"過去レースCSVの読み込みでエラーが出ました: {e}")

if save_history_local:
    ok, msg = save_store_to_disk()
    (st.success if ok else st.error)(msg)

if load_history_local:
    ok, msg = load_store_from_disk()
    (st.success if ok else st.error)(msg)

if history_backup_file is not None:
    try:
        restore_history_backup(history_backup_file)
        ok, msg = save_store_to_disk()
        st.success("履歴バックアップを復元しました。")
        if ok:
            st.info(msg)
    except Exception as e:
        st.error(f"履歴バックアップの復元でエラーが出ました: {e}")

if backup_history:
    payload = history_backup_payload()
    if payload is None:
        st.error("保存できる過去データがありません。")
    else:
        st.download_button("履歴バックアップJSONをダウンロード", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), file_name="keiba_history_backup.json", mime="application/json", use_container_width=True)

if clear_history:
    st.session_state.history_df = None
    st.session_state.summary_data = None
    st.session_state.ranked_prediction_df = None
    st.session_state.preview_race_df = None
    st.session_state.preview_title = "レースランキング"
    st.session_state.generated_svg_text = None
    ok, msg = delete_store_from_disk()
    st.success(msg if ok else msg)

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
        st.error("先に過去レースCSVを取り込むか、保存データを読み込んでください。")
    elif pred_file is None:
        st.error("予想CSVを選択してください。")
    else:
        try:
            pred_raw = read_uploaded_csv(pred_file)
            ranked = prepare_prediction_df(pred_raw, st.session_state.summary_data)
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
            st.session_state.generated_svg_text = build_race_svg_text(preview_df, preview_title)
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
    st.markdown(f'<div class="small-note">予想CSV読込完了: {len(ranked_df):,}頭 / {unique_race_count(ranked_df)}レース</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="small-note">まだ読み込んでいません。</div>', unsafe_allow_html=True)

if make_image:
    if st.session_state.preview_race_df is None or st.session_state.preview_race_df.empty:
        st.error("先に予想CSVを読み込んでください。")
    else:
        st.session_state.generated_svg_text = build_race_svg_text(st.session_state.preview_race_df, st.session_state.preview_title)
        st.success("画像を作成しました。")

if save_image:
    if st.session_state.generated_svg_text is None:
        st.error("先に画像を作成してください。")
    else:
        st.download_button("SVG画像をダウンロード", data=st.session_state.generated_svg_text.encode("utf-8"), file_name="keiba_rank_image.svg", mime="image/svg+xml", use_container_width=True)

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
    st.markdown(f'<div class="metric-card"><div class="metric-label">保存済み履歴件数</div><div class="metric-value">{history_count}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">保存済み条件数</div><div class="metric-value">{condition_count}</div></div>', unsafe_allow_html=True)
c3, c4 = st.columns(2)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">予想CSVレース数</div><div class="metric-value">{prediction_race_count}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">予想CSV馬数</div><div class="metric-value">{prediction_horse_count}</div></div>', unsafe_allow_html=True)
