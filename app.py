
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="競馬 ランクアプリ v8.0 Master Logic", layout="wide")

APP_DIR = Path(__file__).resolve().parent
MASTER_PATH = APP_DIR / "master_logic.csv"
SUMMARY_PATH = APP_DIR / "master_summary.csv"
PRED_STORE_PATH = APP_DIR / "saved_prediction_log.json"

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
[data-baseweb="select"] > div { background:#13233d !important; color:#fff !important; border:1px solid rgba(216,92,92,.45) !important; border-radius:16px !important; min-height:3rem !important; }
[data-baseweb="select"] * { color:#fff !important; }
.stButton > button, .stDownloadButton > button { border-radius:18px !important; font-weight:800 !important; padding:.82rem 1rem !important; border:none !important; box-shadow:none !important; font-size:1rem !important; }
.green-btn button { background:linear-gradient(90deg,#14b76b,#1ed37f) !important; color:#fff !important; }
.orange-btn button { background:linear-gradient(90deg,#cc8b16,#f0a21a) !important; color:#fff !important; }
.red-btn button { background:linear-gradient(90deg,#b94e4e,#d85c5c) !important; color:#fff !important; }
.dark-btn button { background:#1f3151 !important; color:#fff !important; border:1px solid rgba(255,255,255,.12) !important; }
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
</style>
""", unsafe_allow_html=True)

PRED_REQUIRED = ["date","track","raceNo","raceName","horseNo","horseName","jockey","trainer","sire","distance","going"]

def normalize_columns(df):
    rename_map = {
        "date":"date","track":"track","raceNo":"raceNo","race_name":"raceName","raceName":"raceName",
        "horseNo":"horseNo","horse_no":"horseNo","horseName":"horseName","horse_name":"horseName",
        "jockey":"jockey","trainer":"trainer","sire":"sire","damSire":"damSire","distance":"distance",
        "going":"going","fieldSize":"fieldSize","style":"style","prevTrack":"prevTrack",
        "prev_track":"prevTrack","winStyle":"winStyle","win_style":"winStyle","raceNo.":"raceNo",
        "場所":"track","芝・ダ":"surface","距離":"distance","馬場状態":"going","馬番":"horseNo","馬名":"horseName",
        "騎手":"jockey","調教師":"trainer","種牡馬":"sire","母父馬":"damSire","脚質":"style","前走場所":"prevTrack",
        "決め手":"winStyle","日付":"date","レース名":"raceName","Ｒ":"raceNo","R":"raceNo",
    }
    df = df.rename(columns=rename_map)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.replace("\u3000", " ", regex=False).str.strip()
    return df

def parse_distance(v):
    s = str(v).strip()
    if s.startswith("芝"):
        return "芝", "".join(ch for ch in s if ch.isdigit())
    if s.startswith("ダ"):
        return "ダ", "".join(ch for ch in s if ch.isdigit())
    if s.startswith("障"):
        return "障", "".join(ch for ch in s if ch.isdigit())
    digits = "".join(ch for ch in s if ch.isdigit())
    return "", digits

def style_tag(v):
    s = str(v).strip()
    if s in ("逃げ",): return "逃げ"
    if s in ("先行",): return "先行"
    if s in ("差し","中団","後方"): return "差し"
    if s in ("追込",): return "追込"
    if "マクリ" in s or "まくり" in s or "ﾏｸﾘ" in s: return "まくり"
    return s if s and s != "nan" else ""

def winstyle_tag(v):
    s = str(v).strip()
    if s in ("逃げ",): return "逃げ切り"
    if s in ("先行",): return "先行押し切り"
    if s in ("差し","中団"): return "差し"
    if s in ("後方","追込"): return "追い込み"
    if "マクリ" in s or "まくり" in s or "ﾏｸﾘ" in s: return "まくり"
    return s if s and s != "nan" else ""

def going_tag(v):
    s = str(v).strip()
    return s if s and s != "nan" else ""

@st.cache_data
def load_master():
    master = pd.read_csv(MASTER_PATH, encoding="utf-8-sig")
    summary = pd.read_csv(SUMMARY_PATH, encoding="utf-8-sig")
    return master, summary

def lookup_condition(master, track, surface, distance, cond, content):
    m = master[
        (master["場所"].astype(str) == str(track)) &
        (master["芝・ダ"].astype(str) == str(surface)) &
        (master["距離"].astype(str) == str(distance)) &
        (master["条件"].astype(str) == str(cond)) &
        (master["内容"].astype(str) == str(content))
    ]
    if m.empty:
        return None
    row = m.sort_values(["母数","単勝率","複勝率"], ascending=[False,False,False]).iloc[0]
    return {
        "rank": str(row["ランク"]),
        "count": int(row["母数"]),
        "win_rate": float(row["単勝率"]),
        "place_rate": float(row["複勝率"]),
        "content": str(row["内容"]),
    }

def rank_to_point(rank):
    return {"S":5,"A":4,"B":3,"C":2,"D":1}.get(str(rank), 0)

def final_rank(avg):
    if avg >= 4.6: return "S"
    if avg >= 3.8: return "A"
    if avg >= 3.0: return "B"
    if avg >= 2.2: return "C"
    return "D"

def classify(rank):
    return {"S":"本命候補","A":"相手本線","B":"強穴","C":"穴候補","D":"軽視候補"}.get(rank, "軽視候補")

def evaluate_prediction(pred_df, master):
    rows = []
    for _, r in pred_df.iterrows():
        surface, dist_num = parse_distance(r.get("distance",""))
        track = r.get("track","")
        sire = r.get("sire","")
        trainer = r.get("trainer","")
        style = style_tag(r.get("style",""))
        prev_track = r.get("prevTrack","")
        going = going_tag(r.get("going",""))
        win_style = winstyle_tag(r.get("winStyle",""))

        cond_values = {
            "血統": sire,
            "調教師": trainer,
            "脚質": style,
            "前走場所": prev_track,
            "馬場": going,
            "勝ち方": win_style,
        }

        details = []
        points = []
        hit_count = 0
        for cond, content in cond_values.items():
            if not content:
                details.append((cond, "", "", "", "", ""))
                continue
            found = lookup_condition(master, track, surface, dist_num, cond, content)
            if found is None:
                details.append((cond, content, "-", "-", "-", "-"))
                continue
            hit_count += 1
            points.append(rank_to_point(found["rank"]))
            details.append((
                cond, content, found["rank"],
                f'{found["win_rate"]:.1%}', f'{found["place_rate"]:.1%}', str(found["count"])
            ))

        avg_point = round(sum(points) / len(points), 2) if points else 0.0
        rank = final_rank(avg_point)
        comment = f"一致{hit_count}件 / 平均{avg_point}"
        out = r.to_dict()
        out["surface"] = surface
        out["distance_num"] = dist_num
        out["一致数"] = hit_count
        out["平均点"] = avg_point
        out["信頼度"] = rank
        out["分類"] = classify(rank)
        out["comment"] = comment
        out["detail_rows"] = details
        rows.append(out)
    return pd.DataFrame(rows)

def race_options(df):
    cols = ["date","track","raceNo","raceName"]
    sub = df[cols].drop_duplicates().fillna("")
    return [(f'{x["date"]} {x["track"]} {x["raceNo"]} {x["raceName"]}', x.to_dict()) for _, x in sub.iterrows()]

def filter_race(df, race_dict):
    out = df.copy()
    for col, val in race_dict.items():
        out = out[out[col].astype(str) == str(val)]
    return out.copy()

def render_preview(race_df, title):
    subtitle = ""
    if not race_df.empty:
        first = race_df.iloc[0]
        subtitle = f'{first.get("raceName","")} / {first.get("distance","")} / {len(race_df)}頭'
    html = [f'<div class="preview-panel"><div class="preview-title">{title}</div>']
    if subtitle:
        html.append(f'<div class="preview-sub">{subtitle}</div>')
    for _, row in race_df.iterrows():
        rank = str(row.get("信頼度","D"))
        cat = row.get("分類","")
        name = row.get("horseName","")
        html.append(
            f'<div class="preview-row"><div><div class="preview-name">{name}</div><div class="preview-class">{cat} / 一致{row.get("一致数",0)}件 / 平均{row.get("平均点",0)}</div></div><div class="rank-box rank-{rank}">{rank}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

def render_condition_table(race_df):
    rows = []
    for _, row in race_df.iterrows():
        detail_text = []
        for cond, content, rank, winr, placer, count in row.get("detail_rows", []):
            detail_text.append(f"{cond}:{content or '-'} / {rank} / 単{winr} / 複{placer} / 母数{count}")
        rows.append(
            f'<tr><td>{row.get("horseName","")}</td><td>{row.get("信頼度","")}</td><td>{row.get("分類","")}</td><td>{row.get("一致数","")}</td><td>{row.get("平均点","")}</td><td class="cond-cond">{"<br>".join(detail_text)}</td></tr>'
        )
    html = '<table class="cond-table"><thead><tr><th>馬名</th><th>信頼度</th><th>分類</th><th>一致数</th><th>平均点</th><th>条件詳細</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)

st.markdown('<div class="main-title">競馬 ランクアプリ<br>v8.0 Master Logic</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">競馬場×芝ダ×距離を土台に、血統・調教師・脚質・前走場所・馬場・勝ち方を同じ重みで評価します。</div>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
内部では各条件ごとに <b>単勝率 / 複勝率 / 母数</b> を保持します。<br>
最終評価は、各条件のランクを <b>同じ重み</b> で平均して作ります。<br>
使う条件は <b>血統 / 調教師 / 脚質 / 前走場所 / 馬場 / 勝ち方</b> の6本です。
</div>
""", unsafe_allow_html=True)

if not MASTER_PATH.exists() or not SUMMARY_PATH.exists():
    st.error("集計マスタCSVがありません。 app.py と同じ場所に master_logic.csv と master_summary.csv を置いてください。")
    st.stop()

master, summary = load_master()

st.markdown('<div class="section-card"><div class="section-title">集計マスタ状況</div></div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">マスタ行数</div><div class="metric-value">{len(master):,}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">舞台数</div><div class="metric-value">{len(summary):,}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">条件種類</div><div class="metric-value">{master["条件"].nunique()}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-card"><div class="section-title">予想レースCSV</div></div>', unsafe_allow_html=True)
pred_file = st.file_uploader("予想CSV", type=["csv"], label_visibility="collapsed")

if "evaluated_df" not in st.session_state:
    st.session_state["evaluated_df"] = None
if "preview_title" not in st.session_state:
    st.session_state["preview_title"] = "レースランキング"

race_label = None
if pred_file is not None:
    try:
        temp = pd.read_csv(pred_file)
        temp = normalize_columns(temp)
        opts = race_options(temp)
        labels = [x[0] for x in opts]
        if labels:
            race_label = st.selectbox("対象レース", labels)
        pred_file.seek(0)
    except Exception:
        st.selectbox("対象レース", ["CSV確認中"], disabled=True)
elif st.session_state["evaluated_df"] is not None:
    opts = race_options(st.session_state["evaluated_df"])
    labels = [x[0] for x in opts]
    if labels:
        race_label = st.selectbox("対象レース", labels)
else:
    st.selectbox("対象レース", ["先にCSVを読み込んでください"], disabled=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="green-btn">', unsafe_allow_html=True)
    run_eval = st.button("予想CSVを評価", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="dark-btn">', unsafe_allow_html=True)
    save_eval = st.button("評価結果CSVを保存", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if run_eval:
    if pred_file is None:
        st.error("予想CSVを選択してください。")
    else:
        try:
            pred = pd.read_csv(pred_file)
            pred = normalize_columns(pred)
            for col in PRED_REQUIRED:
                if col not in pred.columns:
                    raise ValueError(f"必須列不足: {col}")
            evaluated = evaluate_prediction(pred, master)
            st.session_state["evaluated_df"] = evaluated
            st.success(f"評価完了: {len(evaluated):,}頭 / {evaluated[['date','track','raceNo','raceName']].drop_duplicates().shape[0]}レース")
        except Exception as e:
            st.error(f"評価中にエラーが出ました: {e}")

show_df = st.session_state["evaluated_df"]
if show_df is not None:
    opts = race_options(show_df)
    race_map = dict(opts)
    if race_label and race_label in race_map:
        race_df = filter_race(show_df, race_map[race_label]).sort_values(["平均点","一致数","horseNo"], ascending=[False,False,True])
        st.session_state["preview_title"] = race_label
    else:
        race_df = show_df.head(0)
        if opts:
            first_label, first_dict = opts[0]
            race_df = filter_race(show_df, first_dict).sort_values(["平均点","一致数","horseNo"], ascending=[False,False,True])
            st.session_state["preview_title"] = first_label

    st.markdown('<div class="section-card"><div class="section-title">画像プレビュー</div></div>', unsafe_allow_html=True)
    render_preview(race_df, st.session_state["preview_title"])

    st.markdown('<div class="section-card"><div class="section-title">条件詳細</div></div>', unsafe_allow_html=True)
    render_condition_table(race_df)

    export_df = show_df.drop(columns=["detail_rows"]).copy()
    if save_eval:
        st.download_button(
            "評価結果CSVをダウンロード",
            data=export_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="評価結果_master_logic.csv",
            mime="text/csv",
            use_container_width=True
        )
else:
    st.markdown('<div class="preview-panel"><div class="preview-sub">予想CSVを読み込んで評価してください。</div></div>', unsafe_allow_html=True)
