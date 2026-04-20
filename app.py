
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="競馬 ランクアプリ v8.6 Ultra Compact", layout="wide")

APP_DIR = Path(__file__).resolve().parent
MASTER_PATH = APP_DIR / "master_logic.csv"
SUMMARY_PATH = APP_DIR / "master_summary.csv"

PRED_REQUIRED = ["date","track","raceNo","raceName","horseNo","horseName","jockey","trainer","sire","distance","going"]
CONDITION_WEIGHTS = {"血統":1.2,"調教師":1.2,"脚質":1.0,"前走場所":0.8,"馬場":0.8,"勝ち方":1.0}

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #071223 0%, #0a1730 100%); }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.block-container { max-width: 860px; padding-top: 0.55rem; padding-bottom: 1rem; }
.main-title { font-size: 1.45rem; font-weight: 800; color: #ffffff; line-height: 1.15; margin-bottom: 0.2rem; }
.sub-title { color: #dce9ff; font-size: 0.82rem; line-height: 1.45; margin-bottom: 0.5rem; }
.section-card { background: linear-gradient(180deg, rgba(10,20,40,.97) 0%, rgba(8,16,32,.97) 100%); border:1px solid rgba(122,154,214,.18); border-radius:16px; padding:.6rem .75rem; margin-bottom:.55rem; }
.section-title { color:#fff; font-size:1rem; font-weight:800; margin:0; }
[data-testid="stMarkdownContainer"] p, label[data-testid="stWidgetLabel"] { color:#f8fbff !important; font-weight:600 !important; }
[data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"] { background:#13233d !important; border:1px solid rgba(46,204,113,.28) !important; border-radius:12px !important; color:#fff !important; }
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] div, [data-testid="stFileUploader"] p { color:#fff !important; }
[data-baseweb="select"] > div { background:#13233d !important; color:#fff !important; border:1px solid rgba(216,92,92,.40) !important; border-radius:12px !important; min-height:2.5rem !important; }
[data-baseweb="select"] * { color:#fff !important; }
.stButton > button, .stDownloadButton > button {
    background:#1f3151 !important; color:#ffffff !important; border:1px solid rgba(255,255,255,.16) !important;
    border-radius:11px !important; padding:.56rem .65rem !important; font-weight:700 !important; font-size:.86rem !important;
}
.stButton > button:disabled, .stDownloadButton > button:disabled {
    background:#233654 !important; color:#dbe7ff !important; opacity:1 !important;
}
.ultra-wrap { background:#091426; border:1px solid rgba(126,156,214,.18); border-radius:18px; padding:10px 10px 8px 10px; }
.ultra-title { font-size:1.25rem; font-weight:800; color:#fff; line-height:1.08; margin-bottom:2px; }
.ultra-sub { color:#cfddff; font-size:.76rem; margin-bottom:7px; }
.ultra-row { display:grid; grid-template-columns: 1fr 54px; gap:6px; align-items:center; min-height:26px; padding:3px 6px; border-radius:10px; margin-bottom:3px; background:rgba(255,255,255,.025); }
.ultra-name { color:#fff; font-size:.82rem; font-weight:800; line-height:1.02; }
.ultra-meta { color:#cfddff; font-size:.58rem; margin-top:1px; line-height:1.0; }
.ultra-rank { text-align:center; border-radius:10px; padding:4px 0; font-weight:800; font-size:.82rem; color:#fff; }
.rank-S { background:#4c2fa8; } .rank-A { background:#1f8b58; } .rank-B { background:#2c6eb8; } .rank-C { background:#a97115; } .rank-D { background:#5a6578; }
.cond-table { width:100%; border-collapse:collapse; }
.cond-table th, .cond-table td { text-align:left; padding:8px 6px; border-bottom:1px solid rgba(255,255,255,.08); color:#f8fbff; vertical-align:top; font-size:.76rem; }
.cond-table th { color:#fff; font-size:.8rem; font-weight:700; }
.cond-cond { font-size:.68rem; color:#d7e5ff; line-height:1.25; word-break:break-word; }
@media (max-width: 480px) {
  .block-container { max-width: 420px; padding-top: .35rem; }
  .main-title { font-size: 1.28rem; }
  .sub-title { font-size: .74rem; margin-bottom:.35rem; }
  .ultra-title { font-size:1.08rem; }
  .ultra-sub { font-size:.68rem; }
  .ultra-row { min-height:23px; padding:2px 5px; margin-bottom:2px; }
  .ultra-name { font-size:.74rem; }
  .ultra-meta { font-size:.52rem; }
  .ultra-rank { font-size:.74rem; padding:3px 0; }
}
</style>
""", unsafe_allow_html=True)

if "evaluated_df" not in st.session_state:
    st.session_state["evaluated_df"] = None
if "preview_title" not in st.session_state:
    st.session_state["preview_title"] = "レースランキング"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "date":"date","track":"track","raceNo":"raceNo","race_name":"raceName","raceName":"raceName",
        "horseNo":"horseNo","horse_no":"horseNo","horseName":"horseName","horse_name":"horseName",
        "jockey":"jockey","trainer":"trainer","sire":"sire","damSire":"damSire","distance":"distance",
        "going":"going","fieldSize":"fieldSize","style":"style","prevTrack":"prevTrack",
        "prev_track":"prevTrack","winStyle":"winStyle","win_style":"winStyle",
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
    return "", "".join(ch for ch in s if ch.isdigit())

def style_tag(v):
    s = str(v).strip()
    if s in ("逃げ",): return "逃げ"
    if s in ("先行",): return "先行"
    if s in ("差し", "中団", "後方"): return "差し"
    if s in ("追込",): return "追込"
    if "マクリ" in s or "まくり" in s or "ﾏｸﾘ" in s: return "まくり"
    return s if s and s != "nan" else ""

def winstyle_tag(v):
    s = str(v).strip()
    if s in ("逃げ",): return "逃げ切り"
    if s in ("先行",): return "先行押し切り"
    if s in ("差し", "中団"): return "差し"
    if s in ("後方", "追込"): return "追い込み"
    if "マクリ" in s or "まくり" in s or "ﾏｸﾘ" in s: return "まくり"
    return s if s and s != "nan" else ""

def going_tag(v):
    s = str(v).strip()
    return s if s and s != "nan" else ""

@st.cache_data
def load_master():
    return (
        pd.read_csv(MASTER_PATH, encoding="utf-8-sig"),
        pd.read_csv(SUMMARY_PATH, encoding="utf-8-sig")
    )

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
    row = m.sort_values(["母数", "単勝率", "複勝率"], ascending=[False, False, False]).iloc[0]
    return {"rank": str(row["ランク"]), "count": int(row["母数"]), "win_rate": float(row["単勝率"]), "place_rate": float(row["複勝率"])}

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

def evaluate_prediction(pred_df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in pred_df.iterrows():
        surface, dist_num = parse_distance(r.get("distance", ""))
        track = r.get("track", "")
        cond_values = {
            "血統": r.get("sire", ""),
            "調教師": r.get("trainer", ""),
            "脚質": style_tag(r.get("style", "")),
            "前走場所": r.get("prevTrack", ""),
            "馬場": going_tag(r.get("going", "")),
            "勝ち方": winstyle_tag(r.get("winStyle", "")),
        }
        details = []
        weighted_points = []
        raw_points = []
        hit_count = 0
        valid_weight_sum = 0.0
        for cond, content in cond_values.items():
            weight = CONDITION_WEIGHTS.get(cond, 1.0)
            if not content:
                details.append((cond, "", "", "", "", "", weight, ""))
                continue
            found = lookup_condition(master, track, surface, dist_num, cond, content)
            if found is None:
                details.append((cond, content, "-", "-", "-", "-", weight, ""))
                continue
            hit_count += 1
            point = rank_to_point(found["rank"])
            raw_points.append(point)
            weighted_points.append(point * weight)
            valid_weight_sum += weight
            details.append((cond, content, found["rank"], f'{found["win_rate"]:.1%}', f'{found["place_rate"]:.1%}', str(found["count"]), weight, round(point * weight, 2)))
        avg_point = round(sum(raw_points) / len(raw_points), 2) if raw_points else 0.0
        weighted_avg = round(sum(weighted_points) / valid_weight_sum, 2) if valid_weight_sum else 0.0
        rank = final_rank(weighted_avg)
        out = r.to_dict()
        out["surface"] = surface
        out["distance_num"] = dist_num
        out["一致数"] = hit_count
        out["平均点"] = avg_point
        out["加重点"] = weighted_avg
        out["信頼度"] = rank
        out["分類"] = classify(rank)
        out["comment"] = f"一致{hit_count}件 / 平均{avg_point} / 加重{weighted_avg}"
        out["detail_rows"] = details
        rows.append(out)
    return pd.DataFrame(rows)

def race_options(df: pd.DataFrame):
    cols = ["date","track","raceNo","raceName"]
    sub = df[cols].drop_duplicates().fillna("")
    return [(f'{x["date"]} {x["track"]} {x["raceNo"]} {x["raceName"]}', x.to_dict()) for _, x in sub.iterrows()]

def filter_race(df: pd.DataFrame, race_dict):
    out = df.copy()
    for col, val in race_dict.items():
        out = out[out[col].astype(str) == str(val)]
    return out.copy()

def render_ultra_preview(race_df: pd.DataFrame, title: str):
    subtitle = ""
    if not race_df.empty:
        first = race_df.iloc[0]
        subtitle = f'{first.get("raceName","")} / {first.get("distance","")} / {len(race_df)}頭'
    html = [f'<div class="ultra-wrap"><div class="ultra-title">{title}</div>']
    if subtitle:
        html.append(f'<div class="ultra-sub">{subtitle}</div>')
    for _, row in race_df.iterrows():
        rank = str(row.get("信頼度","D"))
        cat = row.get("分類","")
        name = row.get("horseName","")
        meta = f'{cat} / 一致{row.get("一致数",0)} / 加重{row.get("加重点",0)}'
        html.append(
            f'<div class="ultra-row"><div><div class="ultra-name">{name}</div><div class="ultra-meta">{meta}</div></div>'
            f'<div class="ultra-rank rank-{rank}">{rank}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

def render_condition_table(race_df: pd.DataFrame):
    rows = []
    for _, row in race_df.iterrows():
        detail_text = []
        for cond, content, rank, winr, placer, count, weight, score in row.get("detail_rows", []):
            detail_text.append(f"{cond}:{content or '-'} / {rank} / 単{winr} / 複{placer} / 母数{count} / 重み{weight} / 加点{score}")
        rows.append(
            f'<tr><td>{row.get("horseName","")}</td><td>{row.get("信頼度","")}</td><td>{row.get("分類","")}</td>'
            f'<td>{row.get("一致数","")}</td><td>{row.get("平均点","")}</td><td>{row.get("加重点","")}</td>'
            f'<td class="cond-cond">{"<br>".join(detail_text)}</td></tr>'
        )
    html = '<table class="cond-table"><thead><tr><th>馬名</th><th>信頼度</th><th>分類</th><th>一致数</th><th>平均点</th><th>加重点</th><th>条件詳細</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    st.markdown(html, unsafe_allow_html=True)

st.markdown('<div class="main-title">競馬 ランクアプリ v8.6</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">超コンパクト版。評価後は下の一覧をそのままスクショしてください。</div>', unsafe_allow_html=True)

if not MASTER_PATH.exists() or not SUMMARY_PATH.exists():
    st.error("master_logic.csv と master_summary.csv を app.py と同じ場所に置いてください。")
    st.stop()

master, summary = load_master()

with st.expander("入力と操作", expanded=True):
    pred_file = st.file_uploader("予想CSV", type=["csv"], label_visibility="visible")

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
        run_eval = st.button("予想CSVを評価", use_container_width=True)
    with c2:
        save_eval = st.button("評価結果CSVを保存", use_container_width=True)

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
                n_race = evaluated[["date","track","raceNo","raceName"]].drop_duplicates().shape[0]
                st.success(f"評価完了: {len(evaluated):,}頭 / {n_race}レース")
            except Exception as e:
                st.error(f"評価中にエラーが出ました: {e}")

show_df = st.session_state["evaluated_df"]
if show_df is not None:
    opts = race_options(show_df)
    race_map = dict(opts)
    race_label_current = None
    if 'race_label' in locals() and race_label and race_label in race_map:
        race_label_current = race_label
        race_df = filter_race(show_df, race_map[race_label]).sort_values(["加重点","一致数","horseNo"], ascending=[False, False, True])
    else:
        first_label, first_dict = opts[0]
        race_label_current = first_label
        race_df = filter_race(show_df, first_dict).sort_values(["加重点","一致数","horseNo"], ascending=[False, False, True])

    render_ultra_preview(race_df, race_label_current)

    if save_eval:
        export_df = show_df.drop(columns=["detail_rows"]).copy()
        st.download_button(
            "評価結果CSVをダウンロード",
            data=export_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name="評価結果_weighted_logic.csv",
            mime="text/csv",
            use_container_width=True
        )

    with st.expander("条件詳細を見る", expanded=False):
        render_condition_table(race_df)
else:
    st.markdown('<div class="ultra-wrap"><div class="ultra-sub">予想CSVを読み込んで評価してください。</div></div>', unsafe_allow_html=True)
