
import os
import re
import io
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="競馬ランクアプリ v12.1 SNS Save", layout="centered")

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
DEFAULT_FILES = {
    "prev3c": os.path.join(BASE_DIR, "prev3c_category_stats.csv"),
    "prev4c": os.path.join(BASE_DIR, "prev4c_category_stats.csv"),
    "prevtrack": os.path.join(BASE_DIR, "prevtrack_roi_stats.csv"),
}

EVAL_MAP = {"かなり向く":1.25, "向く":1.10, "普通":1.00, "やや不向き":0.85, "不向き":0.70}

def norm_text(v):
    if pd.isna(v):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(v)).strip().split())

def norm_track(v):
    s = norm_text(v)
    mapping = {
        "東京競馬場":"東京","中山競馬場":"中山","中京競馬場":"中京","阪神競馬場":"阪神",
        "京都競馬場":"京都","新潟競馬場":"新潟","福島競馬場":"福島","小倉競馬場":"小倉",
        "札幌競馬場":"札幌","函館競馬場":"函館",
    }
    return mapping.get(s, s)

def norm_surface(v):
    s = norm_text(v)
    if s.startswith("芝"):
        return "芝"
    if s.startswith("ダ") or s == "ダート":
        return "ダ"
    return s

def read_csv_any(file_or_path):
    last = None
    for enc in ["utf-8-sig", "cp932", "shift_jis", "utf-8"]:
        try:
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception as e:
            last = e
    raise last

def parse_race_label(v):
    s = norm_text(v)
    m = re.search(r"(東京|中山|中京|阪神|京都|新潟|福島|小倉|札幌|函館)\s*(\d+)\s*R", s)
    if not m:
        return "", np.nan
    return m.group(1), float(m.group(2))

CANDS = {
    "date":["date","日付","開催日","年月日","日付S"],
    "場所":["場所","track","競馬場","開催","場名"],
    "raceNo":["raceNo","race_number","R","レース番号","race_no","レースNo","R番号"],
    "raceName":["raceName","race_name","レース名"],
    "horseNo":["horseNo","horse_number","馬番"],
    "horseName":["horseName","horse_name","馬名"],
    "distance":["distance","距離"],
    "surface":["芝ダ","芝・ダ","surface"],
    "raceLabel":["レース","race"],
    "prevTrack":["前走競馬場","前走場所","prevTrack"],
    "prevSurface":["前走芝ダ","prevSurface"],
    "prevDistance":["前走距離数値","前走距離","prevDistance"],
    "prev3cCat":["前3角位置カテゴリ","前走3角カテゴリ","prev3cCat"],
    "prev4cCat":["前4角位置カテゴリ","前走4角カテゴリ","prev4cCat"],
    "prevStraight":["前走直線ロジック点","前走直線点","prevStraight"],
    "prev2Straight":["前々走直線ロジック点","前々走直線点","prev2Straight"],
    "paceEval":["展開予想評価","脚質展開評価","paceEval"],
    "straightEval":["直線相性評価","場所直線相性評価","straightEval"],
}

def rename_first_match(df, candidates):
    out = df.copy()
    normalized = {c: norm_text(c) for c in out.columns}
    for target, opts in candidates.items():
        if target in out.columns:
            continue
        found = None
        for o in opts:
            for c, nc in normalized.items():
                if nc == o:
                    found = c
                    break
            if found:
                break
        if found:
            out = out.rename(columns={found: target})
    return out

def prepare_race_df(df):
    df = rename_first_match(df, CANDS)
    for col in CANDS.keys():
        if col not in df.columns:
            df[col] = ""

    parsed = df["raceLabel"].apply(parse_race_label)
    df["場所"] = np.where(df["場所"].astype(str).str.strip() != "", df["場所"], parsed.apply(lambda x: x[0]))
    df["raceNo"] = np.where(df["raceNo"].astype(str).str.strip() != "", df["raceNo"], parsed.apply(lambda x: x[1]))

    for col in ["場所","prevTrack"]:
        df[col] = df[col].apply(norm_track)
    for col in ["surface","prevSurface"]:
        df[col] = df[col].apply(norm_surface)
    for col in ["horseName","raceName","date","prev3cCat","prev4cCat","paceEval","straightEval"]:
        df[col] = df[col].apply(norm_text)
    for col in ["distance","raceNo","horseNo","prevDistance","prevStraight","prev2Straight"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["prevStraight"] = df["prevStraight"].fillna(50.0).clip(0, 100)
    df["prev2Straight"] = df["prev2Straight"].fillna(50.0).clip(0, 100)
    df["距離表示"] = np.where(df["surface"].astype(str) != "", df["surface"] + df["distance"].fillna(0).astype(int).astype(str), "")
    df["レース"] = df.apply(lambda r: f"{r['場所']}{int(r['raceNo'])}R" if pd.notna(r["raceNo"]) and norm_text(r["場所"]) else norm_text(r["raceLabel"]), axis=1)

    # 最重要: 日付 + 場所 + R で識別。日付 + R だけで混ぜない。
    df["レース識別ID"] = df.apply(
        lambda r: f"{r['date']}_{r['場所']}_{int(r['raceNo'])}R" if pd.notna(r["raceNo"]) else f"{r['date']}_{r['場所']}_{r['レース']}",
        axis=1
    )
    return df

def load_stat_defaults():
    prev3c = read_csv_any(DEFAULT_FILES["prev3c"])
    prev4c = read_csv_any(DEFAULT_FILES["prev4c"])
    prevtrack = read_csv_any(DEFAULT_FILES["prevtrack"])

    prev3c = prev3c.rename(columns={"前走競馬場":"prevTrack","前走芝ダ":"prevSurface","前走距離数値":"prevDistance","前3角位置カテゴリ":"prev3cCat","件数":"count","複勝率":"place_rate"})
    prev4c = prev4c.rename(columns={"前走競馬場":"prevTrack","前走芝ダ":"prevSurface","前走距離数値":"prevDistance","前4角位置カテゴリ":"prev4cCat","件数":"count","複勝率":"place_rate"})
    prevtrack = prevtrack.rename(columns={"競馬場":"場所","芝ダ":"surface","距離数値":"distance","前走場所":"prevTrack","件数":"count","複勝率":"place_rate"})

    for d in [prev3c, prev4c]:
        d["prevTrack"] = d["prevTrack"].apply(norm_track)
        d["prevSurface"] = d["prevSurface"].apply(norm_surface)
        d["prevDistance"] = pd.to_numeric(d["prevDistance"], errors="coerce")
        d["count"] = pd.to_numeric(d["count"], errors="coerce")
        d["place_rate"] = pd.to_numeric(d["place_rate"], errors="coerce")
    prev3c["prev3cCat"] = prev3c["prev3cCat"].apply(norm_text)
    prev4c["prev4cCat"] = prev4c["prev4cCat"].apply(norm_text)

    prevtrack["場所"] = prevtrack["場所"].apply(norm_track)
    prevtrack["surface"] = prevtrack["surface"].apply(norm_surface)
    prevtrack["distance"] = pd.to_numeric(prevtrack["distance"], errors="coerce")
    prevtrack["prevTrack"] = prevtrack["prevTrack"].apply(norm_track)
    prevtrack["count"] = pd.to_numeric(prevtrack["count"], errors="coerce")
    prevtrack["place_rate"] = pd.to_numeric(prevtrack["place_rate"], errors="coerce")
    return prev3c, prev4c, prevtrack

def map_rate_to_coef(rate, min_rate, max_rate):
    if pd.isna(rate):
        return 1.0
    if pd.isna(min_rate) or pd.isna(max_rate) or max_rate <= min_rate:
        return 1.0
    q = (rate - min_rate) / (max_rate - min_rate)
    if q >= 0.80:
        return 1.25
    if q >= 0.60:
        return 1.10
    if q >= 0.40:
        return 1.00
    if q >= 0.20:
        return 0.85
    return 0.70

def hist_coef_prev3c(row, stat):
    sub = stat[(stat["prevTrack"] == row["prevTrack"]) & (stat["prevSurface"] == row["prevSurface"]) & (stat["prevDistance"] == row["prevDistance"])]
    if sub.empty:
        return 1.0
    hit = sub[sub["prev3cCat"] == row["prev3cCat"]]
    rate = hit["place_rate"].iloc[0] if not hit.empty else np.nan
    return map_rate_to_coef(rate, sub["place_rate"].min(), sub["place_rate"].max())

def hist_coef_prev4c(row, stat):
    sub = stat[(stat["prevTrack"] == row["prevTrack"]) & (stat["prevSurface"] == row["prevSurface"]) & (stat["prevDistance"] == row["prevDistance"])]
    if sub.empty:
        return 1.0
    hit = sub[sub["prev4cCat"] == row["prev4cCat"]]
    rate = hit["place_rate"].iloc[0] if not hit.empty else np.nan
    return map_rate_to_coef(rate, sub["place_rate"].min(), sub["place_rate"].max())

def hist_coef_prevtrack(row, stat):
    sub = stat[(stat["場所"] == row["場所"]) & (stat["surface"] == row["surface"]) & (stat["distance"] == row["distance"])]
    if sub.empty:
        return 1.0
    hit = sub[sub["prevTrack"] == row["prevTrack"]]
    rate = hit["place_rate"].iloc[0] if not hit.empty else np.nan
    return map_rate_to_coef(rate, sub["place_rate"].min(), sub["place_rate"].max())

def confidence_from_score(score):
    return float(np.clip(20 + score * 1.2, 5, 95))

def total_rank(score):
    if pd.isna(score):
        return ""
    if score >= 65:
        return "S"
    if score >= 56:
        return "A"
    if score >= 47:
        return "B"
    if score >= 38:
        return "C"
    return "D"

def assign_relative_ranks(df):
    out = df.copy()
    out["相対評価"] = ""
    for race_id in out["レース識別ID"].unique():
        idx = out[out["レース識別ID"] == race_id].sort_values(["総合点","horseNo"], ascending=[False, True]).index.tolist()
        n = len(idx)
        if n == 0:
            continue
        s_n = max(1, round(n * 0.10))
        a_n = max(1, round(n * 0.20))
        b_n = max(1, round(n * 0.25))
        tail_n = max(1, round(n * 0.15))
        for i, ix in enumerate(idx):
            if i < s_n:
                out.at[ix, "相対評価"] = "S"
            elif i < s_n + a_n:
                out.at[ix, "相対評価"] = "A"
            elif i < s_n + a_n + b_n:
                out.at[ix, "相対評価"] = "B"
            elif i < n - tail_n:
                out.at[ix, "相対評価"] = "C"
            else:
                out.at[ix, "相対評価"] = "D"
    return out

def recommend_for_race(g):
    g = g.sort_values(["総合点","horseNo"], ascending=[False, True]).reset_index(drop=True)
    single = g.iloc[0]
    pair = "見送り"
    trio = "見送り"
    use = g[(g["トータルランク"].isin(["S","A"])) & (g["相対評価"].isin(["S","A","B"]))]
    if len(use) >= 2:
        pair = " / ".join([f'{int(single["horseNo"])}-{int(r["horseNo"])}' for _, r in use.iloc[1:4].iterrows()])
    if len(use) >= 4:
        a, b, c = use.iloc[1], use.iloc[2], use.iloc[3]
        trio = f'{int(single["horseNo"])}-{int(a["horseNo"])}-{int(b["horseNo"])} / {int(single["horseNo"])}-{int(a["horseNo"])}-{int(c["horseNo"])} / {int(single["horseNo"])}-{int(b["horseNo"])}-{int(c["horseNo"])}'
    short_comment = f'本体{single["本体点"]:.1f}×展開{single["展開位置補正"]:.2f}×場所{single["前走場所直線補正"]:.2f}'
    return single, pair, trio, short_comment


def render_rank_cards(g):
    badge_map = {"S": "#d4af37", "A": "#d6deef", "B": "#c7a85a", "C": "#7b8db7", "D": "#53627f"}
    rows = []
    for _, r in g.iterrows():
        rank_color = badge_map.get(r["トータルランク"], "#7b8db7")
        rows.append(f"""
        <div class="horse-row">
          <div class="horse-left">
            <div class="horse-no">{int(r["horseNo"])}</div>
            <div class="horse-name">{r["horseName"]}</div>
          </div>
          <div class="rank-wrap">
            <div class="rank-label">ランク</div>
            <div class="rank-box" style="border-color:{rank_color};">{r["トータルランク"]}</div>
          </div>
        </div>
        """)

    html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        html, body {{
          margin:0;
          padding:0;
          background:transparent;
          font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .card {{
          background:#061734;
          border-radius:28px;
          padding:20px 18px 16px 18px;
          box-shadow:0 8px 24px rgba(0,0,0,0.18);
          color:white;
          box-sizing:border-box;
          width:100%;
        }}
        .title {{
          font-size:23px;
          font-weight:900;
          color:white;
          line-height:1.2;
        }}
        .subtitle {{
          font-size:15px;
          color:#c3d0e8;
          margin-top:6px;
        }}
        .rows {{
          margin-top:14px;
        }}
        .horse-row {{
          display:flex;
          align-items:center;
          justify-content:space-between;
          gap:12px;
          background:rgba(255,255,255,0.03);
          border:1px solid rgba(255,255,255,0.06);
          border-radius:20px;
          padding:12px 14px;
          margin:8px 0;
          box-sizing:border-box;
        }}
        .horse-left {{
          display:flex;
          align-items:center;
          gap:10px;
          min-width:0;
          flex:1;
        }}
        .horse-no {{
          font-size:15px;
          color:#aebee0;
          font-weight:700;
          min-width:20px;
          text-align:center;
        }}
        .horse-name {{
          font-size:24px;
          color:white;
          font-weight:850;
          line-height:1.05;
          white-space:nowrap;
          overflow:hidden;
          text-overflow:ellipsis;
        }}
        .rank-wrap {{
          display:flex;
          flex-direction:column;
          align-items:center;
          gap:4px;
          flex-shrink:0;
        }}
        .rank-label {{
          font-size:12px;
          color:#b9c7e8;
        }}
        .rank-box {{
          width:50px;
          height:50px;
          border-radius:15px;
          border:3px solid #7b8db7;
          display:flex;
          align-items:center;
          justify-content:center;
          color:white;
          font-weight:900;
          font-size:22px;
          box-sizing:border-box;
        }}
      </style>
    </head>
    <body>
      <div class="card">
        <div class="title">{g.iloc[0]["date"]} {g.iloc[0]["レース"]}</div>
        <div class="subtitle">{g.iloc[0]["raceName"]} / {g.iloc[0]["距離表示"]}</div>
        <div class="rows">{''.join(rows)}</div>
      </div>
    </body>
    </html>
    """
    height = 135 + len(g) * 86
    components.html(html, height=height, scrolling=False)


def get_font(size, bold=False):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def draw_fit_text(draw, xy, text, font, fill, max_width):
    x, y = xy
    t = str(text)
    while len(t) > 0:
        bbox = draw.textbbox((x,y), t, font=font)
        if bbox[2] - bbox[0] <= max_width:
            break
        t = t[:-1]
    if t != str(text):
        t = t[:-1] + "…"
    draw.text((x,y), t, font=font, fill=fill)

def make_sns_image(saved):
    items = [r for r in saved if float(r.get("参考信頼度", 0)) >= 90.0]
    if not items:
        return None
    items = sorted(items, key=lambda r: (str(r["日付"]), str(r["場所"]), int(r["R"])))
    date = items[0]["日付"]

    W = 1080
    H = max(1350, 260 + len(items) * 118 + 120)
    img = Image.new("RGB", (W, H), (5, 16, 42))
    draw = ImageDraw.Draw(img)

    # gradient-ish bands
    for y in range(H):
        r = 5 + int(8 * y / H)
        g = 16 + int(12 * y / H)
        b = 42 + int(28 * y / H)
        draw.line([(0,y),(W,y)], fill=(r,g,b))

    title_font = get_font(68, True)
    sub_font = get_font(26, False)
    race_font = get_font(34, True)
    horse_font = get_font(42, True)
    small_font = get_font(24, False)
    note_font = get_font(22, False)

    draw.text((70, 66), f"{date} 推奨馬", font=title_font, fill=(255,255,255))
    draw.rounded_rectangle((70, 152, 1010, 158), radius=3, fill=(214,175,55))

    y = 215
    for r in items:
        draw.rounded_rectangle((62, y, 1018, y+92), radius=28, fill=(10, 31, 72), outline=(47,70,120), width=2)
        label = f'{r["場所"]}{int(r["R"])}R'
        draw.text((92, y+26), label, font=race_font, fill=(234, 214, 137))
        draw.text((280, y+24), "🔥", font=horse_font, fill=(255, 90, 60))
        draw.rounded_rectangle((352, y+18, 422, y+74), radius=16, fill=(214,175,55))
        draw.text((374, y+26), str(int(r["馬番"])), font=race_font, fill=(5, 16, 42))
        draw_fit_text(draw, (446, y+22), r["馬名"], horse_font, (255,255,255), 390)
        conf = float(r.get("参考信頼度", 0))
        draw.text((860, y+31), f"{conf:.1f}%", font=small_font, fill=(210,225,255))
        y += 112

    draw.text((70, H-72), "信頼度90%以上のみ掲載", font=note_font, fill=(185,196,220))
    draw.text((70, H-42), "※保存済みの単複おすすめ1より作成", font=note_font, fill=(150,162,190))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio

def add_saved_recs(new_recs):
    if "saved_recs" not in st.session_state:
        st.session_state.saved_recs = []
    # key = 日付 + 場所 + R, update existing
    store = {f'{r["日付"]}_{r["場所"]}_{r["R"]}': r for r in st.session_state.saved_recs}
    for r in new_recs:
        store[f'{r["日付"]}_{r["場所"]}_{r["R"]}'] = r
    st.session_state.saved_recs = list(store.values())

def saved_df():
    if "saved_recs" not in st.session_state:
        st.session_state.saved_recs = []
    return pd.DataFrame(st.session_state.saved_recs)

st.title("競馬ランクアプリ v12.1 SNS Save")
st.write("ランキング計算は1会場ずつ安全に行い、単複おすすめ1だけを保存して、最後に3会場まとめSNS画像を作成します。")

if "saved_recs" not in st.session_state:
    st.session_state.saved_recs = []

uploaded = st.file_uploader("1会場分の予想CSVをアップロード", type=["csv"])

current_recs = []
if uploaded is None:
    st.info("まず1会場6レース分のCSVを読み込んでください。レース識別IDは 日付 + 場所 + R で作成します。")
else:
    prev3c_stat, prev4c_stat, prevtrack_stat = load_stat_defaults()
    df = prepare_race_df(read_csv_any(uploaded))

    df["本体点"] = (df["prevStraight"] * 0.30 + df["prev2Straight"] * 0.20).round(2)
    df["3角履歴係数"] = df.apply(lambda r: hist_coef_prev3c(r, prev3c_stat), axis=1)
    df["4角履歴係数"] = df.apply(lambda r: hist_coef_prev4c(r, prev4c_stat), axis=1)
    df["展開履歴係数"] = ((df["3角履歴係数"] + df["4角履歴係数"]) / 2).round(2)
    df["展開予想係数"] = df["paceEval"].map(EVAL_MAP).fillna(1.0)
    df["展開位置補正"] = ((df["展開履歴係数"] + df["展開予想係数"]) / 2).round(2)
    df["前走場所履歴係数"] = df.apply(lambda r: hist_coef_prevtrack(r, prevtrack_stat), axis=1)
    df["直線相性係数"] = df["straightEval"].map(EVAL_MAP).fillna(1.0)
    df["前走場所直線補正"] = ((df["前走場所履歴係数"] + df["直線相性係数"]) / 2).round(2)
    df["総合点"] = (df["本体点"] * df["展開位置補正"] * df["前走場所直線補正"]).round(2)
    df["トータルランク"] = df["総合点"].apply(total_rank)
    df = assign_relative_ranks(df)

    tab1, tab2, tab3 = st.tabs(["ランキング", "おすすめ買い目", "保存・SNS画像"])

    with tab1:
        for race_id in df["レース識別ID"].unique():
            g = df[df["レース識別ID"] == race_id].sort_values(["総合点","horseNo"], ascending=[False, True]).reset_index(drop=True)
            render_rank_cards(g)
            st.divider()

    with tab2:
        for race_id in df["レース識別ID"].unique():
            g = df[df["レース識別ID"] == race_id].sort_values(["総合点","horseNo"], ascending=[False, True]).reset_index(drop=True)
            st.subheader(f'{g.iloc[0]["date"]} {g.iloc[0]["レース"]} {g.iloc[0]["raceName"]}')
            single, pair, trio, comment = recommend_for_race(g)
            conf = confidence_from_score(single["総合点"])
            st.markdown("### 単複おすすめ1")
            st.write(f'候補: {int(single["horseNo"])} {single["horseName"]}')
            st.caption(f'トータル{single["トータルランク"]} / 総合点 {single["総合点"]:.2f} / 参考信頼度 {conf:.2f}% / {comment}')
            st.markdown("### 馬連おすすめ1")
            st.write(pair)
            st.markdown("### 三連複おすすめ1")
            st.write(trio)
            st.divider()

            current_recs.append({
                "日付": single["date"],
                "場所": single["場所"],
                "R": int(single["raceNo"]),
                "馬番": int(single["horseNo"]),
                "馬名": single["horseName"],
                "単複おすすめ1": f'{int(single["horseNo"])} {single["horseName"]}',
                "参考信頼度": round(float(conf), 2),
                "短評": comment,
            })

    with tab3:
        st.subheader("この会場の推奨馬")
        if current_recs:
            st.dataframe(pd.DataFrame(current_recs), use_container_width=True, hide_index=True)
        if st.button("この会場の推奨馬を保存", type="primary"):
            add_saved_recs(current_recs)
            st.success("この会場の単複おすすめ1を保存しました。")

        st.subheader("保存済み推奨馬")
        sdf = saved_df()
        if sdf.empty:
            st.info("まだ保存済み推奨馬はありません。")
        else:
            st.dataframe(sdf.sort_values(["日付","場所","R"]), use_container_width=True, hide_index=True)
            csv = sdf.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("保存済み推奨馬CSVをダウンロード", data=csv.encode("utf-8-sig"), file_name="saved_recommendations.csv", mime="text/csv")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("3会場まとめSNS画像を作成"):
                img = make_sns_image(st.session_state.saved_recs)
                if img is None:
                    st.warning("信頼度90%以上の推奨馬はありません")
                else:
                    st.image(img, caption="SNS投稿用画像", use_container_width=True)
                    st.download_button("SNS画像PNGをダウンロード", data=img.getvalue(), file_name="sns_recommendations.png", mime="image/png")
        with col2:
            if st.button("保存済み推奨馬をクリア"):
                st.session_state.saved_recs = []
                st.success("保存済み推奨馬をクリアしました。")
