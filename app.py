
import os
import re
import io
import unicodedata
import time
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

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


def get_gemini_api_key():
    """アプリ入力欄 / Streamlit Secrets / 環境変数からGemini APIキーを読む"""
    input_key = st.session_state.get("GEMINI_API_KEY_INPUT", "")
    if input_key:
        return input_key

    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.environ.get("GEMINI_API_KEY", "")

    return key

def strip_code_fence(text_value):
    s = str(text_value or "").strip()
    s = re.sub(r"^```(?:csv|text|python)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def build_gemini_prediction_prompt(date, place, race_no, use_day_bias=True):
    bias_text = "当日の馬場状態・トラックバイアス・風・当日ここまでの傾向も検索して反映してください。" if use_day_bias else "前日版として公開情報中心で作成してください。"
    return f"""
あなたは競馬予想データ作成AIです。
画像は使わず、ネット検索で取得できる最新の公開情報だけを使って、
指定レースの出走馬CSVを作成してください。

【対象レース】
日付: {date}
場所: {place}
R: {race_no}R

【自動取得すること】
以下はユーザーが入力しないため、必ずネット検索で自動取得してください。
・レース名
・芝/ダ
・距離
・出走馬
・馬番
・前走/前々走情報
・前走3角/4角通過順
・当日の馬場状態、TB、風、当日ここまでの傾向

【今回の目的】
このCSVは競馬ランクアプリに直接読み込ませます。
必ず以下の列だけを、1行目ヘッダーとしてCSV形式で出力してください。
説明文、補足、コードブロックは禁止です。

【出力列】
date,場所,raceNo,raceName,horseNo,horseName,distance,surface,prevTrack,prevSurface,prevDistance,prev3cCat,prev4cCat,prevStraight,prev2Straight,paceEval,straightEval

【列の意味】
date: YYYY-MM-DD
場所: 東京/中山/京都/阪神/福島など
raceNo: 数字のみ
raceName: ネット検索で取得したレース名
horseNo: 馬番
horseName: 馬名
distance: ネット検索で取得した距離数値のみ
surface: ネット検索で取得した 芝 または ダ
prevTrack: 前走競馬場
prevSurface: 前走の芝/ダ
prevDistance: 前走距離数値
prev3cCat: 前走3角位置カテゴリ。候補は「1番手」「2-3番手」「4-6番手」「7-10番手」「11番手以下」
prev4cCat: 前走4角位置カテゴリ。候補は「1番手」「2-3番手」「4-6番手」「7-10番手」「11番手以下」
prevStraight: 前走直線ロジック点。0〜100の数値
prev2Straight: 前々走直線ロジック点。0〜100の数値
paceEval: 展開予想評価。候補は「かなり向く」「向く」「普通」「やや不向き」「不向き」
straightEval: 直線相性評価。候補は「かなり向く」「向く」「普通」「やや不向き」「不向き」

【重視】
・レース名、芝/ダ、距離を必ず自動取得すること
・出走馬、馬番、馬名を正確に取得すること
・前走/前々走の内容から直線ロジック点を付けること
・前走3角/4角通過順をカテゴリ化すること
・{bias_text}
・オッズ、人気による補正はしないこと
・推測で雑に埋めないこと
・ただしアプリ投入のため主要列は最大限埋めること

【最終出力】
CSV本文のみ。
"""

def call_gemini_with_search(prompt, model_name="gemini-2.5-flash-lite", use_search=True, retry_no_search=True):
    """Gemini API。429対策として検索なしフォールバックも可能。"""
    if genai is None or types is None:
        raise RuntimeError("google-genai が読み込めません。requirements.txt に google-genai が入っているか確認してください。")

    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY が設定されていません。アプリ内入力欄、またはStreamlit Secretsに設定してください。")

    client = genai.Client(api_key=api_key)

    def _generate(search_enabled):
        if search_enabled:
            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.2,
            )
        else:
            config = types.GenerateContentConfig(
                temperature=0.2,
            )
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    try:
        return _generate(use_search)
    except Exception as e:
        msg = str(e)
        # 429は無料枠/レート制限/検索グラウンディング制限の可能性が高い。
        if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()) and retry_no_search and use_search:
            time.sleep(3)
            try:
                st.warning("Geminiの検索付き実行が429制限に達しました。検索なしで1回だけ再試行します。")
                return _generate(False)
            except Exception as e2:
                raise RuntimeError(
                    "Gemini APIが429 RESOURCE_EXHAUSTEDになりました。"
                    "無料枠・レート制限・検索グラウンディングの上限に達している可能性があります。"
                    "時間を置く、検索なしにする、modelをflash-liteにする、または課金/クォータ確認をしてください。\n\n"
                    f"検索付きエラー: {msg}\n\n検索なし再試行エラー: {str(e2)}"
                )
        raise

def gemini_csv_to_df(csv_text):
    cleaned = strip_code_fence(csv_text)
    if not cleaned:
        raise ValueError("GeminiからCSV本文が返りませんでした。")
    return pd.read_csv(io.StringIO(cleaned))


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





def _rank_score(v):
    return {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}.get(str(v), 0)

def _eval_score(v):
    return {"かなり向く": 5, "向く": 4, "普通": 3, "やや不向き": 2, "不向き": 1}.get(norm_text(v), 3)

def _position_score(v):
    s = norm_text(v)
    if "1" in s and "番手" in s:
        return 1
    if "2-3" in s or "2~3" in s or "2〜3" in s:
        return 2
    if "4-6" in s or "4~6" in s or "4〜6" in s:
        return 3
    if "7-10" in s or "7~10" in s or "7〜10" in s:
        return 4
    if "11" in s:
        return 5
    return 3

def _horse_label(row):
    return f'{int(row["horseNo"])} {row["horseName"]}'

def _pair_text(honmei, mate):
    return f'{int(honmei["horseNo"])} - {int(mate["horseNo"])}'

def _trio_text(honmei, a, b):
    return f'{int(honmei["horseNo"])} - {int(a["horseNo"])} - {int(b["horseNo"])}'

def _comment_text(row):
    vals = []
    for k in ["短評", "comment", "コメント", "評価コメント"]:
        if k in row.index:
            vals.append(norm_text(row.get(k, "")))
    return " ".join([v for v in vals if v])

def _keyword_count(text_value, words):
    s = norm_text(text_value)
    return sum(1 for w in words if w in s)

def judge_honmei_type(honmei, conf):
    """
    本命タイプ判定:
    1・2着型 = 馬連向き
    3着型 = 三連複/ワイド向き
    """
    rank = str(honmei.get("トータルランク", ""))
    score = float(honmei.get("総合点", 0) or 0)
    body = float(honmei.get("本体点", 0) or 0)
    pace = float(honmei.get("展開位置補正", 1) or 1)
    place = float(honmei.get("前走場所直線補正", 1) or 1)
    pace_eval = _eval_score(honmei.get("paceEval", "普通"))
    straight_eval = _eval_score(honmei.get("straightEval", "普通"))
    pos4 = _position_score(honmei.get("prev4cCat", ""))
    comment = _comment_text(honmei)

    win_words = ["勝ち切り", "押し切り", "主役", "軸上位", "能力上位", "前進", "頭まで", "連軸", "好位", "先行", "安定"]
    third_words = ["堅実", "相手向き", "複勝向き", "3着候補", "差し届けば", "展開待ち", "取りこぼし", "詰め甘い", "善戦型"]

    win_points = 0
    third_points = 0

    if rank in ["S", "A"]:
        win_points += 2
    elif rank == "B":
        third_points += 1

    if conf >= 95:
        win_points += 2
    elif conf >= 90:
        third_points += 1

    if score >= 62:
        win_points += 2
    elif score < 52:
        third_points += 2

    if body >= 42:
        win_points += 1
    elif body < 36:
        third_points += 1

    if pace >= 1.08 and place >= 1.00:
        win_points += 1
    if pace < 1.00 or place < 1.00:
        third_points += 1

    if pace_eval >= 4 and straight_eval >= 3:
        win_points += 1
    if pace_eval <= 3 and straight_eval >= 3:
        third_points += 1

    if pos4 <= 3:
        win_points += 1
    elif pos4 >= 4:
        third_points += 1

    win_points += _keyword_count(comment, win_words) * 2
    third_points += _keyword_count(comment, third_words) * 2

    if third_points >= win_points + 2:
        return "3着型", f"勝ち切りより3着内安定寄り（判定 {win_points}-{third_points}）"
    return "1・2着型", f"1〜2着に来るイメージを優先（判定 {win_points}-{third_points}）"

def _prepare_candidates(g, honmei):
    cand = g[g.index != 0].copy()
    if cand.empty:
        return cand

    h_pace = _eval_score(honmei.get("paceEval", "普通"))
    h_straight = _eval_score(honmei.get("straightEval", "普通"))
    h_3c = _position_score(honmei.get("prev3cCat", ""))
    h_4c = _position_score(honmei.get("prev4cCat", ""))

    cand["rank_score"] = cand["トータルランク"].apply(_rank_score)
    cand["relative_score"] = cand["相対評価"].apply(_rank_score)
    cand["pace_score"] = cand["paceEval"].apply(_eval_score)
    cand["straight_score"] = cand["straightEval"].apply(_eval_score)
    cand["pos3_score"] = cand["prev3cCat"].apply(_position_score)
    cand["pos4_score"] = cand["prev4cCat"].apply(_position_score)

    cand["same_gap"] = (
        (cand["pace_score"] - h_pace).abs()
        + (cand["straight_score"] - h_straight).abs() * 0.5
        + (cand["pos3_score"] - h_3c).abs() * 0.7
        + (cand["pos4_score"] - h_4c).abs() * 0.9
    )
    cand["same_score"] = (
        cand["rank_score"] * 18
        + cand["relative_score"] * 6
        + cand["総合点"] * 0.70
        - cand["same_gap"] * 10
    )

    cand["diff_gap"] = (
        (cand["pace_score"] - h_pace).abs()
        + (cand["pos3_score"] - h_3c).abs() * 0.8
        + (cand["pos4_score"] - h_4c).abs() * 1.0
    )
    cand["comp_fit"] = cand["diff_gap"].apply(lambda x: 14 if 1.0 <= x <= 3.5 else (7 if x > 0 else 0))
    cand["comp_score"] = (
        cand["comp_fit"]
        + cand["rank_score"] * 12
        + cand["総合点"] * 0.55
        + cand["展開位置補正"] * 8
    )

    cand["return_gap"] = (
        (cand["pace_score"] - h_pace).abs()
        + (cand["straight_score"] - h_straight).abs()
        + (cand["pos3_score"] - h_3c).abs() * 0.8
        + (cand["pos4_score"] - h_4c).abs() * 0.8
    )
    cand["plus_comment"] = cand.apply(
        lambda r: 1 if any(k in _comment_text(r) for k in ["向く", "上積", "先行", "差し", "外", "内", "粘", "伸", "好位", "妙味", "穴"]) else 0,
        axis=1
    )
    cand["return_score"] = (
        cand["return_gap"].clip(upper=5) * 5
        + cand["rank_score"] * 9
        + cand["relative_score"] * 4
        + cand["総合点"] * 0.45
        + cand["plus_comment"] * 8
    )
    return cand

def _pick_same(cand, used):
    pool = cand[~cand["horseNo"].isin(used)].copy()
    pool = pool[pool["トータルランク"].isin(["S", "A", "B"])].copy()
    if pool.empty:
        pool = cand[(~cand["horseNo"].isin(used)) & (cand["トータルランク"].astype(str) != "D")].copy()
    if pool.empty:
        return None
    return pool.sort_values(["same_score", "総合点", "horseNo"], ascending=[False, False, True]).iloc[0]

def _pick_comp(cand, used):
    pool = cand[~cand["horseNo"].isin(used)].copy()
    pool = pool[pool["トータルランク"].isin(["S", "A", "B", "C"])].copy()
    pool = pool[pool["総合点"] >= 35].copy()
    if pool.empty:
        pool = cand[(~cand["horseNo"].isin(used)) & (cand["トータルランク"].astype(str) != "D")].copy()
    if pool.empty:
        return None
    return pool.sort_values(["comp_score", "総合点", "horseNo"], ascending=[False, False, True]).iloc[0]

def _pick_return(cand, used, allow_d=True):
    pool = cand[~cand["horseNo"].isin(used)].copy()
    if allow_d:
        non_d = pool[pool["トータルランク"].astype(str) != "D"].copy()
        d_pool = pool[
            (pool["トータルランク"].astype(str) == "D")
            & (pool["plus_comment"] == 1)
            & (pool["総合点"] >= 32)
        ].copy()
        pool = pd.concat([non_d, d_pool], ignore_index=False)
    else:
        pool = pool[pool["トータルランク"].astype(str) != "D"]
    pool = pool[pool["総合点"] >= 32].copy()
    if pool.empty:
        pool = cand[(~cand["horseNo"].isin(used)) & (cand["トータルランク"].astype(str) != "D")].copy()
    if pool.empty:
        return None
    return pool.sort_values(["return_score", "総合点", "horseNo"], ascending=[False, False, True]).iloc[0]

def _pick_top_rest(cand, used, allow_d=False):
    pool = cand[~cand["horseNo"].isin(used)].copy()
    if not allow_d:
        pool = pool[pool["トータルランク"].astype(str) != "D"]
    if pool.empty:
        return None
    return pool.sort_values(["総合点", "horseNo"], ascending=[False, True]).iloc[0]

def _build_trio_bets(honmei, mates3):
    a, b, c = mates3[0], mates3[1], mates3[2]
    return [
        (_trio_text(honmei, a, b), "三連複"),
        (_trio_text(honmei, a, c), "三連複"),
        (_trio_text(honmei, b, c), "三連複"),
    ]

def _strong_buy_ok(honmei, conf, honmei_type, mates):
    if conf < 95.0 or honmei_type != "1・2着型" or len(mates) < 3:
        return False, "強気条件未満"
    ab_count = sum(1 for m in mates if str(m.get("トータルランク", "")) in ["S", "A", "B"])
    d_count = sum(1 for m in mates if str(m.get("トータルランク", "")) == "D")
    avg_score = sum(float(m.get("総合点", 0) or 0) for m in mates) / len(mates)
    if ab_count >= 2 and d_count <= 1 and avg_score >= 42:
        return True, "本命信頼度95%以上かつ相手3頭の中にA/B評価が2頭以上"
    return False, "相手3頭のまとまりが強気条件未満"

def recommend_for_race(g):
    """
    おすすめ馬券ロジック:
    - 本命は既存の単複おすすめ1をそのまま使用
    - 信頼度90%未満は見送り
    - 通常買い / 強気買い / 見送りを自動判定
    - 1・2着型: 通常=馬連3点、強気=馬連3点+三連複3点
    - 3着型: 通常=三連複3点、必要ならワイド2〜3点
    """
    g = g.sort_values(["総合点","horseNo"], ascending=[False, True]).reset_index(drop=True)
    honmei = g.iloc[0]
    conf = confidence_from_score(honmei["総合点"])
    short_comment = f'本体{honmei["本体点"]:.1f}×展開{honmei["展開位置補正"]:.2f}×場所{honmei["前走場所直線補正"]:.2f}'

    if conf < 90.0:
        return {
            "honmei": honmei,
            "confidence": conf,
            "status": "見送り",
            "honmei_type": "対象外",
            "bet_strength": "見送り",
            "bet_type": "見送り",
            "bets": [],
            "wide_bets": [],
            "umaren_bets": [],
            "trio_bets": [],
            "mates": [],
            "reason": "本命信頼度が90%未満",
            "short_comment": short_comment,
        }

    honmei_type, type_reason = judge_honmei_type(honmei, conf)
    cand = _prepare_candidates(g, honmei)
    if cand.empty:
        return {
            "honmei": honmei,
            "confidence": conf,
            "status": "見送り",
            "honmei_type": honmei_type,
            "bet_strength": "見送り",
            "bet_type": "見送り",
            "bets": [],
            "wide_bets": [],
            "umaren_bets": [],
            "trio_bets": [],
            "mates": [],
            "reason": "おすすめ馬券に必要な相手が揃わない",
            "short_comment": short_comment,
        }

    used = {honmei["horseNo"]}

    same = _pick_same(cand, used)
    if same is not None:
        used.add(same["horseNo"])

    comp = _pick_comp(cand, used)
    if comp is not None:
        used.add(comp["horseNo"])

    ret = _pick_return(cand, used, allow_d=True)
    if ret is not None:
        used.add(ret["horseNo"])

    mates = [m for m in [same, comp, ret] if m is not None]

    if len(mates) < 3:
        return {
            "honmei": honmei,
            "confidence": conf,
            "status": "見送り",
            "honmei_type": honmei_type,
            "bet_strength": "見送り",
            "bet_type": "見送り",
            "bets": [],
            "wide_bets": [],
            "umaren_bets": [],
            "trio_bets": [],
            "mates": mates,
            "reason": "おすすめ馬券に必要な相手が揃わない",
            "short_comment": short_comment,
        }

    same_reason = f'{_horse_label(same)}は本命と展開・位置取りが近い同展開相手。'
    comp_reason = f'{_horse_label(comp)}は本命と位置/展開にズレがあり、展開ズレを拾う補完相手。'
    ret_reason = f'{_horse_label(ret)}は違う勝ち筋で配当上振れを狙う回収相手。'

    d_count = sum(1 for m in mates if str(m.get("トータルランク", "")) == "D")
    d_reason = ""
    if d_count >= 1:
        d_horses = " / ".join([_horse_label(m) for m in mates if str(m.get("トータルランク", "")) == "D"])
        d_reason = f'\nD評価採用理由：{d_horses}は回収相手枠限定で、短評/展開面のプラス材料を評価。'

    base_reason = f'本命は{type_reason}。\n' + same_reason + "\n" + comp_reason + "\n" + ret_reason + d_reason

    umaren_bets = [
        (_pair_text(honmei, same), "同展開相手"),
        (_pair_text(honmei, comp), "補完相手"),
        (_pair_text(honmei, ret), "回収相手"),
    ]
    trio_bets = _build_trio_bets(honmei, mates)

    strong_ok, strong_reason = _strong_buy_ok(honmei, conf, honmei_type, mates)

    if strong_ok:
        return {
            "honmei": honmei,
            "confidence": conf,
            "status": "買い対象",
            "honmei_type": honmei_type,
            "bet_strength": "強気",
            "bet_type": "馬連3点＋三連複3点",
            "bets": umaren_bets + trio_bets,
            "wide_bets": [],
            "umaren_bets": umaren_bets,
            "trio_bets": trio_bets,
            "mates": mates,
            "reason": base_reason + f'\n強気買い理由：{strong_reason}',
            "short_comment": short_comment,
        }

    if honmei_type == "1・2着型":
        return {
            "honmei": honmei,
            "confidence": conf,
            "status": "買い対象",
            "honmei_type": honmei_type,
            "bet_strength": "通常",
            "bet_type": "馬連3点",
            "bets": umaren_bets,
            "wide_bets": [],
            "umaren_bets": umaren_bets,
            "trio_bets": [],
            "mates": mates,
            "reason": base_reason + "\n通常買い理由：本命が1・2着型のため馬連を優先。",
            "short_comment": short_comment,
        }

    # 3着型は三連複3点を優先。相手が不安定な場合はワイド2〜3点に切り替え。
    mate_ab_count = sum(1 for m in mates if str(m.get("トータルランク", "")) in ["S", "A", "B"])
    if mate_ab_count >= 2:
        return {
            "honmei": honmei,
            "confidence": conf,
            "status": "買い対象",
            "honmei_type": honmei_type,
            "bet_strength": "通常",
            "bet_type": "三連複3点",
            "bets": trio_bets,
            "wide_bets": [],
            "umaren_bets": [],
            "trio_bets": trio_bets,
            "mates": mates,
            "reason": base_reason + "\n通常買い理由：本命が3着型のため三連複で複勝力を活かす。",
            "short_comment": short_comment,
        }

    wide_bets = [
        (_pair_text(honmei, same), "同展開相手"),
        (_pair_text(honmei, comp), "補完相手"),
    ]
    if ret is not None and str(ret.get("トータルランク", "")) != "D":
        wide_bets.append((_pair_text(honmei, ret), "回収相手"))

    if len(wide_bets) < 2:
        return {
            "honmei": honmei,
            "confidence": conf,
            "status": "見送り",
            "honmei_type": honmei_type,
            "bet_strength": "見送り",
            "bet_type": "見送り",
            "bets": [],
            "wide_bets": [],
            "umaren_bets": [],
            "trio_bets": [],
            "mates": mates,
            "reason": "本命が3着型だが、ワイド/三連複の相手が不安定",
            "short_comment": short_comment,
        }

    return {
        "honmei": honmei,
        "confidence": conf,
        "status": "買い対象",
        "honmei_type": honmei_type,
        "bet_strength": "通常",
        "bet_type": "ワイド" + str(len(wide_bets)) + "点",
        "bets": wide_bets,
        "wide_bets": wide_bets,
        "umaren_bets": [],
        "trio_bets": [],
        "mates": mates,
        "reason": base_reason + "\n通常買い理由：本命が3着型かつ相手のまとまりが弱いためワイドを優先。",
        "short_comment": short_comment,
    }


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
    # iPhone/Streamlit CloudではHTMLカードの実高さが計算より大きくなりやすいので、
    # かなり余裕を持たせて全頭が切れないようにする。
    height = 190 + len(g) * 112
    components.html(html, height=height, scrolling=False)



def get_font(size, bold=False):
    # Streamlit Cloudでは packages.txt に fonts-noto-cjk を入れると下記Notoが使えます。
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansJP-Bold.otf" if bold else "/usr/share/fonts/opentype/noto/NotoSansJP-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansJP-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/truetype/arphic-bkai00mp/bkai00mp.ttf",
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
    items = [r for r in saved if float(r.get("参考信頼度", 0) or 0) >= 90.0]
    if not items:
        return None

    # 画像表示用の場所名・R番号を必ず作り直す
    # 保存済みデータに「東京11R」のように場所+Rが混ざっていても、ここで正規化する
    def clean_item(r):
        rr = dict(r)
        place_raw = norm_text(rr.get("場所", ""))
        race_raw = norm_text(rr.get("レース", ""))
        single_raw = norm_text(rr.get("単複おすすめ1", ""))

        # 場所にRが混ざっているケースを補正
        m = re.search(r"(福島|新潟|東京|中山|中京|京都|阪神|小倉|札幌|函館)\s*(\d+)\s*R", place_raw)
        if not m:
            m = re.search(r"(福島|新潟|東京|中山|中京|京都|阪神|小倉|札幌|函館)\s*(\d+)\s*R", race_raw)
        if m:
            rr["場所"] = m.group(1)
            rr["R"] = int(m.group(2))
        else:
            rr["場所"] = norm_track(place_raw)
            try:
                rr["R"] = int(float(rr.get("R", 0) or 0))
            except Exception:
                rr["R"] = 0

        # 馬番/馬名も念のため補正
        try:
            rr["馬番"] = int(float(rr.get("馬番", 0) or 0))
        except Exception:
            m2 = re.search(r"(\d+)", single_raw)
            rr["馬番"] = int(m2.group(1)) if m2 else 0

        rr["馬名"] = norm_text(rr.get("馬名", ""))
        if not rr["馬名"] and single_raw:
            rr["馬名"] = re.sub(r"^\d+\s*", "", single_raw)

        rr["参考信頼度"] = float(rr.get("参考信頼度", 0) or 0)
        return rr

    items = [clean_item(r) for r in items]

    # 競馬場ごと → R順で並べる
    # JRAの並びに近い形。必要ならここだけ変更すればOK。
    track_order = {
        "福島": 1,
        "東京": 2,
        "京都": 3,
        "阪神": 4,
        "中山": 5,
        "中京": 6,
        "新潟": 7,
        "小倉": 8,
        "札幌": 9,
        "函館": 10,
    }

    items = sorted(
        items,
        key=lambda r: (
            track_order.get(norm_track(r.get("場所", "")), 99),
            int(float(r.get("R", 0) or 0)),
            int(float(r.get("馬番", 0) or 0)),
        )
    )

    date = str(items[0]["日付"])

    W = 1080
    row_h = 108
    H = max(1280, 285 + len(items) * row_h + 60)
    img = Image.new("RGB", (W, H), (250, 249, 245))
    draw = ImageDraw.Draw(img)

    for y in range(H):
        shade = int(7 * y / H)
        draw.line([(0, y), (W, y)], fill=(250 - shade, 249 - shade, 245 - shade))

    title_font = get_font(58, True)
    date_font = get_font(38, True)
    race_font = get_font(34, True)
    horse_no_font = get_font(38, True)
    horse_font = get_font(42, True)
    conf_font = get_font(24, True)

    navy = (16, 33, 65)
    gold = (209, 166, 59)
    red = (223, 55, 53)
    gray = (92, 100, 116)
    line = (224, 211, 178)
    white = (255, 255, 255)

    # header
    header_x1, header_y1, header_x2, header_y2 = 56, 45, 1024, 195
    draw.rounded_rectangle((header_x1, header_y1, header_x2, header_y2), radius=34, fill=navy, outline=gold, width=4)

    title = "本日の推奨馬"
    tb = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((W - (tb[2] - tb[0])) / 2, 64), title, font=title_font, fill=white)

    db = draw.textbbox((0, 0), date, font=date_font)
    draw.text(((W - (db[2] - db[0])) / 2, 132), date, font=date_font, fill=(238, 224, 174))

    draw.line((70, 235, 1010, 235), fill=gold, width=4)

    y = 282
    for r in items:
        draw.rounded_rectangle((58, y, 1022, y + 84), radius=26, fill=white, outline=line, width=3)

        race_label = f'{norm_track(r["場所"])}{int(float(r.get("R", 0) or 0))}R'
        badge_x1, badge_y1, badge_x2, badge_y2 = 84, y + 18, 250, y + 66
        draw.rounded_rectangle((badge_x1, badge_y1, badge_x2, badge_y2), radius=15, fill=red)
        rb = draw.textbbox((0, 0), race_label, font=race_font)
        rw = rb[2] - rb[0]
        rf = race_font
        if rw > (badge_x2 - badge_x1 - 18):
            rf = get_font(30, True)
            rb = draw.textbbox((0, 0), race_label, font=rf)
            rw = rb[2] - rb[0]
        draw.text((badge_x1 + (badge_x2 - badge_x1 - rw) / 2, y + 22), race_label, font=rf, fill=white)

        no_text = str(int(float(r.get("馬番", 0) or 0)))
        draw.text((292, y + 22), no_text, font=horse_no_font, fill=gold)

        draw_fit_text(draw, (365, y + 19), r["馬名"], horse_font, navy, 445)

        conf = float(r.get("参考信頼度", 0) or 0)
        conf_text = f"{conf:.1f}%"
        cb = draw.textbbox((0, 0), conf_text, font=conf_font)
        draw.text((980 - (cb[2]-cb[0]), y + 30), conf_text, font=conf_font, fill=gray)

        y += row_h

    draw.line((70, H - 55, 1010, H - 55), fill=gold, width=3)

    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio


def safe_race_no(row):
    """raceNo が空/NaNでも、レース表記からR番号を復元する"""
    try:
        v = row.get("raceNo", np.nan)
        if pd.notna(v):
            return int(float(v))
    except Exception:
        pass

    for key in ["レース", "raceLabel", "raceName"]:
        s = norm_text(row.get(key, ""))
        m = re.search(r"(\d+)\s*R", s)
        if m:
            return int(m.group(1))
    return 0



def add_saved_recs(new_recs):
    if "saved_recs" not in st.session_state:
        st.session_state.saved_recs = []

    cleaned = []
    for r in new_recs:
        rr = dict(r)
        place_raw = norm_text(rr.get("場所", ""))
        m = re.search(r"(福島|新潟|東京|中山|中京|京都|阪神|小倉|札幌|函館)\s*(\d+)\s*R", place_raw)
        if m:
            rr["場所"] = m.group(1)
            rr["R"] = int(m.group(2))
        else:
            rr["場所"] = norm_track(place_raw)
            try:
                rr["R"] = int(float(rr.get("R", 0) or 0))
            except Exception:
                rr["R"] = 0
        cleaned.append(rr)

    # key = 日付 + 場所 + R, update existing
    store = {}
    for r in st.session_state.saved_recs:
        rr = dict(r)
        place = norm_track(norm_text(rr.get("場所", "")))
        try:
            race_no = int(float(rr.get("R", 0) or 0))
        except Exception:
            race_no = 0
        rr["場所"] = place
        rr["R"] = race_no
        store[f'{rr.get("日付","")}_{place}_{race_no}R'] = rr

    for r in cleaned:
        store[f'{r["日付"]}_{r["場所"]}_{r["R"]}R'] = r

    st.session_state.saved_recs = list(store.values())


def saved_df():
    if "saved_recs" not in st.session_state:
        st.session_state.saved_recs = []
    return pd.DataFrame(st.session_state.saved_recs)

st.title("競馬ランクアプリ v12.2 Gemini Auto")
st.write("CSV読込でも、Gemini APIで1レースずつ自動予想でも使えます。ランキング・買い目・SNS画像は従来通り表示します。")

if "saved_recs" not in st.session_state:
    st.session_state.saved_recs = []
if "auto_df" not in st.session_state:
    st.session_state.auto_df = None

with st.expander("Gemini APIで自動予想", expanded=False):
    api_key_input = st.text_input(
        "Gemini APIキー（アプリ内入力・任意）",
        type="password",
        placeholder="AIza...",
        help="Streamlit SecretsにGEMINI_API_KEYを設定している場合は未入力でも動きます。GitHubには保存されません。",
    )
    if api_key_input:
        st.session_state["GEMINI_API_KEY_INPUT"] = api_key_input

    st.caption("入力するのは日付・場所・Rだけです。レース名、芝/ダ、距離はGeminiが検索して自動取得します。")

    c1, c2, c3 = st.columns(3)
    with c1:
        auto_date = st.text_input("日付", value="2026-04-26", key="auto_date")
    with c2:
        auto_place = st.selectbox("場所", ["東京", "京都", "福島", "中山", "阪神", "中京", "新潟", "小倉", "札幌", "函館"], key="auto_place")
    with c3:
        auto_race_no = st.number_input("単発R", min_value=1, max_value=12, value=11, step=1, key="auto_race_no")

    use_day_bias = st.checkbox("馬場・TB検索込み", value=True)
    model_name = st.text_input("Gemini model", value="gemini-2.5-flash-lite")
    retry_no_search = st.checkbox("429時は検索なしで再試行", value=True)

    st.markdown("#### 単発予想")
    if st.button("この1レースを全自動予想", type="primary"):
        prompt = build_gemini_prediction_prompt(
            auto_date, auto_place, int(auto_race_no), use_day_bias=use_day_bias
        )
        try:
            with st.spinner("Geminiでレース名・芝ダ・距離・出走馬・馬場TBを検索してCSVを作成中です..."):
                csv_data = call_gemini_with_search(prompt, model_name=model_name, use_search=use_day_bias, retry_no_search=retry_no_search)
                st.session_state.auto_df = gemini_csv_to_df(csv_data)
            st.success("Gemini予想CSVを作成しました。下のランキングに反映します。")
            with st.expander("Gemini生成CSVを確認"):
                st.dataframe(st.session_state.auto_df, use_container_width=True)
                st.download_button(
                    "Gemini生成CSVをダウンロード",
                    data=st.session_state.auto_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                    file_name=f"{auto_date}_{auto_place}{int(auto_race_no)}R_gemini.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error("Gemini API実行でエラーが発生しました。")
            st.code(str(e))

    st.markdown("#### 7〜12Rまとめ予想")
    st.caption("ボタン1つで7R〜12Rを順番に実行し、6レース分を1つのCSVにまとめます。429対策のため連続実行の間隔を空けられます。")

    b1, b2, b3 = st.columns(3)
    with b1:
        batch_start = st.number_input("開始R", min_value=1, max_value=12, value=7, step=1, key="batch_start")
    with b2:
        batch_end = st.number_input("終了R", min_value=1, max_value=12, value=12, step=1, key="batch_end")
    with b3:
        batch_sleep = st.number_input("実行間隔 秒", min_value=0, max_value=60, value=5, step=1, key="batch_sleep")

    if st.button("7〜12Rを一気に全自動予想"):
        if int(batch_start) > int(batch_end):
            st.error("開始Rは終了R以下にしてください。")
        else:
            dfs = []
            errors = []
            progress = st.progress(0)
            race_nums = list(range(int(batch_start), int(batch_end) + 1))
            status_box = st.empty()

            for i, rn in enumerate(race_nums, start=1):
                status_box.info(f"{auto_place}{rn}RをGeminiで予想中...（{i}/{len(race_nums)}）")
                prompt = build_gemini_prediction_prompt(
                    auto_date, auto_place, int(rn), use_day_bias=use_day_bias
                )
                try:
                    csv_data = call_gemini_with_search(
                        prompt,
                        model_name=model_name,
                        use_search=use_day_bias,
                        retry_no_search=retry_no_search,
                    )
                    one_df = gemini_csv_to_df(csv_data)
                    dfs.append(one_df)
                    st.success(f"{auto_place}{rn}R 完了")
                except Exception as e:
                    errors.append(f"{auto_place}{rn}R: {str(e)}")
                    st.warning(f"{auto_place}{rn}R は取得できませんでした。次へ進みます。")

                progress.progress(i / len(race_nums))
                if i < len(race_nums) and int(batch_sleep) > 0:
                    time.sleep(int(batch_sleep))

            if dfs:
                st.session_state.auto_df = pd.concat(dfs, ignore_index=True)
                status_box.success(f"{auto_place}{int(batch_start)}〜{int(batch_end)}R のGemini予想CSVを作成しました。下のランキングに反映します。")
                with st.expander("まとめ生成CSVを確認", expanded=True):
                    st.dataframe(st.session_state.auto_df, use_container_width=True)
                    st.download_button(
                        "7〜12RまとめCSVをダウンロード",
                        data=st.session_state.auto_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                        file_name=f"{auto_date}_{auto_place}_{int(batch_start)}-{int(batch_end)}R_gemini.csv",
                        mime="text/csv",
                    )
            else:
                status_box.error("全レース取得できませんでした。10〜15分置くか、検索込みOFF/flash-liteで再実行してください。")

            if errors:
                with st.expander("取得できなかったレース"):
                    for er in errors:
                        st.code(er)

uploaded = st.file_uploader("CSVをアップロードして使う場合はこちら", type=["csv"])

current_recs = []

if uploaded is not None:
    source_df = read_csv_any(uploaded)
elif st.session_state.auto_df is not None:
    source_df = st.session_state.auto_df.copy()
else:
    source_df = None

if source_df is None:
    st.info("CSVをアップロードするか、Gemini APIで1レース自動予想を実行してください。")
else:
    prev3c_stat, prev4c_stat, prevtrack_stat = load_stat_defaults()
    df = prepare_race_df(source_df)

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
            rec = recommend_for_race(g)
            honmei = rec["honmei"]
            conf = rec["confidence"]

            st.markdown("### 単複おすすめ1")
            st.write(f'候補: {int(honmei["horseNo"])} {honmei["horseName"]}')
            st.caption(f'トータル{honmei["トータルランク"]} / 総合点 {honmei["総合点"]:.2f} / 参考信頼度 {conf:.2f}% / {rec["short_comment"]}')

            st.markdown("### おすすめ馬券")
            if rec["status"] == "見送り":
                st.write("見送り")
                st.caption(f'理由：{rec["reason"]}')
            else:
                st.write(f'本命タイプ：{rec["honmei_type"]}')
                st.write(f'勝負度：{rec["bet_strength"]}')
                st.write(f'おすすめ馬券：{rec["bet_type"]}')

                if rec.get("umaren_bets"):
                    st.write("馬連：")
                    for bet, role in rec["umaren_bets"]:
                        st.write(f'{bet}　{role}')

                if rec.get("trio_bets"):
                    st.write("三連複：")
                    for bet, role in rec["trio_bets"]:
                        st.write(bet)

                if rec.get("wide_bets"):
                    st.write("ワイド：")
                    for bet, role in rec["wide_bets"]:
                        st.write(f'{bet}　{role}')

                st.caption("理由：\n" + rec["reason"])

            st.divider()

            bet_values = [b[0] for b in rec.get("bets", [])]
            current_recs.append({
                "日付": honmei["date"],
                "場所": honmei["場所"],
                "R": safe_race_no(honmei),
                "馬番": int(honmei["horseNo"]),
                "馬名": honmei["horseName"],
                "単複おすすめ1": f'{int(honmei["horseNo"])} {honmei["horseName"]}',
                "参考信頼度": round(float(conf), 2),
                "短評": rec["short_comment"],
                "買い対象": 1 if rec["status"] == "買い対象" else 0,
                "本命タイプ": rec["honmei_type"],
                "勝負度": rec["bet_strength"],
                "おすすめ馬券": rec["bet_type"],
                "買い目1": bet_values[0] if len(bet_values) > 0 else "",
                "買い目2": bet_values[1] if len(bet_values) > 1 else "",
                "買い目3": bet_values[2] if len(bet_values) > 2 else "",
                "買い目4": bet_values[3] if len(bet_values) > 3 else "",
                "買い目5": bet_values[4] if len(bet_values) > 4 else "",
                "買い目6": bet_values[5] if len(bet_values) > 5 else "",
                "相手選定理由": rec["reason"],
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
