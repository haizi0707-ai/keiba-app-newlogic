import os
import re
import io
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="競馬ランクアプリ v12.9 Dabista Vertical SNS Save", layout="centered")

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()

# v12.9:
# まず相対位置版の履歴ファイルを読みに行きます。
# なければ旧ファイルへフォールバックします。
DEFAULT_FILES = {
    "prev3c": os.path.join(BASE_DIR, "prev3c_relative_category_stats.csv"),
    "prev3c_fallback": os.path.join(BASE_DIR, "prev3c_category_stats.csv"),
    "prev4c": os.path.join(BASE_DIR, "prev4c_relative_category_stats.csv"),
    "prev4c_fallback": os.path.join(BASE_DIR, "prev4c_category_stats.csv"),
    "prevtrack": os.path.join(BASE_DIR, "prevtrack_roi_stats.csv"),
}

EVAL_MAP = {"かなり向く":1.25, "向く":1.10, "普通":1.00, "やや不向き":0.85, "不向き":0.70}

def resolve_file(primary, fallback=None):
    """相対位置版ファイルがあれば優先、なければ旧ファイルを読む"""
    if primary and os.path.exists(primary):
        return primary
    if fallback and os.path.exists(fallback):
        return fallback
    return primary

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
    "raceNo":["raceNo","race_number","R","Ｒ","レース番号","race_no","レースNo","R番号"],
    "raceName":["raceName","race_name","レース名"],
    "horseNo":["horseNo","horse_number","馬番"],
    "horseName":["horseName","horse_name","馬名"],
    "distance":["distance","距離","距離数値"],
    "surface":["芝ダ","芝・ダ","surface"],
    "raceLabel":["レース","race"],
    "prevTrack":["前走競馬場","前走場所","prevTrack"],
    "prevSurface":["前走芝ダ","前芝・ダ","prevSurface"],
    "prevDistance":["前走距離数値","前走距離","前距離","prevDistance"],

    # v12.2 追加：予想CSV側にこの3列があれば、アプリ側で相対位置補正します。
    "prevFieldSize":["前走頭数","前走出走頭数","prevFieldSize","fieldSize"],
    "prev3cPos":["前3角通過順","前走3角通過順","前3角順位","前3角","前3角通過","prev3cPos"],
    "prev4cPos":["前4角通過順","前走4角通過順","前4角順位","前4角","前4角通過","prev4cPos"],
    "prev3cRel":["前3角相対位置","前走3角相対位置","prev3cRel"],
    "prev4cRel":["前4角相対位置","前走4角相対位置","prev4cRel"],

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

def normalize_existing_position_cat(v):
    s = norm_text(v)
    s = s.replace("２", "2").replace("３", "3").replace("４", "4").replace("６", "6")
    s = s.replace("７", "7").replace("１０", "10").replace("１１", "11")
    s = s.replace("~", "-").replace("〜", "-").replace("－", "-")
    aliases = {
        "逃げ": "1番手",
        "先頭": "1番手",
        "1": "1番手",
        "1番": "1番手",
        "1番手": "1番手",
        "2-3": "2-3番手",
        "2-3番手": "2-3番手",
        "2番手": "2-3番手",
        "3番手": "2-3番手",
        "4-6": "4-6番手",
        "4-6番手": "4-6番手",
        "4番手": "4-6番手",
        "5番手": "4-6番手",
        "6番手": "4-6番手",
        "7-10": "7-10番手",
        "7-10番手": "7-10番手",
        "7番手": "7-10番手",
        "8番手": "7-10番手",
        "9番手": "7-10番手",
        "10番手": "7-10番手",
        "11番手以下": "11番手以下",
        "11以下": "11番手以下",
        "後方": "11番手以下",
        "最後方": "11番手以下",
    }
    return aliases.get(s, s)

def calc_relative_position(pos, field_size):
    if pd.isna(pos) or pd.isna(field_size):
        return np.nan
    try:
        pos = float(pos)
        field_size = float(field_size)
    except Exception:
        return np.nan
    if field_size <= 0 or pos <= 0:
        return np.nan
    return pos / field_size

def rel_to_app_category(rel, raw_pos=None):
    """
    予想CSV側の前走頭数・通過順から、アプリ既存5カテゴリに変換。
    通過順位1番手は無条件で「1番手」。
    2番手以下は相対位置で判定。
    """
    if raw_pos is not None and not pd.isna(raw_pos):
        try:
            if float(raw_pos) <= 1:
                return "1番手"
        except Exception:
            pass

    if pd.isna(rel):
        return ""

    try:
        rel = float(rel)
    except Exception:
        return ""

    if rel <= 0.25:
        return "2-3番手"
    if rel <= 0.45:
        return "4-6番手"
    if rel <= 0.85:
        return "7-10番手"
    return "11番手以下"

def apply_relative_position_logic(df):
    """
    予想CSVに前走頭数・前3角通過順・前4角通過順がある場合、
    前3角位置カテゴリ・前4角位置カテゴリを相対位置ベースに上書きします。
    ない場合は、入力済みカテゴリをそのまま使います。
    """
    out = df.copy()

    out["prev3cCat_original"] = out["prev3cCat"].apply(normalize_existing_position_cat)
    out["prev4cCat_original"] = out["prev4cCat"].apply(normalize_existing_position_cat)

    out["prev3cRel_calc"] = out.apply(lambda r: calc_relative_position(r["prev3cPos"], r["prevFieldSize"]), axis=1)
    out["prev4cRel_calc"] = out.apply(lambda r: calc_relative_position(r["prev4cPos"], r["prevFieldSize"]), axis=1)

    out["prev3cRel_final"] = np.where(pd.notna(out["prev3cRel"]), out["prev3cRel"], out["prev3cRel_calc"])
    out["prev4cRel_final"] = np.where(pd.notna(out["prev4cRel"]), out["prev4cRel"], out["prev4cRel_calc"])

    out["prev3cCat_relative"] = out.apply(lambda r: rel_to_app_category(r["prev3cRel_final"], r["prev3cPos"]), axis=1)
    out["prev4cCat_relative"] = out.apply(lambda r: rel_to_app_category(r["prev4cRel_final"], r["prev4cPos"]), axis=1)

    out["prev3cCat"] = np.where(
        out["prev3cCat_relative"].astype(str).str.strip() != "",
        out["prev3cCat_relative"],
        out["prev3cCat_original"],
    )
    out["prev4cCat"] = np.where(
        out["prev4cCat_relative"].astype(str).str.strip() != "",
        out["prev4cCat_relative"],
        out["prev4cCat_original"],
    )

    out["通過順補正"] = np.where(
        pd.notna(out["prev3cRel_final"]) | pd.notna(out["prev4cRel_final"]),
        "総頭数補正あり",
        "CSVカテゴリ使用",
    )
    return out

def prepare_race_df(df):
    df = rename_first_match(df, CANDS)
    for col in CANDS.keys():
        if col not in df.columns:
            df[col] = ""

    parsed = df["raceLabel"].apply(parse_race_label)
    df["場所"] = np.where(df["場所"].astype(str).str.strip() != "", df["場所"], parsed.apply(lambda x: x[0]))
    df["raceNo"] = np.where(df["raceNo"].astype(str).str.strip() != "", df["raceNo"], parsed.apply(lambda x: x[1]))
    df["raceNo"] = pd.to_numeric(df["raceNo"], errors="coerce")

    # 0R対策：raceNo が0/NaNの場合は、レース表記から再抽出する
    def restore_race_no(row):
        try:
            v = row.get("raceNo", np.nan)
            if pd.notna(v) and int(float(v)) > 0:
                return int(float(v))
        except Exception:
            pass

        for key in ["raceLabel", "レース", "raceName"]:
            s = norm_text(row.get(key, ""))
            m = re.search(r"(\d{1,2})\s*R", s, flags=re.IGNORECASE)
            if m:
                n = int(m.group(1))
                if n > 0:
                    return n
            m = re.search(r"第\s*(\d{1,2})\s*競走", s)
            if m:
                n = int(m.group(1))
                if n > 0:
                    return n
            if re.fullmatch(r"\d{1,2}", s):
                n = int(s)
                if 1 <= n <= 12:
                    return n
        return np.nan

    df["raceNo"] = df.apply(restore_race_no, axis=1)

    for col in ["場所","prevTrack"]:
        df[col] = df[col].apply(norm_track)
    for col in ["surface","prevSurface"]:
        df[col] = df[col].apply(norm_surface)
    for col in ["horseName","raceName","date","prev3cCat","prev4cCat","paceEval","straightEval"]:
        df[col] = df[col].apply(norm_text)

    num_cols = [
        "distance","raceNo","horseNo","prevDistance","prevStraight","prev2Straight",
        "prevFieldSize","prev3cPos","prev4cPos","prev3cRel","prev4cRel",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["prevStraight"] = df["prevStraight"].fillna(50.0).clip(0, 100)
    df["prev2Straight"] = df["prev2Straight"].fillna(50.0).clip(0, 100)

    # v12.9: 予想CSV側の前走頭数・通過順から相対位置カテゴリを自動作成
    df = apply_relative_position_logic(df)

    df["距離表示"] = np.where(df["surface"].astype(str) != "", df["surface"] + df["distance"].fillna(0).astype(int).astype(str), "")
    df["レース"] = df.apply(lambda r: f"{r['場所']}{int(r['raceNo'])}R" if pd.notna(r["raceNo"]) and norm_text(r["場所"]) else norm_text(r["raceLabel"]), axis=1)

    # 最重要: 日付 + 場所 + R で識別。日付 + R だけで混ぜない。
    df["レース識別ID"] = df.apply(
        lambda r: f"{r['date']}_{r['場所']}_{int(r['raceNo'])}R" if pd.notna(r["raceNo"]) else f"{r['date']}_{r['場所']}_{r['レース']}",
        axis=1
    )
    return df

def load_stat_defaults():
    prev3c_path = resolve_file(DEFAULT_FILES["prev3c"], DEFAULT_FILES["prev3c_fallback"])
    prev4c_path = resolve_file(DEFAULT_FILES["prev4c"], DEFAULT_FILES["prev4c_fallback"])

    prev3c = read_csv_any(prev3c_path)
    prev4c = read_csv_any(prev4c_path)
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
    prev3c["prev3cCat"] = prev3c["prev3cCat"].apply(normalize_existing_position_cat)
    prev4c["prev4cCat"] = prev4c["prev4cCat"].apply(normalize_existing_position_cat)

    prevtrack["場所"] = prevtrack["場所"].apply(norm_track)
    prevtrack["surface"] = prevtrack["surface"].apply(norm_surface)
    prevtrack["distance"] = pd.to_numeric(prevtrack["distance"], errors="coerce")
    prevtrack["prevTrack"] = prevtrack["prevTrack"].apply(norm_track)
    prevtrack["count"] = pd.to_numeric(prevtrack["count"], errors="coerce")
    prevtrack["place_rate"] = pd.to_numeric(prevtrack["place_rate"], errors="coerce")

    return prev3c, prev4c, prevtrack, os.path.basename(prev3c_path), os.path.basename(prev4c_path)

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
    s = normalize_existing_position_cat(v)
    if s == "1番手":
        return 1
    if s == "2-3番手":
        return 2
    if s == "4-6番手":
        return 3
    if s == "7-10番手":
        return 4
    if s == "11番手以下":
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


def fit_text_to_width(draw, text, font, max_width):
    t = str(text)
    while len(t) > 0:
        bbox = draw.textbbox((0, 0), t, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return t
        t = t[:-1]
    return ""

def draw_centered_text(draw, box, text, font, fill):
    x1, y1, x2, y2 = box
    t = str(text)
    bbox = draw.textbbox((0, 0), t, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    # bbox の上方向オフセットを補正して、見た目の上下中央に寄せる
    x = x1 + (x2 - x1 - tw) / 2 - bbox[0]
    y = y1 + (y2 - y1 - th) / 2 - bbox[1]
    draw.text((x, y), t, font=font, fill=fill)

def draw_fit_centered_text(draw, box, text, font, fill):
    x1, y1, x2, y2 = box
    t = fit_text_to_width(draw, text, font, x2 - x1)
    if t != str(text):
        t = fit_text_to_width(draw, str(text) + "…", font, x2 - x1)
        if not t.endswith("…"):
            t = fit_text_to_width(draw, str(text)[:-1] + "…", font, x2 - x1)
    draw_centered_text(draw, box, t, font, fill)


def extract_opponent_marks(r):
    """
    保存済み推奨馬の買い目から、本命以外の相手馬番を最大3頭抽出して
    SNS画像用の「◯1 ▲2 △3」形式にする。
    """
    try:
        honmei_no = int(float(r.get("馬番", 0) or 0))
    except Exception:
        honmei_no = 0

    opponents = []
    for key in ["買い目1", "買い目2", "買い目3", "買い目4", "買い目5", "買い目6"]:
        text = norm_text(r.get(key, ""))
        if not text:
            continue

        nums = re.findall(r"\d+", text)
        for n in nums:
            try:
                num = int(n)
            except Exception:
                continue
            if num <= 0:
                continue
            if honmei_no and num == honmei_no:
                continue
            if num not in opponents:
                opponents.append(num)
            if len(opponents) >= 3:
                break
        if len(opponents) >= 3:
            break

    marks = ["◯", "▲", "△"]
    return " ".join([f"{marks[i]}{opponents[i]}" for i in range(min(3, len(opponents)))])

def make_sns_image(saved):
    items = [r for r in saved if float(r.get("参考信頼度", 0) or 0) >= 90.0]
    if not items:
        return None

    def clean_item(r):
        rr = dict(r)
        place_raw = norm_text(rr.get("場所", ""))
        race_raw = norm_text(rr.get("レース", ""))
        single_raw = norm_text(rr.get("単複おすすめ1", ""))

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

        try:
            rr["馬番"] = int(float(rr.get("馬番", 0) or 0))
        except Exception:
            m2 = re.search(r"(\d+)", single_raw)
            rr["馬番"] = int(m2.group(1)) if m2 else 0

        rr["馬名"] = norm_text(rr.get("馬名", ""))
        if not rr["馬名"] and single_raw:
            rr["馬名"] = re.sub(r"^\d+\s*", "", single_raw)

        rr["参考信頼度"] = float(rr.get("参考信頼度", 0) or 0)
        rr["相手表示"] = extract_opponent_marks(rr)
        return rr

    items = [clean_item(r) for r in items]

    track_order = {
        "福島": 1, "東京": 2, "京都": 3, "阪神": 4, "中山": 5,
        "中京": 6, "新潟": 7, "小倉": 8, "札幌": 9, "函館": 10,
    }
    items = sorted(
        items,
        key=lambda r: (
            track_order.get(norm_track(r.get("場所", "")), 99),
            int(float(r.get("R", 0) or 0)),
            int(float(r.get("馬番", 0) or 0)),
        )
    )

    raw_date = str(items[0].get("日付", ""))
    dt = pd.to_datetime(raw_date, errors="coerce")
    if pd.isna(dt):
        date_main = raw_date.replace("/", ".").replace("-", ".")
        weekday = ""
    else:
        date_main = dt.strftime("%Y.%m.%d")
        weekday = dt.strftime("%A").upper()

    # v12.9 縦画面・ダビスタ風ピクセル背景
    # 推奨馬は全頭表示。頭数に応じて縦幅を自動拡張する。
    W = 1080
    row_y = 320
    row_h = 166
    bottom_margin = 110
    H = max(1920, row_y + len(items) * row_h + bottom_margin)
    img = Image.new("RGB", (W, H), (60, 115, 150))
    draw = ImageDraw.Draw(img)

    # 低解像度で背景を描いて拡大し、ダビスタ風のピクセル感を作る
    scale = 6
    sw, sh = W // scale, H // scale
    bg_small = Image.new("RGB", (sw, sh), (79, 137, 176))
    sd = ImageDraw.Draw(bg_small)

    # 空グラデーション
    for y in range(sh):
        t = y / sh
        if y < int(sh * 0.55):
            r = int(67 + 20 * t)
            g = int(126 + 28 * t)
            b = int(170 + 20 * t)
        else:
            r = int(65 - 18 * (t - 0.55))
            g = int(124 - 36 * (t - 0.55))
            b = int(150 - 70 * (t - 0.55))
        sd.line((0, y, sw, y), fill=(max(0, r), max(0, g), max(0, b)))

    # 雲
    cloud = (174, 190, 190)
    cloud_shadow = (130, 150, 158)
    for cx, cy, w in [(22, 75, 38), (126, 50, 34), (152, 118, 26)]:
        sd.rectangle((cx, cy+7, cx+w, cy+12), fill=cloud_shadow)
        sd.rectangle((cx+8, cy, cx+w-5, cy+10), fill=cloud)
        sd.rectangle((cx+15, cy-6, cx+w-15, cy+14), fill=cloud)
        sd.rectangle((cx+2, cy+6, cx+w+8, cy+18), fill=cloud)

    # 遠景の山
    mountain = (89, 123, 145)
    mountain2 = (70, 105, 132)
    sd.polygon([(0, 180), (28, 120), (58, 180)], fill=mountain2)
    sd.polygon([(34, 180), (82, 112), (135, 180)], fill=mountain)
    sd.polygon([(100, 180), (145, 124), (190, 180)], fill=mountain2)
    sd.rectangle((0, 170, sw, 190), fill=(83, 126, 122))

    # スタンド右側
    stand_x = 132
    sd.polygon([(stand_x, 105), (sw, 62), (sw, 86), (stand_x, 128)], fill=(83, 95, 103))
    sd.polygon([(stand_x, 128), (sw, 88), (sw, 180), (stand_x, 190)], fill=(118, 123, 118))
    for yy in range(134, 185, 9):
        sd.line((stand_x, yy, sw, yy-13), fill=(48, 57, 66), width=1)
    for k, col in enumerate([(210, 55, 55), (235, 205, 70), (65, 95, 170), (235, 235, 220)]):
        for x in range(stand_x + 4 + k * 5, sw, 20):
            for yy in range(136, 178, 12):
                sd.rectangle((x, yy, x+2, yy+2), fill=col)

    # 芝
    turf_top = int(sh * 0.56)
    sd.rectangle((0, turf_top, sw, sh), fill=(55, 126, 57))
    for yy in range(turf_top, sh, 5):
        c = (45 + (yy % 17), 105 + (yy % 23), 46)
        sd.line((0, yy, sw, yy-16), fill=c, width=1)

    # コース柵
    sd.line((0, sh-96, sw, sh-172), fill=(210, 214, 205), width=3)
    sd.line((0, sh-91, sw, sh-167), fill=(90, 94, 92), width=1)

    # 背景を拡大
    bg = bg_small.resize((W, H), Image.Resampling.NEAREST)
    img.paste(bg)
    draw = ImageDraw.Draw(img)

    # 半透明風の暗幕で文字を読みやすくする
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 70))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 色
    navy = (15, 43, 72)
    white = (248, 248, 241)
    gold = (248, 203, 54)
    pale_gold = (255, 232, 115)
    black = (17, 21, 25)
    line = (236, 236, 224)
    shadow = (20, 23, 28)

    # フォント
    title_font = get_font(64, True)
    eng_font = get_font(34, True)
    date_font = get_font(37, True)
    weekday_font = get_font(29, True)
    place_font = get_font(33, True)
    race_font = get_font(42, True)
    horse_no_font = get_font(58, True)
    horse_font = get_font(46, True)
    mate_font = get_font(29, True)

    # 文字縁取り用
    def stroke_text(pos, text, font, fill, stroke_fill=shadow, stroke_width=3):
        x, y = pos
        draw.text((x, y), str(text), font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)

    # ピクセル風ボックス
    def pixel_panel(box, fill, outline=line, width=4, corner=18):
        x1, y1, x2, y2 = box
        pts = [
            (x1+corner, y1), (x2-corner, y1), (x2, y1+corner),
            (x2, y2-corner), (x2-corner, y2), (x1+corner, y2),
            (x1, y2-corner), (x1, y1+corner)
        ]
        draw.polygon(pts, fill=fill, outline=outline)
        for i in range(1, width):
            pts2 = [
                (x1+corner+i, y1+i), (x2-corner-i, y1+i), (x2-i, y1+corner+i),
                (x2-i, y2-corner-i), (x2-corner-i, y2-i), (x1+corner+i, y2-i),
                (x1+i, y2-corner-i), (x1+i, y1+corner+i)
            ]
            draw.line(pts2 + [pts2[0]], fill=outline, width=1)

    # ヘッダー
    pixel_panel((28, 58, 765, 250), navy, outline=line, width=4, corner=26)
    pixel_panel((795, 80, 1052, 225), navy, outline=line, width=4, corner=18)

    # 馬蹄アイコン風
    stroke_text((72, 115), "♞", get_font(76, True), gold, stroke_fill=(96, 68, 12), stroke_width=2)
    draw.text((150, 82), "TODAY'S PICKS", font=eng_font, fill=pale_gold, stroke_width=2, stroke_fill=shadow)
    draw.text((150, 135), "本日の推奨馬", font=title_font, fill=white, stroke_width=3, stroke_fill=shadow)

    draw.text((835, 108), date_main, font=date_font, fill=pale_gold, stroke_width=2, stroke_fill=shadow)
    draw.text((875, 160), weekday, font=weekday_font, fill=(183, 217, 236), stroke_width=2, stroke_fill=shadow)

    # 行：保存済み推奨馬を全頭表示
    for i, r in enumerate(items):
        y = row_y + i * row_h

        place = norm_track(r.get("場所", ""))
        race_no = int(float(r.get("R", 0) or 0))
        horse_no = int(float(r.get("馬番", 0) or 0))
        horse_name = norm_text(r.get("馬名", ""))
        mate_text = norm_text(r.get("相手表示", ""))

        # place/race left
        stroke_text((62, y + 2), place, place_font, pale_gold, stroke_width=3)
        stroke_text((62, y + 44), f"{race_no}R", race_font, pale_gold, stroke_width=3)

        # horse number pixel badge
        bx1, by1, bx2, by2 = 184, y - 4, 284, y + 102
        pixel_panel((bx1, by1, bx2, by2), gold, outline=black, width=4, corner=12)
        draw_centered_text(draw, (bx1, by1, bx2, by2), str(horse_no), horse_no_font, black)

        # horse name
        name_x = 325
        stroke_text((name_x, y + 6), fit_text_to_width(draw, horse_name, horse_font, 650), horse_font, white, stroke_width=4)

        # mates
        if mate_text:
            mate_text_spaced = mate_text.replace("◯", "○ ").replace("▲", "▲ ").replace("△", "△ ")
            stroke_text((name_x, y + 72), mate_text_spaced, mate_font, white, stroke_width=3)

        # underline
        draw.line((name_x, y + 119, 965, y + 119), fill=line, width=3)
        draw.line((name_x, y + 123, 965, y + 123), fill=shadow, width=2)

    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio

def safe_race_no(row):
    """
    raceNo が空/NaN/0でも、レース表記からR番号を復元する。
    0R対策：
    ・raceNo が 0 の場合は無効扱い
    ・レース / raceLabel / レース識別ID / raceName / 元R表記から再抽出
    ・「京都11R」「11R」「第11競走」「11」などを広めに拾う
    """
    # まず数値raceNo。ただし0以下は無効。
    try:
        v = row.get("raceNo", np.nan)
        if pd.notna(v):
            n = int(float(v))
            if n > 0:
                return n
    except Exception:
        pass

    # 次に文字列から復元
    for key in ["レース", "raceLabel", "レース識別ID", "raceName", "R", "Ｒ", "raceNo"]:
        s = norm_text(row.get(key, ""))
        if not s:
            continue

        # 京都11R / 11R / 11 R
        m = re.search(r"(\d{1,2})\s*R", s, flags=re.IGNORECASE)
        if m:
            n = int(m.group(1))
            if n > 0:
                return n

        # 第11競走
        m = re.search(r"第\s*(\d{1,2})\s*競走", s)
        if m:
            n = int(m.group(1))
            if n > 0:
                return n

        # 単独の1〜12数字だけならRとして扱う
        if re.fullmatch(r"\d{1,2}", s):
            n = int(s)
            if 1 <= n <= 12:
                return n

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

st.title("競馬ランクアプリ v12.9 Dabista Vertical SNS Save")
st.write("ランキング計算は1会場ずつ安全に行い、単複おすすめ1だけを保存して、最後に3会場まとめSNS画像を作成します。")
st.caption("v12.9: 前走頭数・前3角通過順・前4角通過順があれば、通過順位を相対位置に補正して評価します。")
st.caption("履歴ファイルは prev3c_relative_category_stats.csv / prev4c_relative_category_stats.csv を優先して読み込みます。")

if "saved_recs" not in st.session_state:
    st.session_state.saved_recs = []

uploaded = st.file_uploader("1会場分の予想CSVをアップロード", type=["csv"])

current_recs = []
if uploaded is None:
    st.info("まず1会場6レース分のCSVを読み込んでください。レース識別IDは 日付 + 場所 + R で作成します。")
    st.info("推奨追加列: 前走頭数, 前3角通過順, 前4角通過順。これらがあると総頭数補正が有効になります。")
else:
    prev3c_stat, prev4c_stat, prevtrack_stat, prev3c_file, prev4c_file = load_stat_defaults()
    df = prepare_race_df(read_csv_any(uploaded))

    st.caption(f"3角履歴ファイル: {prev3c_file} / 4角履歴ファイル: {prev4c_file}")

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

    tab1, tab2, tab3, tab4 = st.tabs(["ランキング", "おすすめ買い目", "保存・SNS画像", "通過順補正確認"])

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

    with tab4:
        st.subheader("通過順補正確認")
        st.caption("前走頭数・前3角通過順・前4角通過順がCSVにある場合、ここで総頭数補正後のカテゴリを確認できます。")
        check_cols = [
            "date", "レース", "horseNo", "horseName",
            "prevFieldSize", "prev3cPos", "prev4cPos",
            "prev3cRel_final", "prev4cRel_final",
            "prev3cCat_original", "prev4cCat_original",
            "prev3cCat", "prev4cCat",
            "3角履歴係数", "4角履歴係数",
            "通過順補正",
        ]
        show_df = df[check_cols].rename(columns={
            "date": "日付",
            "horseNo": "馬番",
            "horseName": "馬名",
            "prevFieldSize": "前走頭数",
            "prev3cPos": "前3角通過順",
            "prev4cPos": "前4角通過順",
            "prev3cRel_final": "前3角相対位置",
            "prev4cRel_final": "前4角相対位置",
            "prev3cCat_original": "前3角元カテゴリ",
            "prev4cCat_original": "前4角元カテゴリ",
            "prev3cCat": "前3角補正後カテゴリ",
            "prev4cCat": "前4角補正後カテゴリ",
        })
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        export_cols = [
            "date","場所","レース","raceName","horseNo","horseName",
            "prevFieldSize","prev3cPos","prev4cPos","prev3cRel_final","prev4cRel_final",
            "prev3cCat","prev4cCat","通過順補正",
            "相対評価","トータルランク","本体点","展開位置補正","前走場所直線補正","総合点",
        ]
        export_df = df[export_cols].rename(columns={
            "date":"日付",
            "raceName":"レース名",
            "horseNo":"馬番",
            "horseName":"馬名",
            "prevFieldSize":"前走頭数",
            "prev3cPos":"前3角通過順",
            "prev4cPos":"前4角通過順",
            "prev3cRel_final":"前3角相対位置",
            "prev4cRel_final":"前4角相対位置",
            "prev3cCat":"前3角位置カテゴリ",
            "prev4cCat":"前4角位置カテゴリ",
        })
        result_csv = export_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "補正確認付き予想結果CSVをダウンロード",
            data=result_csv.encode("utf-8-sig"),
            file_name="keiba_rank_v122_relative_results.csv",
            mime="text/csv",
        )
