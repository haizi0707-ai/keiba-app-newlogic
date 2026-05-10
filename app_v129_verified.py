# -*- coding: utf-8 -*-
import os
import re
import io
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="競馬ランクアプリ v12.8 Verified Axis", layout="centered")

BASE_DIR = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()

DEFAULT_FILES = {
    "prev3c": os.path.join(BASE_DIR, "prev3c_relative_category_stats.csv"),
    "prev3c_fallback": os.path.join(BASE_DIR, "prev3c_category_stats.csv"),
    "prev4c": os.path.join(BASE_DIR, "prev4c_relative_category_stats.csv"),
    "prev4c_fallback": os.path.join(BASE_DIR, "prev4c_category_stats.csv"),
    "prevtrack": os.path.join(BASE_DIR, "prevtrack_roi_stats.csv"),
}

EVAL_MAP = {"かなり向く": 1.25, "向く": 1.10, "普通": 1.00, "やや不向き": 0.85, "不向き": 0.70}

TRACK_ORDER = {
    "福島": 1, "新潟": 2, "東京": 3, "中山": 4, "中京": 5,
    "京都": 6, "阪神": 7, "小倉": 8, "札幌": 9, "函館": 10
}

PLACES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]


def resolve_file(primary, fallback=None):
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
        "東京競馬場": "東京", "中山競馬場": "中山", "中京競馬場": "中京", "阪神競馬場": "阪神",
        "京都競馬場": "京都", "新潟競馬場": "新潟", "福島競馬場": "福島", "小倉競馬場": "小倉",
        "札幌競馬場": "札幌", "函館競馬場": "函館",
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
    "date": ["date", "日付", "開催日", "年月日", "日付S"],
    "場所": ["場所", "track", "競馬場", "開催", "場名"],
    "raceNo": ["raceNo", "race_number", "R", "Ｒ", "レース番号", "race_no", "レースNo", "R番号"],
    "raceName": ["raceName", "race_name", "レース名"],
    "horseNo": ["horseNo", "horse_number", "馬番"],
    "horseName": ["horseName", "horse_name", "馬名"],
    "distance": ["distance", "距離", "距離数値"],
    "surface": ["芝ダ", "芝・ダ", "surface"],
    "raceLabel": ["レース", "race"],
    "prevTrack": ["前走競馬場", "前走場所", "prevTrack"],
    "prevSurface": ["前走芝ダ", "前芝・ダ", "prevSurface"],
    "prevDistance": ["前走距離数値", "前走距離", "前距離", "prevDistance"],
    "prevFieldSize": ["前走頭数", "前走出走頭数", "prevFieldSize", "fieldSize"],
    "prev3cPos": ["前3角通過順", "前走3角通過順", "前3角順位", "前3角", "前3角通過", "prev3cPos"],
    "prev4cPos": ["前4角通過順", "前走4角通過順", "前4角順位", "前4角", "前4角通過", "prev4cPos"],
    "prev3cRel": ["前3角相対位置", "前走3角相対位置", "prev3cRel"],
    "prev4cRel": ["前4角相対位置", "前走4角相対位置", "prev4cRel"],
    "prev3cCat": ["前3角位置カテゴリ", "前走3角カテゴリ", "prev3cCat"],
    "prev4cCat": ["前4角位置カテゴリ", "前走4角カテゴリ", "prev4cCat"],
    "prevStraight": ["前走直線ロジック点", "前走直線点", "prevStraight"],
    "prev2Straight": ["前々走直線ロジック点", "前々走直線点", "prev2Straight"],
    "paceEval": ["展開予想評価", "脚質展開評価", "paceEval"],
    "straightEval": ["直線相性評価", "場所直線相性評価", "straightEval"],
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
        "逃げ": "1番手", "先頭": "1番手", "1": "1番手", "1番": "1番手", "1番手": "1番手",
        "2-3": "2-3番手", "2-3番手": "2-3番手", "2番手": "2-3番手", "3番手": "2-3番手",
        "4-6": "4-6番手", "4-6番手": "4-6番手", "4番手": "4-6番手", "5番手": "4-6番手", "6番手": "4-6番手",
        "7-10": "7-10番手", "7-10番手": "7-10番手", "7番手": "7-10番手", "8番手": "7-10番手", "9番手": "7-10番手", "10番手": "7-10番手",
        "11番手以下": "11番手以下", "11以下": "11番手以下", "後方": "11番手以下", "最後方": "11番手以下",
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
    out = df.copy()
    out["prev3cCat_original"] = out["prev3cCat"].apply(normalize_existing_position_cat)
    out["prev4cCat_original"] = out["prev4cCat"].apply(normalize_existing_position_cat)
    out["prev3cRel_calc"] = out.apply(lambda r: calc_relative_position(r["prev3cPos"], r["prevFieldSize"]), axis=1)
    out["prev4cRel_calc"] = out.apply(lambda r: calc_relative_position(r["prev4cPos"], r["prevFieldSize"]), axis=1)
    out["prev3cRel_final"] = np.where(pd.notna(out["prev3cRel"]), out["prev3cRel"], out["prev3cRel_calc"])
    out["prev4cRel_final"] = np.where(pd.notna(out["prev4cRel"]), out["prev4cRel"], out["prev4cRel_calc"])
    out["prev3cCat_relative"] = out.apply(lambda r: rel_to_app_category(r["prev3cRel_final"], r["prev3cPos"]), axis=1)
    out["prev4cCat_relative"] = out.apply(lambda r: rel_to_app_category(r["prev4cRel_final"], r["prev4cPos"]), axis=1)
    out["prev3cCat"] = np.where(out["prev3cCat_relative"].astype(str).str.strip() != "", out["prev3cCat_relative"], out["prev3cCat_original"])
    out["prev4cCat"] = np.where(out["prev4cCat_relative"].astype(str).str.strip() != "", out["prev4cCat_relative"], out["prev4cCat_original"])
    out["通過順補正"] = np.where(pd.notna(out["prev3cRel_final"]) | pd.notna(out["prev4cRel_final"]), "総頭数補正あり", "CSVカテゴリ使用")
    return out


def safe_race_no(row):
    try:
        v = row.get("raceNo", np.nan)
        if pd.notna(v):
            n = int(float(v))
            if n > 0:
                return n
    except Exception:
        pass
    for key in ["レース", "raceLabel", "レース識別ID", "raceName", "R", "Ｒ", "raceNo"]:
        s = norm_text(row.get(key, ""))
        if not s:
            continue
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
    return 0


def prepare_race_df(df):
    df = rename_first_match(df, CANDS)
    for col in CANDS.keys():
        if col not in df.columns:
            df[col] = ""
    parsed = df["raceLabel"].apply(parse_race_label)
    df["場所"] = np.where(df["場所"].astype(str).str.strip() != "", df["場所"], parsed.apply(lambda x: x[0]))
    df["raceNo"] = np.where(df["raceNo"].astype(str).str.strip() != "", df["raceNo"], parsed.apply(lambda x: x[1]))
    df["raceNo"] = pd.to_numeric(df["raceNo"], errors="coerce")

    df["raceNo"] = df.apply(safe_race_no, axis=1)
    for col in ["場所", "prevTrack"]:
        df[col] = df[col].apply(norm_track)
    for col in ["surface", "prevSurface"]:
        df[col] = df[col].apply(norm_surface)
    for col in ["horseName", "raceName", "date", "prev3cCat", "prev4cCat", "paceEval", "straightEval"]:
        df[col] = df[col].apply(norm_text)
    num_cols = ["distance", "raceNo", "horseNo", "prevDistance", "prevStraight", "prev2Straight", "prevFieldSize", "prev3cPos", "prev4cPos", "prev3cRel", "prev4cRel"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["prevStraight"] = df["prevStraight"].fillna(50.0).clip(0, 100)
    df["prev2Straight"] = df["prev2Straight"].fillna(50.0).clip(0, 100)
    df = apply_relative_position_logic(df)
    df["距離表示"] = np.where(df["surface"].astype(str) != "", df["surface"] + df["distance"].fillna(0).astype(int).astype(str), "")
    df["レース"] = df.apply(lambda r: f"{r['場所']}{int(r['raceNo'])}R" if pd.notna(r["raceNo"]) and norm_text(r["場所"]) else norm_text(r["raceLabel"]), axis=1)
    df["レース識別ID"] = df.apply(lambda r: f"{r['date']}_{r['場所']}_{int(r['raceNo'])}R" if pd.notna(r["raceNo"]) else f"{r['date']}_{r['場所']}_{r['レース']}", axis=1)
    return df


# =========================================================
# 履歴係数
# =========================================================

def load_csv_if_exists(path):
    if path and os.path.exists(path):
        return read_csv_any(path)
    return pd.DataFrame()


def load_stat_defaults():
    prev3c_path = resolve_file(DEFAULT_FILES["prev3c"], DEFAULT_FILES["prev3c_fallback"])
    prev4c_path = resolve_file(DEFAULT_FILES["prev4c"], DEFAULT_FILES["prev4c_fallback"])
    prevtrack_path = DEFAULT_FILES["prevtrack"]

    prev3c = load_csv_if_exists(prev3c_path)
    prev4c = load_csv_if_exists(prev4c_path)
    prevtrack = load_csv_if_exists(prevtrack_path)

    if not prev3c.empty:
        prev3c = prev3c.rename(columns={"前走競馬場": "prevTrack", "前走芝ダ": "prevSurface", "前走距離数値": "prevDistance", "前3角位置カテゴリ": "prev3cCat", "件数": "count", "複勝率": "place_rate"})
        prev3c["prevTrack"] = prev3c["prevTrack"].apply(norm_track)
        prev3c["prevSurface"] = prev3c["prevSurface"].apply(norm_surface)
        prev3c["prevDistance"] = pd.to_numeric(prev3c["prevDistance"], errors="coerce")
        prev3c["count"] = pd.to_numeric(prev3c["count"], errors="coerce")
        prev3c["place_rate"] = pd.to_numeric(prev3c["place_rate"], errors="coerce")
        prev3c["prev3cCat"] = prev3c["prev3cCat"].apply(normalize_existing_position_cat)

    if not prev4c.empty:
        prev4c = prev4c.rename(columns={"前走競馬場": "prevTrack", "前走芝ダ": "prevSurface", "前走距離数値": "prevDistance", "前4角位置カテゴリ": "prev4cCat", "件数": "count", "複勝率": "place_rate"})
        prev4c["prevTrack"] = prev4c["prevTrack"].apply(norm_track)
        prev4c["prevSurface"] = prev4c["prevSurface"].apply(norm_surface)
        prev4c["prevDistance"] = pd.to_numeric(prev4c["prevDistance"], errors="coerce")
        prev4c["count"] = pd.to_numeric(prev4c["count"], errors="coerce")
        prev4c["place_rate"] = pd.to_numeric(prev4c["place_rate"], errors="coerce")
        prev4c["prev4cCat"] = prev4c["prev4cCat"].apply(normalize_existing_position_cat)

    if not prevtrack.empty:
        prevtrack = prevtrack.rename(columns={"競馬場": "場所", "芝ダ": "surface", "距離数値": "distance", "前走場所": "prevTrack", "件数": "count", "複勝率": "place_rate"})
        prevtrack["場所"] = prevtrack["場所"].apply(norm_track)
        prevtrack["surface"] = prevtrack["surface"].apply(norm_surface)
        prevtrack["distance"] = pd.to_numeric(prevtrack["distance"], errors="coerce")
        prevtrack["prevTrack"] = prevtrack["prevTrack"].apply(norm_track)
        prevtrack["count"] = pd.to_numeric(prevtrack["count"], errors="coerce")
        prevtrack["place_rate"] = pd.to_numeric(prevtrack["place_rate"], errors="coerce")

    return prev3c, prev4c, prevtrack, os.path.basename(str(prev3c_path)), os.path.basename(str(prev4c_path))


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
    if stat.empty:
        return 1.0
    sub = stat[(stat["prevTrack"] == row["prevTrack"]) & (stat["prevSurface"] == row["prevSurface"]) & (stat["prevDistance"] == row["prevDistance"])]
    if sub.empty:
        return 1.0
    hit = sub[sub["prev3cCat"] == row["prev3cCat"]]
    rate = hit["place_rate"].iloc[0] if not hit.empty else np.nan
    return map_rate_to_coef(rate, sub["place_rate"].min(), sub["place_rate"].max())


def hist_coef_prev4c(row, stat):
    if stat.empty:
        return 1.0
    sub = stat[(stat["prevTrack"] == row["prevTrack"]) & (stat["prevSurface"] == row["prevSurface"]) & (stat["prevDistance"] == row["prevDistance"])]
    if sub.empty:
        return 1.0
    hit = sub[sub["prev4cCat"] == row["prev4cCat"]]
    rate = hit["place_rate"].iloc[0] if not hit.empty else np.nan
    return map_rate_to_coef(rate, sub["place_rate"].min(), sub["place_rate"].max())


def hist_coef_prevtrack(row, stat):
    if stat.empty:
        return 1.0
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
        idx = out[out["レース識別ID"] == race_id].sort_values(["総合点", "horseNo"], ascending=[False, True]).index.tolist()
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


# =========================================================
# 検証条件クリア本命・相手
# =========================================================

def verified_axis_condition(row):
    change = norm_text(row.get("直線替わり", ""))
    content = norm_text(row.get("前走内容判定_v2", ""))
    p3 = normalize_existing_position_cat(row.get("prev3cCat", ""))
    p4 = normalize_existing_position_cat(row.get("prev4cCat", ""))

    # A: 長→短 × 先行粘り × 前4角2-3
    if change == "長→短" and content == "先行粘り" and p4 == "2-3番手":
        return True, "本命A:長→短×先行粘り×前4角2-3"

    # B: 長→短 × 前3/4角7-10
    if change == "長→短" and (p3 == "7-10番手" or p4 == "7-10番手"):
        return True, "本命B:長→短×前3/4角7-10"

    # C: 同タイプ × 好位差し届かず × 4-6 or 7-10
    if change == "同タイプ" and content == "好位差し届かず":
        if p3 in ["4-6番手", "7-10番手"] or p4 in ["4-6番手", "7-10番手"]:
            return True, "本命C:同タイプ×好位差し届かず×中団"

    return False, ""


def _horse_label(row):
    return f'{int(row["horseNo"])} {row["horseName"]}'


def _pair_text(honmei, mate):
    return f'{int(honmei["horseNo"])} - {int(mate["horseNo"])}'


def extract_opponent_marks(r):
    try:
        honmei_no = int(float(r.get("馬番", 0) or 0))
    except Exception:
        honmei_no = 0
    opponents = []
    for key in ["買い目1", "買い目2", "買い目3"]:
        text = norm_text(r.get(key, ""))
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


def recommend_for_race(g):
    g = g.sort_values(["総合点", "horseNo"], ascending=[False, True]).reset_index(drop=True)
    honmei = g.iloc[0]
    cond_ok, cond_reason = verified_axis_condition(honmei)

    conf = confidence_from_score(honmei["総合点"])
    short_comment = f'本体{honmei["本体点"]:.1f}×展開{honmei["展開位置補正"]:.2f}×場所{honmei["前走場所直線補正"]:.2f}'

    base = {"honmei": honmei, "confidence": conf, "short_comment": short_comment}

    if not cond_ok:
        return {
            **base,
            "status": "見送り",
            "honmei_type": "検証条件外",
            "bet_strength": "見送り",
            "bet_type": "見送り",
            "bets": [],
            "mates": [],
            "reason": "過去5年検証条件に未該当",
        }

    mates = []
    for i in range(1, min(4, len(g))):
        mates.append(g.iloc[i])

    bets = []
    roles = ["○相手1", "▲相手2", "△相手3"]
    for i, mate in enumerate(mates):
        bets.append((_pair_text(honmei, mate), roles[i]))

    reason = cond_reason
    if mates:
        reason += "\n相手はレース内予想順位2〜4位を基本採用。"

    return {
        **base,
        "status": "買い対象",
        "honmei_type": "検証条件クリア",
        "bet_strength": "通常",
        "bet_type": "相手○▲△",
        "bets": bets,
        "mates": mates,
        "reason": reason,
    }




# =========================================================
# ランキングカード表示
# =========================================================

def render_rank_cards(g):
    """
    ランキングタブ用カード表示。
    Streamlit Cloudでも落ちないように、components.htmlを使わず
    st.markdown + st.dataframeで安定表示する。
    """
    if g is None or len(g) == 0:
        st.info("表示対象がありません。")
        return

    race_title = f'{g.iloc[0].get("date", "")} {g.iloc[0].get("レース", "")}'
    race_name = norm_text(g.iloc[0].get("raceName", ""))
    distance = norm_text(g.iloc[0].get("距離表示", ""))

    st.markdown(f"### {race_title}　{race_name}")
    if distance:
        st.caption(distance)

    show_cols = []
    col_map = {
        "horseNo": "馬番",
        "horseName": "馬名",
        "総合点": "総合点",
        "トータルランク": "ランク",
        "相対評価": "相対",
        "本体点": "本体",
        "展開位置補正": "展開補正",
        "前走場所直線補正": "直線補正",
        "prev3cCat": "前3角",
        "prev4cCat": "前4角",
        "paceEval": "展開評価",
        "straightEval": "直線評価",
    }

    for c in col_map:
        if c in g.columns:
            show_cols.append(c)

    if not show_cols:
        st.dataframe(g, use_container_width=True, hide_index=True)
        return

    view = g[show_cols].copy()
    view = view.rename(columns=col_map)

    for c in ["馬番"]:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0).astype(int)

    for c in ["総合点", "本体", "展開補正", "直線補正"]:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").round(2)

    st.dataframe(view, use_container_width=True, hide_index=True)


# =========================================================
# 画像
# =========================================================

def get_font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\meiryob.ttc" if bold else r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\YuGothB.ttc" if bold else r"C:\Windows\Fonts\YuGothM.ttc",
        r"C:\Windows\Fonts\msgothic.ttc",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_centered_text(draw, box, text, font, fill):
    x1, y1, x2, y2 = box
    bbox = draw.textbbox((0, 0), str(text), font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = x1 + (x2 - x1 - tw) / 2 - bbox[0]
    y = y1 + (y2 - y1 - th) / 2 - bbox[1]
    draw.text((x, y), str(text), font=font, fill=fill)


def draw_fit_text(draw, xy, text, font, fill, max_width):
    x, y = xy
    t = str(text)
    while len(t) > 0:
        bbox = draw.textbbox((x, y), t, font=font)
        if bbox[2] - bbox[0] <= max_width:
            break
        t = t[:-1]
    if t != str(text):
        t = t[:-1] + "…"
    draw.text((x, y), t, font=font, fill=fill)


def make_sns_image(saved):
    items = []
    for r in saved:
        try:
            if int(r.get("買い対象", 0)) == 1:
                items.append(dict(r))
        except Exception:
            pass

    if not items:
        return None

    def clean_item(r):
        rr = dict(r)
        place_raw = norm_text(rr.get("場所", ""))
        race_raw = norm_text(rr.get("レース", ""))
        single_raw = norm_text(rr.get("単複おすすめ1", ""))
        m = re.search(r"(福島|新潟|東京|中山|中京|京都|阪神|小倉|札幌|函館)\s*(\d+)\s*R", place_raw) or re.search(r"(福島|新潟|東京|中山|中京|京都|阪神|小倉|札幌|函館)\s*(\d+)\s*R", race_raw)
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
        rr["馬名"] = norm_text(rr.get("馬名", "")) or re.sub(r"^\d+\s*", "", single_raw)
        rr["相手表示"] = extract_opponent_marks(rr)
        return rr

    items = [clean_item(r) for r in items]
    items = sorted(items, key=lambda r: (TRACK_ORDER.get(norm_track(r.get("場所", "")), 99), int(float(r.get("R", 0) or 0)), int(float(r.get("馬番", 0) or 0))))

    raw_date = str(items[0].get("日付", ""))
    dt = pd.to_datetime(raw_date, errors="coerce")
    date_main = raw_date.replace("/", ".").replace("-", ".") if pd.isna(dt) else dt.strftime("%Y.%m.%d")
    weekday = "" if pd.isna(dt) else dt.strftime("%A").upper()

    W = 1080
    row_y = 360
    row_h = 155
    bottom_margin = 125
    H = max(1920, row_y + len(items) * row_h + bottom_margin)

    img = Image.new("RGB", (W, H), (10, 14, 22))
    draw = ImageDraw.Draw(img)

    white = (248, 248, 244)
    gold = (229, 191, 72)
    muted = (145, 149, 160)
    line = (42, 47, 57)
    circle_line = (48, 40, 28)
    bg = (10, 14, 22)

    for offset, width in [(0, 2), (55, 2)]:
        draw.ellipse((680 + offset, -120 + offset, 1260 - offset, 460 - offset), outline=circle_line, width=width)

    small_eng_font = get_font(25, True)
    title_font = get_font(70, True)
    date_font = get_font(40, False)
    weekday_font = get_font(22, True)
    place_font = get_font(31, True)
    race_font = get_font(36, True)
    horse_no_font = get_font(42, True)
    horse_font = get_font(45, True)
    mate_font = get_font(25, False)

    draw.text((80, 92), "T O D A Y ' S   P I C K S", font=small_eng_font, fill=gold)
    draw.text((80, 150), "本日の推奨馬", font=title_font, fill=white)
    draw.text((80, 245), "過去5年検証条件クリア", font=get_font(31, True), fill=gold)
    draw.text((670, 135), date_main, font=date_font, fill=gold)
    if weekday:
        draw.text((780, 190), weekday, font=weekday_font, fill=muted)

    draw.line((80, 310, 1000, 310), fill=line, width=2)

    y_start = 370
    for i, r in enumerate(items):
        y = y_start + i * row_h
        if i > 0:
            draw.line((80, y - 42, 1000, y - 42), fill=(31, 36, 45), width=2)

        place = norm_track(r.get("場所", ""))
        race_no = int(float(r.get("R", 0) or 0))
        horse_no = int(float(r.get("馬番", 0) or 0))
        horse_name = norm_text(r.get("馬名", ""))
        mate_text = norm_text(r.get("相手表示", ""))

        draw.text((80, y - 6), place, font=place_font, fill=gold)
        draw.text((80, y + 34), f"{race_no}R", font=race_font, fill=gold)

        cx, cy, rad = 250, y + 43, 42
        draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=gold)
        draw_centered_text(draw, (cx - rad, cy - rad, cx + rad, cy + rad), str(horse_no), horse_no_font, bg)

        draw_fit_text(draw, (340, y - 6), horse_name, horse_font, white, 620)
        if mate_text:
            mate_text_spaced = mate_text.replace("◯", "○ ").replace("▲", "▲ ").replace("△", "△ ")
            draw.text((340, y + 58), mate_text_spaced, font=mate_font, fill=muted)

    draw.line((80, H - 110, 1000, H - 110), fill=(31, 36, 45), width=2)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    bio.seek(0)
    return bio


def add_saved_recs(new_recs):
    if "saved_recs" not in st.session_state:
        st.session_state.saved_recs = []
    store = {}
    for r in st.session_state.saved_recs:
        rr = dict(r)
        store[f'{rr.get("日付","")}_{rr.get("場所","")}_{rr.get("R","")}'] = rr
    for r in new_recs:
        rr = dict(r)
        store[f'{rr.get("日付","")}_{rr.get("場所","")}_{rr.get("R","")}'] = rr
    st.session_state.saved_recs = list(store.values())


def saved_df():
    if "saved_recs" not in st.session_state:
        st.session_state.saved_recs = []
    return pd.DataFrame(st.session_state.saved_recs)


# =========================================================
# 画面
# =========================================================

st.title("競馬ランクアプリ v12.8 Verified Axis")
st.write("JV-Linkで作成したv12.2用CSVを読み込み、過去5年検証条件クリア馬を推奨馬として表示します。")
st.caption("信頼度90%以上ではなく、直線替わり×前走内容×前4角位置などの検証条件を優先します。")

if "saved_recs" not in st.session_state:
    st.session_state.saved_recs = []

uploaded = st.file_uploader("1会場分または複数レース分の予想CSVをアップロード", type=["csv"])
current_recs = []

if uploaded is None:
    st.info("CSVをアップロードしてください。")
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
            g = df[df["レース識別ID"] == race_id].sort_values(["総合点", "horseNo"], ascending=[False, True]).reset_index(drop=True)
            render_rank_cards(g)
            st.divider()

    with tab2:
        for race_id in df["レース識別ID"].unique():
            g = df[df["レース識別ID"] == race_id].sort_values(["総合点", "horseNo"], ascending=[False, True]).reset_index(drop=True)
            st.subheader(f'{g.iloc[0]["date"]} {g.iloc[0]["レース"]} {g.iloc[0]["raceName"]}')

            rec = recommend_for_race(g)
            honmei = rec["honmei"]
            conf = confidence_from_score(honmei["総合点"])
            short_comment = rec["short_comment"]

            st.markdown("### 単複おすすめ1")
            st.write(f'候補: {int(honmei["horseNo"])} {honmei["horseName"]}')
            st.caption(f'トータル{honmei["トータルランク"]} / 総合点 {honmei["総合点"]:.2f} / 参考信頼度 {conf:.2f}% / {short_comment}')

            st.markdown("### おすすめ馬券")
            if rec["status"] == "見送り":
                st.write("見送り")
                st.caption(f'理由：{rec["reason"]}')
            else:
                st.write(f'本命タイプ：{rec["honmei_type"]}')
                st.write(f'おすすめ馬券：{rec["bet_type"]}')
                for bet, role in rec.get("bets", []):
                    st.write(f"{bet}　{role}")
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
                "短評": short_comment,
                "買い対象": 1 if rec["status"] == "買い対象" else 0,
                "本命タイプ": rec["honmei_type"],
                "勝負度": rec.get("bet_strength", "通常"),
                "おすすめ馬券": rec["bet_type"],
                "買い目1": bet_values[0] if len(bet_values) > 0 else "",
                "買い目2": bet_values[1] if len(bet_values) > 1 else "",
                "買い目3": bet_values[2] if len(bet_values) > 2 else "",
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
            st.dataframe(sdf.sort_values(["日付", "場所", "R"]), use_container_width=True, hide_index=True)
            csv_text = sdf.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("保存済み推奨馬CSVをダウンロード", data=csv_text.encode("utf-8-sig"), file_name="saved_recommendations.csv", mime="text/csv")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("3会場まとめSNS画像を作成"):
                img = make_sns_image(st.session_state.saved_recs)
                if img is None:
                    st.warning("検証条件クリアの推奨馬はありません")
                else:
                    st.image(img, caption="SNS投稿用画像", use_container_width=True)
                    st.download_button("SNS画像PNGをダウンロード", data=img.getvalue(), file_name="sns_recommendations.png", mime="image/png")

        with col2:
            if st.button("保存済み推奨馬をクリア"):
                st.session_state.saved_recs = []
                st.success("保存済み推奨馬をクリアしました。")

    with tab4:
        st.subheader("通過順補正確認")
        check_cols = [
            "date", "レース", "horseNo", "horseName",
            "prevFieldSize", "prev3cPos", "prev4cPos",
            "prev3cRel_final", "prev4cRel_final",
            "prev3cCat_original", "prev4cCat_original",
            "prev3cCat", "prev4cCat",
            "3角履歴係数", "4角履歴係数", "通過順補正"
        ]
        show_cols = [c for c in check_cols if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

        result_csv = df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("補正確認付き予想結果CSVをダウンロード", data=result_csv.encode("utf-8-sig"), file_name="keiba_rank_v128_verified_results.csv", mime="text/csv")
