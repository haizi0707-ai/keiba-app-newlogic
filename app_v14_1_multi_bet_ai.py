
# -*- coding: utf-8 -*-
import os
import re
import io
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title='競馬ランクアプリ v14.1 Multi Bet AI', layout='centered')

BASE_DIR = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()

DEFAULT_FILES = {
    'prev3c': os.path.join(BASE_DIR, 'prev3c_relative_category_stats.csv'),
    'prev3c_fallback': os.path.join(BASE_DIR, 'prev3c_category_stats.csv'),
    'prev4c': os.path.join(BASE_DIR, 'prev4c_relative_category_stats.csv'),
    'prev4c_fallback': os.path.join(BASE_DIR, 'prev4c_category_stats.csv'),
    'prevtrack': os.path.join(BASE_DIR, 'prevtrack_roi_stats.csv'),
}

EVAL_MAP = {'かなり向く': 1.25, '向く': 1.10, '普通': 1.00, 'やや不向き': 0.85, '不向き': 0.70}
RANK_POINT = {'S': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}
VALID_POS = {'1番手', '2-3番手', '4-6番手', '7-10番手', '11番手以下'}


def norm_text(v):
    if pd.isna(v):
        return ''
    return ' '.join(unicodedata.normalize('NFKC', str(v)).strip().split())


def norm_track(v):
    s = norm_text(v)
    m = {
        '東京競馬場': '東京', '中山競馬場': '中山', '中京競馬場': '中京',
        '阪神競馬場': '阪神', '京都競馬場': '京都', '新潟競馬場': '新潟',
        '福島競馬場': '福島', '小倉競馬場': '小倉', '札幌競馬場': '札幌',
        '函館競馬場': '函館',
    }
    return m.get(s, s)


def norm_surface(v):
    s = norm_text(v)
    if s.startswith('芝'):
        return '芝'
    if s.startswith('ダ') or s == 'ダート':
        return 'ダ'
    return s


def read_csv_any(file_or_path):
    last = None
    for enc in ['utf-8-sig', 'cp932', 'shift_jis', 'utf-8']:
        try:
            if hasattr(file_or_path, 'seek'):
                file_or_path.seek(0)
            return pd.read_csv(file_or_path, encoding=enc)
        except Exception as e:
            last = e
    raise last


def resolve_file(primary, fallback=None):
    if primary and os.path.exists(primary):
        return primary
    if fallback and os.path.exists(fallback):
        return fallback
    return primary


def parse_race_label(v):
    s = norm_text(v)
    m = re.search(r'(東京|中山|中京|阪神|京都|新潟|福島|小倉|札幌|函館)\s*(\d+)\s*R', s)
    if not m:
        return '', np.nan
    return m.group(1), float(m.group(2))


CANDS = {
    'date': ['date', '日付', '開催日', '年月日', '日付S'],
    '場所': ['場所', 'track', '競馬場', '開催', '場名'],
    'raceNo': ['raceNo', 'race_number', 'R', 'Ｒ', 'レース番号', 'race_no', 'レースNo', 'R番号'],
    'raceName': ['raceName', 'race_name', 'レース名'],
    'horseNo': ['horseNo', 'horse_number', '馬番'],
    'horseName': ['horseName', 'horse_name', '馬名'],
    'distance': ['distance', '距離', '距離数値'],
    'surface': ['芝ダ', '芝・ダ', 'surface'],
    'raceLabel': ['レース', 'race'],
    'prevTrack': ['前走競馬場', '前走場所', 'prevTrack'],
    'prevSurface': ['前走芝ダ', '前芝・ダ', 'prevSurface'],
    'prevDistance': ['前走距離数値', '前走距離', '前距離', 'prevDistance'],
    'prevFieldSize': ['前走頭数', '前走出走頭数', 'prevFieldSize', 'fieldSize'],
    'prev3cPos': ['前3角通過順', '前走3角通過順', '前3角順位', '前3角', '前3角通過', 'prev3cPos'],
    'prev4cPos': ['前4角通過順', '前走4角通過順', '前4角順位', '前4角', '前4角通過', 'prev4cPos'],
    'prev3cRel': ['前3角相対位置', '前走3角相対位置', 'prev3cRel'],
    'prev4cRel': ['前4角相対位置', '前走4角相対位置', 'prev4cRel'],
    'prev3cCat': ['前3角位置カテゴリ', '前走3角カテゴリ', 'prev3cCat'],
    'prev4cCat': ['前4角位置カテゴリ', '前走4角カテゴリ', 'prev4cCat'],
    'prevStraight': ['前走直線ロジック点', '前走直線点', 'prevStraight'],
    'prev2Straight': ['前々走直線ロジック点', '前々走直線点', 'prev2Straight'],
    'paceEval': ['展開予想評価', '脚質展開評価', 'paceEval'],
    'straightEval': ['直線相性評価', '場所直線相性評価', 'straightEval'],
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
    s = s.replace('２', '2').replace('３', '3').replace('４', '4').replace('６', '6')
    s = s.replace('７', '7').replace('１０', '10').replace('１１', '11')
    s = s.replace('~', '-').replace('〜', '-').replace('－', '-')
    aliases = {
        '逃げ': '1番手', '先頭': '1番手', '1': '1番手', '1番': '1番手', '1番手': '1番手',
        '2-3': '2-3番手', '2-3番手': '2-3番手', '2番手': '2-3番手', '3番手': '2-3番手',
        '4-6': '4-6番手', '4-6番手': '4-6番手', '4番手': '4-6番手', '5番手': '4-6番手', '6番手': '4-6番手',
        '7-10': '7-10番手', '7-10番手': '7-10番手', '7番手': '7-10番手', '8番手': '7-10番手', '9番手': '7-10番手', '10番手': '7-10番手',
        '11番手以下': '11番手以下', '11以下': '11番手以下', '後方': '11番手以下', '最後方': '11番手以下',
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
                return '1番手'
        except Exception:
            pass
    if pd.isna(rel):
        return ''
    try:
        rel = float(rel)
    except Exception:
        return ''
    if rel <= 0.25:
        return '2-3番手'
    if rel <= 0.45:
        return '4-6番手'
    if rel <= 0.85:
        return '7-10番手'
    return '11番手以下'


def apply_relative_position_logic(df):
    out = df.copy()
    out['prev3cCat_original'] = out['prev3cCat'].apply(normalize_existing_position_cat)
    out['prev4cCat_original'] = out['prev4cCat'].apply(normalize_existing_position_cat)
    out['prev3cRel_calc'] = out.apply(lambda r: calc_relative_position(r['prev3cPos'], r['prevFieldSize']), axis=1)
    out['prev4cRel_calc'] = out.apply(lambda r: calc_relative_position(r['prev4cPos'], r['prevFieldSize']), axis=1)
    out['prev3cRel_final'] = np.where(pd.notna(out['prev3cRel']), out['prev3cRel'], out['prev3cRel_calc'])
    out['prev4cRel_final'] = np.where(pd.notna(out['prev4cRel']), out['prev4cRel'], out['prev4cRel_calc'])
    out['prev3cCat_relative'] = out.apply(lambda r: rel_to_app_category(r['prev3cRel_final'], r['prev3cPos']), axis=1)
    out['prev4cCat_relative'] = out.apply(lambda r: rel_to_app_category(r['prev4cRel_final'], r['prev4cPos']), axis=1)
    out['prev3cCat'] = np.where(out['prev3cCat_relative'].astype(str).str.strip() != '', out['prev3cCat_relative'], out['prev3cCat_original'])
    out['prev4cCat'] = np.where(out['prev4cCat_relative'].astype(str).str.strip() != '', out['prev4cCat_relative'], out['prev4cCat_original'])
    out['通過順補正'] = np.where(pd.notna(out['prev3cRel_final']) | pd.notna(out['prev4cRel_final']), '総頭数補正あり', 'CSVカテゴリ使用')
    return out


def prepare_race_df(df):
    df = rename_first_match(df, CANDS)
    for col in CANDS.keys():
        if col not in df.columns:
            df[col] = ''

    parsed = df['raceLabel'].apply(parse_race_label)
    df['場所'] = np.where(df['場所'].astype(str).str.strip() != '', df['場所'], parsed.apply(lambda x: x[0]))
    df['raceNo'] = np.where(df['raceNo'].astype(str).str.strip() != '', df['raceNo'], parsed.apply(lambda x: x[1]))
    df['raceNo'] = pd.to_numeric(df['raceNo'], errors='coerce')

    def restore_race_no(row):
        try:
            v = row.get('raceNo', np.nan)
            if pd.notna(v) and int(float(v)) > 0:
                return int(float(v))
        except Exception:
            pass
        for key in ['raceLabel', 'レース', 'raceName']:
            s = norm_text(row.get(key, ''))
            m = re.search(r'(\d{1,2})\s*R', s, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
        return np.nan

    df['raceNo'] = df.apply(restore_race_no, axis=1)

    for col in ['場所', 'prevTrack']:
        df[col] = df[col].apply(norm_track)
    for col in ['surface', 'prevSurface']:
        df[col] = df[col].apply(norm_surface)
    for col in ['horseName', 'raceName', 'date', 'prev3cCat', 'prev4cCat', 'paceEval', 'straightEval']:
        df[col] = df[col].apply(norm_text)

    num_cols = ['distance', 'raceNo', 'horseNo', 'prevDistance', 'prevStraight', 'prev2Straight', 'prevFieldSize', 'prev3cPos', 'prev4cPos', 'prev3cRel', 'prev4cRel']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['prevStraight'] = df['prevStraight'].fillna(50.0).clip(0, 100)
    df['prev2Straight'] = df['prev2Straight'].fillna(50.0).clip(0, 100)
    df = apply_relative_position_logic(df)

    df['距離表示'] = np.where(df['surface'].astype(str) != '', df['surface'] + df['distance'].fillna(0).astype(int).astype(str), '')
    df['レース'] = df.apply(lambda r: f'{r["場所"]}{int(r["raceNo"])}R' if pd.notna(r['raceNo']) and norm_text(r['場所']) else norm_text(r['raceLabel']), axis=1)
    df['レース識別ID'] = df.apply(lambda r: f'{r["date"]}_{r["場所"]}_{int(r["raceNo"])}R' if pd.notna(r['raceNo']) else f'{r["date"]}_{r["場所"]}_{r["レース"]}', axis=1)
    return df


def load_stat_defaults():
    prev3c_path = resolve_file(DEFAULT_FILES['prev3c'], DEFAULT_FILES['prev3c_fallback'])
    prev4c_path = resolve_file(DEFAULT_FILES['prev4c'], DEFAULT_FILES['prev4c_fallback'])
    prevtrack_path = DEFAULT_FILES['prevtrack']

    prev3c = read_csv_any(prev3c_path)
    prev4c = read_csv_any(prev4c_path)
    prevtrack = read_csv_any(prevtrack_path)

    prev3c = prev3c.rename(columns={'前走競馬場': 'prevTrack', '前走芝ダ': 'prevSurface', '前走距離数値': 'prevDistance', '前3角位置カテゴリ': 'prev3cCat', '件数': 'count', '複勝率': 'place_rate'})
    prev4c = prev4c.rename(columns={'前走競馬場': 'prevTrack', '前走芝ダ': 'prevSurface', '前走距離数値': 'prevDistance', '前4角位置カテゴリ': 'prev4cCat', '件数': 'count', '複勝率': 'place_rate'})
    prevtrack = prevtrack.rename(columns={'競馬場': '場所', '芝ダ': 'surface', '距離数値': 'distance', '前走場所': 'prevTrack', '件数': 'count', '複勝率': 'place_rate'})

    for d in [prev3c, prev4c]:
        d['prevTrack'] = d['prevTrack'].apply(norm_track)
        d['prevSurface'] = d['prevSurface'].apply(norm_surface)
        d['prevDistance'] = pd.to_numeric(d['prevDistance'], errors='coerce')
        d['count'] = pd.to_numeric(d['count'], errors='coerce')
        d['place_rate'] = pd.to_numeric(d['place_rate'], errors='coerce')

    prev3c['prev3cCat'] = prev3c['prev3cCat'].apply(normalize_existing_position_cat)
    prev4c['prev4cCat'] = prev4c['prev4cCat'].apply(normalize_existing_position_cat)

    prevtrack['場所'] = prevtrack['場所'].apply(norm_track)
    prevtrack['surface'] = prevtrack['surface'].apply(norm_surface)
    prevtrack['distance'] = pd.to_numeric(prevtrack['distance'], errors='coerce')
    prevtrack['prevTrack'] = prevtrack['prevTrack'].apply(norm_track)
    prevtrack['count'] = pd.to_numeric(prevtrack['count'], errors='coerce')
    prevtrack['place_rate'] = pd.to_numeric(prevtrack['place_rate'], errors='coerce')
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
    sub = stat[(stat['prevTrack'] == row['prevTrack']) & (stat['prevSurface'] == row['prevSurface']) & (stat['prevDistance'] == row['prevDistance'])]
    if sub.empty:
        return 1.0
    hit = sub[sub['prev3cCat'] == row['prev3cCat']]
    rate = hit['place_rate'].iloc[0] if not hit.empty else np.nan
    return map_rate_to_coef(rate, sub['place_rate'].min(), sub['place_rate'].max())


def hist_coef_prev4c(row, stat):
    sub = stat[(stat['prevTrack'] == row['prevTrack']) & (stat['prevSurface'] == row['prevSurface']) & (stat['prevDistance'] == row['prevDistance'])]
    if sub.empty:
        return 1.0
    hit = sub[sub['prev4cCat'] == row['prev4cCat']]
    rate = hit['place_rate'].iloc[0] if not hit.empty else np.nan
    return map_rate_to_coef(rate, sub['place_rate'].min(), sub['place_rate'].max())


def hist_coef_prevtrack(row, stat):
    sub = stat[(stat['場所'] == row['場所']) & (stat['surface'] == row['surface']) & (stat['distance'] == row['distance'])]
    if sub.empty:
        return 1.0
    hit = sub[sub['prevTrack'] == row['prevTrack']]
    rate = hit['place_rate'].iloc[0] if not hit.empty else np.nan
    return map_rate_to_coef(rate, sub['place_rate'].min(), sub['place_rate'].max())


def confidence_from_score(score):
    return float(np.clip(20 + score * 1.2, 5, 95))


def total_rank(score):
    if pd.isna(score):
        return ''
    if score >= 65:
        return 'S'
    if score >= 56:
        return 'A'
    if score >= 47:
        return 'B'
    if score >= 38:
        return 'C'
    return 'D'


def assign_relative_ranks(df):
    out = df.copy()
    out['相対評価'] = ''
    for race_id in out['レース識別ID'].unique():
        idx = out[out['レース識別ID'] == race_id].sort_values(['総合点', 'horseNo'], ascending=[False, True]).index.tolist()
        n = len(idx)
        if n == 0:
            continue
        s_n = max(1, round(n * 0.10))
        a_n = max(1, round(n * 0.20))
        b_n = max(1, round(n * 0.25))
        tail_n = max(1, round(n * 0.15))
        for i, ix in enumerate(idx):
            if i < s_n:
                out.at[ix, '相対評価'] = 'S'
            elif i < s_n + a_n:
                out.at[ix, '相対評価'] = 'A'
            elif i < s_n + a_n + b_n:
                out.at[ix, '相対評価'] = 'B'
            elif i < n - tail_n:
                out.at[ix, '相対評価'] = 'C'
            else:
                out.at[ix, '相対評価'] = 'D'
    return out


def _rank_score(v):
    return RANK_POINT.get(str(v), 0)


def _eval_score(v):
    return {'かなり向く': 5, '向く': 4, '普通': 3, 'やや不向き': 2, '不向き': 1}.get(norm_text(v), 3)


def _position_score(v):
    s = normalize_existing_position_cat(v)
    return {'1番手': 1, '2-3番手': 2, '4-6番手': 3, '7-10番手': 4, '11番手以下': 5}.get(s, 3)


def _horse_label(row):
    return f'{int(row["horseNo"])} {row["horseName"]}'


def _pair_text(honmei, mate):
    return f'{int(honmei["horseNo"])} - {int(mate["horseNo"])}'


def _trio_text(honmei, a, b):
    return f'{int(honmei["horseNo"])} - {int(a["horseNo"])} - {int(b["horseNo"])}'


def _comment_text(row):
    vals = []
    for k in ['短評', 'comment', 'コメント', '評価コメント']:
        if k in row.index:
            vals.append(norm_text(row.get(k, '')))
    return ' '.join([v for v in vals if v])


def _keyword_count(text_value, words):
    s = norm_text(text_value)
    return sum(1 for w in words if w in s)


def judge_honmei_type(honmei, conf):
    rank = str(honmei.get('トータルランク', ''))
    score = float(honmei.get('総合点', 0) or 0)
    body = float(honmei.get('本体点', 0) or 0)
    pace = float(honmei.get('展開位置補正', 1) or 1)
    place = float(honmei.get('前走場所直線補正', 1) or 1)
    pace_eval = _eval_score(honmei.get('paceEval', '普通'))
    straight_eval = _eval_score(honmei.get('straightEval', '普通'))
    pos4 = _position_score(honmei.get('prev4cCat', ''))
    comment = _comment_text(honmei)

    win_words = ['勝ち切り', '押し切り', '主役', '軸上位', '能力上位', '前進', '頭まで', '連軸', '好位', '先行', '安定']
    third_words = ['堅実', '相手向き', '複勝向き', '3着候補', '差し届けば', '展開待ち', '取りこぼし', '詰め甘い', '善戦型']

    win_points = 0
    third_points = 0
    if rank in ['S', 'A']:
        win_points += 2
    elif rank == 'B':
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
        return '3着型', f'勝ち切りより3着内安定寄り（判定 {win_points}-{third_points}）'
    return '1・2着型', f'1〜2着に来るイメージを優先（判定 {win_points}-{third_points}）'


def _prepare_candidates(g, honmei):
    cand = g[g.index != 0].copy()
    if cand.empty:
        return cand

    h_pace = _eval_score(honmei.get('paceEval', '普通'))
    h_straight = _eval_score(honmei.get('straightEval', '普通'))
    h_3c = _position_score(honmei.get('prev3cCat', ''))
    h_4c = _position_score(honmei.get('prev4cCat', ''))

    cand['rank_score'] = cand['トータルランク'].apply(_rank_score)
    cand['relative_score'] = cand['相対評価'].apply(_rank_score)
    cand['pace_score'] = cand['paceEval'].apply(_eval_score)
    cand['straight_score'] = cand['straightEval'].apply(_eval_score)
    cand['pos3_score'] = cand['prev3cCat'].apply(_position_score)
    cand['pos4_score'] = cand['prev4cCat'].apply(_position_score)

    cand['same_gap'] = (
        (cand['pace_score'] - h_pace).abs()
        + (cand['straight_score'] - h_straight).abs() * 0.5
        + (cand['pos3_score'] - h_3c).abs() * 0.7
        + (cand['pos4_score'] - h_4c).abs() * 0.9
    )
    cand['same_score'] = cand['rank_score'] * 18 + cand['relative_score'] * 6 + cand['総合点'] * 0.70 - cand['same_gap'] * 10

    cand['diff_gap'] = (
        (cand['pace_score'] - h_pace).abs()
        + (cand['pos3_score'] - h_3c).abs() * 0.8
        + (cand['pos4_score'] - h_4c).abs() * 1.0
    )
    cand['comp_fit'] = cand['diff_gap'].apply(lambda x: 14 if 1.0 <= x <= 3.5 else (7 if x > 0 else 0))
    cand['comp_score'] = cand['comp_fit'] + cand['rank_score'] * 12 + cand['総合点'] * 0.55 + cand['展開位置補正'] * 8

    cand['return_gap'] = (
        (cand['pace_score'] - h_pace).abs()
        + (cand['straight_score'] - h_straight).abs()
        + (cand['pos3_score'] - h_3c).abs() * 0.8
        + (cand['pos4_score'] - h_4c).abs() * 0.8
    )
    cand['plus_comment'] = cand.apply(lambda r: 1 if any(k in _comment_text(r) for k in ['向く', '上積', '先行', '差し', '外', '内', '粘', '伸', '好位', '妙味', '穴']) else 0, axis=1)
    cand['return_score'] = (
        cand['return_gap'].clip(upper=5) * 5
        + cand['rank_score'] * 9
        + cand['relative_score'] * 4
        + cand['総合点'] * 0.45
        + cand['plus_comment'] * 8
    )
    return cand


def _pick_same(cand, used):
    pool = cand[~cand['horseNo'].isin(used)].copy()
    pool = pool[pool['トータルランク'].isin(['S', 'A', 'B'])].copy()
    if pool.empty:
        pool = cand[(~cand['horseNo'].isin(used)) & (cand['トータルランク'].astype(str) != 'D')].copy()
    if pool.empty:
        return None
    return pool.sort_values(['same_score', '総合点', 'horseNo'], ascending=[False, False, True]).iloc[0]


def _pick_comp(cand, used):
    pool = cand[~cand['horseNo'].isin(used)].copy()
    pool = pool[pool['トータルランク'].isin(['S', 'A', 'B', 'C'])].copy()
    pool = pool[pool['総合点'] >= 35].copy()
    if pool.empty:
        pool = cand[(~cand['horseNo'].isin(used)) & (cand['トータルランク'].astype(str) != 'D')].copy()
    if pool.empty:
        return None
    return pool.sort_values(['comp_score', '総合点', 'horseNo'], ascending=[False, False, True]).iloc[0]


def _pick_return(cand, used, allow_d=True):
    pool = cand[~cand['horseNo'].isin(used)].copy()
    if allow_d:
        non_d = pool[pool['トータルランク'].astype(str) != 'D'].copy()
        d_pool = pool[(pool['トータルランク'].astype(str) == 'D') & (pool['plus_comment'] == 1) & (pool['総合点'] >= 32)].copy()
        pool = pd.concat([non_d, d_pool], ignore_index=False)
    else:
        pool = pool[pool['トータルランク'].astype(str) != 'D']
    pool = pool[pool['総合点'] >= 32].copy()
    if pool.empty:
        pool = cand[(~cand['horseNo'].isin(used)) & (cand['トータルランク'].astype(str) != 'D')].copy()
    if pool.empty:
        return None
    return pool.sort_values(['return_score', '総合点', 'horseNo'], ascending=[False, False, True]).iloc[0]


# =========================
# v12.8 新ロジック: 馬連△ = 2着残り専用枠
# =========================
def _safe_float(v, default=np.nan):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _is_inner_frame(row, field_n):
    no = int(_safe_float(row.get('horseNo', 0), 0) or 0)
    if no <= 0:
        return False
    if field_n <= 9:
        return no <= 2
    if field_n <= 12:
        return no <= 3
    if field_n <= 16:
        return no <= 4
    return no <= 5


def _is_outer_frame(row, field_n):
    no = int(_safe_float(row.get('horseNo', 0), 0) or 0)
    if no <= 0:
        return False
    if field_n <= 9:
        return no >= field_n - 1
    if field_n <= 12:
        return no >= field_n - 2
    if field_n <= 16:
        return no >= field_n - 3
    return no >= field_n - 4


def _distance_shift_score(row):
    dist = _safe_float(row.get('distance', np.nan))
    prev_dist = _safe_float(row.get('prevDistance', np.nan))
    if pd.isna(dist) or pd.isna(prev_dist):
        return 0
    diff = dist - prev_dist
    raw4 = _safe_float(row.get('prev4cPos', np.nan))
    pos4 = _position_score(row.get('prev4cCat', ''))
    near_front = (not pd.isna(raw4) and raw4 <= 8) or pos4 <= 4
    front = (not pd.isna(raw4) and raw4 <= 5) or pos4 <= 3
    if diff <= -200 and near_front:
        return 10
    if diff >= 200 and front:
        return 8
    return 0


def _second_place_holdover_score(row, honmei, field_n):
    score = 0.0
    raw4 = _safe_float(row.get('prev4cPos', np.nan))
    rel4 = _safe_float(row.get('prev4cRel_final', np.nan))
    pos4 = _position_score(row.get('prev4cCat', ''))

    front4 = (not pd.isna(raw4) and raw4 <= 5) or (not pd.isna(rel4) and rel4 <= 0.45) or pos4 <= 3
    near_front = (not pd.isna(raw4) and raw4 <= 8) or (not pd.isna(rel4) and rel4 <= 0.65) or pos4 <= 4
    very_back = (not pd.isna(raw4) and raw4 >= 12) or pos4 >= 5

    inner = _is_inner_frame(row, field_n)
    outer = _is_outer_frame(row, field_n)

    prev_st = _safe_float(row.get('prevStraight', 50), 50)
    prev2_st = _safe_float(row.get('prev2Straight', 50), 50)
    total = _safe_float(row.get('総合点', 0), 0)
    body = _safe_float(row.get('本体点', 0), 0)
    rank = str(row.get('トータルランク', ''))
    rel_rank = str(row.get('相対評価', ''))

    track = norm_track(row.get('場所', ''))
    surface = norm_surface(row.get('surface', ''))
    dist = _safe_float(row.get('distance', np.nan))

    if front4:
        score += 30
    elif near_front:
        score += 16
    elif very_back:
        score -= 12

    if inner:
        score += 12
    if outer and front4 and surface == 'ダ':
        score += 6

    score += prev_st * 0.18
    score += prev2_st * 0.10
    score += total * 0.22
    score += body * 0.10

    if prev_st >= 75 or prev2_st >= 75:
        score += 10
    if prev_st >= 85:
        score += 6

    if rank in ['S', 'A', 'B']:
        score += 12
    elif rank == 'C':
        score += 6
    elif rank == 'D':
        score -= 8

    if rel_rank in ['S', 'A', 'B']:
        score += 5

    score += _distance_shift_score(row)

    if track == '京都':
        if surface == 'ダ' and front4:
            score += 14
        if surface == '芝' and dist <= 1400 and front4:
            score += 14
        if surface == '芝' and dist >= 2200 and inner and near_front:
            score += 14
        if surface == '芝' and dist >= 2200 and very_back and not inner:
            score -= 8

    if track == '新潟':
        if surface == 'ダ' and front4:
            score += 14
        if surface == '芝' and dist <= 1400 and near_front:
            score += 7

    if track == '東京':
        if surface == '芝':
            if near_front and _eval_score(row.get('straightEval', '普通')) >= 3:
                score += 10
            if _eval_score(row.get('straightEval', '普通')) >= 4:
                score += 6
        if surface == 'ダ' and dist in [1300, 1400, 1600] and near_front:
            score += 10
        if surface == 'ダ' and dist >= 2000 and inner and near_front:
            score += 8

    comment = _comment_text(row)
    score += _keyword_count(comment, ['先行', '好位', '内', '粘', '残', '短縮', '延長', 'ロス', '前受け']) * 5

    if norm_surface(row.get('prevSurface', '')) and norm_surface(row.get('prevSurface', '')) != surface:
        score -= 6

    if very_back and prev_st < 65 and prev2_st < 75:
        score -= 12

    return float(score)


def _pick_second_place_holdover(cand, used, honmei):
    pool = cand[~cand['horseNo'].isin(used)].copy()
    if pool.empty:
        return None

    field_n = len(cand) + 1
    pool['second_place_holdover_score'] = pool.apply(lambda r: _second_place_holdover_score(r, honmei, field_n), axis=1)

    raw4 = pd.to_numeric(pool.get('prev4cPos', np.nan), errors='coerce')
    pos4_score = pool['prev4cCat'].apply(_position_score)
    frontish = (raw4 <= 8) | (pos4_score <= 4)
    has_logic = (pool['prevStraight'] >= 55) | (pool['prev2Straight'] >= 70) | (pool['総合点'] >= 35)
    not_too_low = (pool['総合点'] >= 28) & has_logic

    strong_pool = pool[frontish & not_too_low].copy()
    if strong_pool.empty:
        return _pick_return(cand, used, allow_d=True)

    picked = strong_pool.sort_values(['second_place_holdover_score', '総合点', 'horseNo'], ascending=[False, False, True]).iloc[0]
    if float(picked.get('second_place_holdover_score', 0)) < 45:
        fallback = _pick_return(cand, used, allow_d=True)
        if fallback is not None:
            return fallback
    return picked


# =========================
# v12.9 新ロジック:
# 三連複3列目追加 = 能力残し枠 + 位置取り・展開残り枠
# =========================
def _ability_remaining_score(row):
    """
    既存の○▲△から漏れた中で、
    能力面だけなら3列目に残す価値がある馬を拾うスコア。
    そのレースで一番強い馬ではなく「消し切れない馬」用。
    """
    prev_st = _safe_float(row.get('prevStraight', 50), 50)
    prev2_st = _safe_float(row.get('prev2Straight', 50), 50)
    total = _safe_float(row.get('総合点', 0), 0)
    body = _safe_float(row.get('本体点', 0), 0)
    place_coef = _safe_float(row.get('前走場所直線補正', 1.0), 1.0)

    score = 0.0
    score += body * 0.40
    score += total * 0.30
    score += prev_st * 0.12
    score += prev2_st * 0.08
    score += place_coef * 10
    score += _rank_score(row.get('トータルランク', '')) * 3
    score += _rank_score(row.get('相対評価', '')) * 3

    if prev_st >= 75 or prev2_st >= 75:
        score += 8
    if place_coef >= 1.10:
        score += 5
    if _eval_score(row.get('straightEval', '普通')) >= 4:
        score += 5

    return float(score)


def _pick_ability_remaining(cand, used):
    """
    他1: 能力残し枠
    馬連相手には足りないが、三連複3列目なら残したい馬。
    """
    pool = cand[~cand['horseNo'].isin(used)].copy()
    if pool.empty:
        return None

    # 低すぎる馬は除外。ただし相対評価B以上や直線点が高い馬は残す
    pool['ability_remaining_score'] = pool.apply(_ability_remaining_score, axis=1)
    keep = (
        (pool['総合点'] >= 28)
        | (pool['本体点'] >= 32)
        | (pool['prevStraight'] >= 70)
        | (pool['prev2Straight'] >= 75)
        | (pool['相対評価'].isin(['S', 'A', 'B']))
    )
    pool = pool[keep].copy()
    if pool.empty:
        return None

    return pool.sort_values(
        ['ability_remaining_score', '総合点', 'horseNo'],
        ascending=[False, False, True]
    ).iloc[0]


def _position_remaining_score(row, field_n):
    """
    他2: 位置取り・展開残り枠
    総合点は低くても、前4角前寄り/内枠/距離替わりで3着に残る形を拾う。
    """
    raw4 = _safe_float(row.get('prev4cPos', np.nan))
    rel4 = _safe_float(row.get('prev4cRel_final', np.nan))
    pos4 = _position_score(row.get('prev4cCat', ''))
    inner = _is_inner_frame(row, field_n)
    outer = _is_outer_frame(row, field_n)

    front4 = (not pd.isna(raw4) and raw4 <= 5) or (not pd.isna(rel4) and rel4 <= 0.45) or pos4 <= 3
    near_front = (not pd.isna(raw4) and raw4 <= 8) or (not pd.isna(rel4) and rel4 <= 0.65) or pos4 <= 4
    very_back = (not pd.isna(raw4) and raw4 >= 12) or pos4 >= 5

    track = norm_track(row.get('場所', ''))
    surface = norm_surface(row.get('surface', ''))
    dist = _safe_float(row.get('distance', np.nan))

    score = 0.0
    if front4:
        score += 32
    elif near_front:
        score += 18
    elif very_back:
        score -= 10

    if inner:
        score += 10
    if outer and surface == 'ダ' and front4:
        score += 5

    score += _safe_float(row.get('展開位置補正', 1.0), 1.0) * 12
    score += _safe_float(row.get('本体点', 0), 0) * 0.20
    score += _safe_float(row.get('総合点', 0), 0) * 0.15
    score += _distance_shift_score(row)

    if track == '京都':
        if surface == '芝' and dist <= 1400 and front4:
            score += 12
        if surface == '芝' and dist >= 2200 and inner and near_front:
            score += 12
        if surface == 'ダ' and front4:
            score += 12
    elif track == '新潟':
        if surface == 'ダ' and front4:
            score += 12
        if surface == '芝' and dist <= 1400 and near_front:
            score += 6
    elif track == '東京':
        if surface == '芝' and near_front:
            score += 8
        if surface == 'ダ' and near_front:
            score += 8

    return float(score)


def _pick_position_remaining(cand, used):
    """
    他2: 位置取り・展開残り枠。
    能力残し枠とは別に、3着に残る形の馬を拾う。
    """
    pool = cand[~cand['horseNo'].isin(used)].copy()
    if pool.empty:
        return None

    field_n = len(cand) + 1
    pool['position_remaining_score'] = pool.apply(lambda r: _position_remaining_score(r, field_n), axis=1)

    raw4 = pd.to_numeric(pool.get('prev4cPos', np.nan), errors='coerce')
    pos4_score = pool['prev4cCat'].apply(_position_score)
    frontish = (raw4 <= 8) | (pos4_score <= 4)
    has_min_logic = (
        (pool['総合点'] >= 20)
        | (pool['本体点'] >= 24)
        | (pool['prevStraight'] >= 55)
        | (pool['prev2Straight'] >= 65)
        | (pool['展開位置補正'] >= 1.05)
    )

    pool = pool[frontish & has_min_logic].copy()
    if pool.empty:
        return None

    return pool.sort_values(
        ['position_remaining_score', '総合点', 'horseNo'],
        ascending=[False, False, True]
    ).iloc[0]


def _pick_extra_two(cand, used):
    """
    三連複フォーメーション用の追加2頭。
    他1 = 能力残し枠
    他2 = 位置取り・展開残り枠
    """
    extra = []
    used2 = set(used)

    ability = _pick_ability_remaining(cand, used2)
    if ability is not None:
        extra.append(('能力残し枠', ability))
        used2.add(ability['horseNo'])

    position = _pick_position_remaining(cand, used2)
    if position is not None:
        extra.append(('位置取り・展開残り枠', position))
        used2.add(position['horseNo'])

    # 片方しか取れなかった場合は、能力残しで補完
    if len(extra) < 2:
        fallback = _pick_ability_remaining(cand, used2)
        if fallback is not None:
            extra.append(('能力残し枠', fallback))
            used2.add(fallback['horseNo'])

    return extra[:2]


def _build_trio_bets(honmei, mates3):
    a, b, c = mates3[0], mates3[1], mates3[2]
    return [(_trio_text(honmei, a, b), '三連複'), (_trio_text(honmei, a, c), '三連複'), (_trio_text(honmei, b, c), '三連複')]


def _strong_buy_ok(honmei, conf, honmei_type, mates):
    if conf < 95.0 or honmei_type != '1・2着型' or len(mates) < 3:
        return False, '強気条件未満'
    ab_count = sum(1 for m in mates if str(m.get('トータルランク', '')) in ['S', 'A', 'B'])
    d_count = sum(1 for m in mates if str(m.get('トータルランク', '')) == 'D')
    avg_score = sum(float(m.get('総合点', 0) or 0) for m in mates) / len(mates)
    if ab_count >= 2 and d_count <= 1 and avg_score >= 42:
        return True, '本命信頼度95%以上かつ相手3頭の中にA/B評価が2頭以上'
    return False, '相手3頭のまとまりが強気条件未満'



# =========================
# v14.0 買い目自動選択AI
# =========================
def _num(row, col, default=0.0):
    try:
        v = row.get(col, default)
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def _horse_no(row):
    try:
        return int(float(row.get('horseNo', row.get('馬番', 0)) or 0))
    except Exception:
        return 0


def _bet_pair(h1, h2):
    return f"{_horse_no(h1)}-{_horse_no(h2)}"


def _bet_pair_nums(a, b):
    return f"{int(a)}-{int(b)}"


def _as_nums(horses):
    return [int(_horse_no(h)) for h in horses if h is not None and _horse_no(h) > 0]


def _unique_horses(items):
    out = []
    seen = set()
    for x in items:
        if x is None:
            continue
        no = _horse_no(x)
        if no <= 0 or no in seen:
            continue
        out.append(x)
        seen.add(no)
    return out


def _compact_nums(nums):
    return ",".join(str(int(n)) for n in nums if int(n) > 0)


def _format_auto_bet_display(rec):
    kind = norm_text(rec.get('auto_bet_kind', ''))
    form = norm_text(rec.get('auto_bet_form', ''))
    main = norm_text(rec.get('auto_bet_main', ''))
    if not kind:
        return ""
    line1 = f"推奨 {kind}"
    if form:
        line1 += f" {form}"
    line2 = main
    return line1 + ("\n" + line2 if line2 else "")


def _make_auto_bet(kind, form, main, detail=None, reason=''):
    kind = norm_text(kind)
    form = norm_text(form)
    main = norm_text(main)
    detail = norm_text(detail) if detail is not None else ''
    if not detail:
        detail = f"{kind}{form}：{main}" if form else f"{kind}：{main}"
    return {
        'auto_bet_kind': kind,
        'auto_bet_form': form,
        'auto_bet_main': main,
        'auto_bet_detail': detail,
        'auto_bet_reason': reason,
    }


def _format_auto_bets_display(bets):
    if not bets:
        return ""
    lines = []
    for b in bets[:3]:
        kind = norm_text(b.get('auto_bet_kind', ''))
        form = norm_text(b.get('auto_bet_form', ''))
        main = norm_text(b.get('auto_bet_main', ''))
        if not kind:
            continue
        label = kind
        if form:
            # 画像では長すぎ防止のため形式は短く残す
            label += form.replace('1頭軸流し', '流し').replace('2頭軸流し', '2軸').replace('フォーメーション', 'フォメ')
        if main:
            lines.append(f"{label} {main}")
        else:
            lines.append(label)
    return "\n".join(lines)


def _auto_bet_multi_recommendations(primary, honmei, same, comp, ret, extra_mates, cand, conf, honmei_type):
    """
    1レース1買い目固定ではなく、相性が良い場合は複数券種を出す。
    金額は出さない。
    最大3行まで。
    """
    h_no = _horse_no(honmei)
    same_no = _horse_no(same) if same is not None else 0
    comp_no = _horse_no(comp) if comp is not None else 0
    ret_no = _horse_no(ret) if ret is not None else 0

    h_score = _num(honmei, '総合点', 0)
    h_body = _num(honmei, '本体点', 0)
    h_pace = _num(honmei, '展開位置補正', 1)
    h_pos4 = _position_score(honmei.get('prev4cCat', ''))

    same_score = _num(same, '総合点', 0) if same is not None else 0
    comp_score = _num(comp, '総合点', 0) if comp is not None else 0
    ret_score = _num(ret, '総合点', 0) if ret is not None else 0

    core = _unique_horses([same, comp, ret])
    extras = [m for role, m in (extra_mates or []) if m is not None]
    third_pool = _unique_horses([comp, ret] + extras)
    core_nums = _as_nums(core)
    third_nums = _as_nums(third_pool)[:4]

    primary_kind = norm_text(primary.get('auto_bet_kind', ''))
    primary_form = norm_text(primary.get('auto_bet_form', ''))

    bets = []

    def add_bet(b):
        key = (norm_text(b.get('auto_bet_kind', '')), norm_text(b.get('auto_bet_form', '')), norm_text(b.get('auto_bet_main', '')))
        for old in bets:
            old_key = (norm_text(old.get('auto_bet_kind', '')), norm_text(old.get('auto_bet_form', '')), norm_text(old.get('auto_bet_main', '')))
            if old_key == key:
                return
        if b.get('auto_bet_kind'):
            bets.append(b)

    # 勝ち切り要素が強い時は単勝を足す
    win_add = (
        conf >= 94
        and h_score >= 60
        and h_body >= 40
        and h_pace >= 1.10
        and h_pos4 <= 3
    )
    if win_add:
        add_bet(_make_auto_bet('単勝', '', f'{h_no}', f'単勝：{h_no}', '◎の勝ち切り要素が強いため単勝も候補。'))

    # 既存の主買い目
    if primary_kind and primary_kind != '見送り':
        add_bet(primary)

    # 1・2着型は馬連を補助で出す
    if honmei_type == '1・2着型' and len(core_nums) >= 2:
        maren_nums = core_nums[:3]
        add_bet(_make_auto_bet(
            '馬連',
            '1頭軸流し',
            f'{h_no}-{_compact_nums(maren_nums)}',
            f'馬連一頭軸流し：{h_no}-{_compact_nums(maren_nums)}',
            '◎が1・2着型で、○▲△に相手を整理できるため。'
        ))

    # 安定補助としてワイド。○と△を優先
    if same_no > 0 and (ret_no > 0 or comp_no > 0):
        wide_nums = [same_no]
        if ret_no > 0:
            wide_nums.append(ret_no)
        elif comp_no > 0:
            wide_nums.append(comp_no)
        add_bet(_make_auto_bet(
            'ワイド',
            '1頭軸流し',
            f'{h_no}-{_compact_nums(wide_nums[:2])}',
            f'ワイド一頭軸流し：{h_no}-{_compact_nums(wide_nums[:2])}',
            '◎の3着内を土台に、○と残り目を拾う安定補助。'
        ))

    # ◎○が強く3着候補が広い時は三連複2軸/三連単2軸Mも候補
    two_heads_strong = (
        same is not None
        and conf >= 93
        and h_score >= 56
        and same_score >= 52
        and abs(h_score - same_score) <= 7
        and len(third_nums) >= 2
    )
    if two_heads_strong:
        # 既に主買い目が三連単なら三連複2軸を補助、そうでなければ三連単2軸Mを上振れ候補
        if primary_kind == '三連単':
            add_bet(_make_auto_bet(
                '三連複',
                '2頭軸流し',
                f'{h_no},{same_no}-{_compact_nums(third_nums)}',
                f'三連複二頭軸流し：{h_no},{same_no}-{_compact_nums(third_nums)}',
                '◎○が強く、3列目のズレを拾うため。'
            ))
        else:
            add_bet(_make_auto_bet(
                '三連単',
                '2頭軸M',
                f'{h_no},{same_no}-{_compact_nums(third_nums)}',
                f'三連単二頭軸マルチ：{h_no},{same_no}-{_compact_nums(third_nums)}',
                '◎○が強く、3頭目のズレで上振れを狙うため。'
            ))

    # 相手が散るが軸が堅い時は三連複フォメを補助
    if len(core_nums) >= 3 and len(_as_nums(_unique_horses(core + extras))) >= 5 and conf >= 92:
        third_all = _as_nums(_unique_horses(core + extras))[:5]
        add_bet(_make_auto_bet(
            '三連複',
            'フォメ',
            f'{h_no}-{_compact_nums(core_nums[:3])}-{_compact_nums(third_all)}',
            f'三連複フォーメーション：{h_no}-{_compact_nums(core_nums[:3])}-{_compact_nums(third_all)}',
            '◎は堅いが相手がズレやすく、3列目を広げるため。'
        ))

    # 難解で相手を増やしすぎたくない時は複勝を添える
    if len(bets) == 0:
        add_bet(_make_auto_bet('複勝', '', f'{h_no}', f'複勝：{h_no}', '相手を絞り切れないため。'))

    # 表示が長くなりすぎるため最大3つ
    return bets[:3]


def _candidate_spread_score(cands):
    if cands is None or len(cands) == 0:
        return 0.0
    vals = []
    for _, r in cands.iterrows():
        vals.append(_num(r, '総合点', 0))
    if not vals:
        return 0.0
    return float(max(vals) - min(vals))


def _auto_bet_recommendation(honmei, same, comp, ret, extra_mates, cand, conf, honmei_type):
    """
    全券種・全形式から、そのレースに合う買い目を1つだけ選ぶ。
    金額は出さない。
    """
    same = same if same is not None else None
    comp = comp if comp is not None else None
    ret = ret if ret is not None else None

    extras = [m for role, m in (extra_mates or []) if m is not None]
    core = _unique_horses([same, comp, ret])
    wide_pool = _unique_horses([same, ret, comp])
    spread_pool = _unique_horses([same, comp, ret] + extras)

    h_no = _horse_no(honmei)
    same_no = _horse_no(same) if same is not None else 0
    comp_no = _horse_no(comp) if comp is not None else 0
    ret_no = _horse_no(ret) if ret is not None else 0

    h_score = _num(honmei, '総合点', 0)
    h_body = _num(honmei, '本体点', 0)
    h_pace = _num(honmei, '展開位置補正', 1)
    h_place = _num(honmei, '前走場所直線補正', 1)
    h_rank = str(honmei.get('トータルランク', ''))
    h_rel = str(honmei.get('相対評価', ''))
    h_pos4 = _position_score(honmei.get('prev4cCat', ''))

    second_score = _num(same, '総合点', 0) if same is not None else 0
    gap_to_second = h_score - second_score

    same_score = _num(same, '総合点', 0) if same is not None else 0
    comp_score = _num(comp, '総合点', 0) if comp is not None else 0
    ret_score = _num(ret, '総合点', 0) if ret is not None else 0

    # 相手のまとまり
    core_scores = [same_score, comp_score, ret_score]
    core_min = min(core_scores) if core_scores else 0
    core_avg = sum(core_scores) / len(core_scores) if core_scores else 0
    core_ab = sum(1 for m in core if str(m.get('トータルランク', '')) in ['S', 'A', 'B'])
    extra_good = sum(1 for m in extras if _num(m, '総合点', 0) >= 35 or str(m.get('相対評価', '')) in ['S', 'A', 'B'])

    win_like = (
        conf >= 94
        and h_score >= 62
        and gap_to_second >= 8
        and h_body >= 42
        and h_pace >= 1.12
        and h_pos4 <= 3
    )

    super_win_like = (
        conf >= 95
        and h_score >= 66
        and gap_to_second >= 12
        and h_body >= 43
        and h_pace >= 1.18
        and h_pos4 <= 3
    )

    two_heads_strong = (
        same is not None
        and conf >= 93
        and h_score >= 56
        and same_score >= 52
        and abs(h_score - same_score) <= 6
        and str(same.get('相対評価', '')) in ['S', 'A']
    )

    box_like = (
        len(core) >= 3
        and h_score >= 50
        and core_min >= 42
        and max(core_scores) - min(core_scores) <= 12
        and not super_win_like
    )

    wide_like = (
        conf >= 90
        and (
            core_min < 38
            or len(extras) >= 1
            or ret_score < 40
            or honmei_type == '3着型'
        )
    )

    difficult_like = (
        core_min < 32
        and extra_good == 0
    )

    # 1. 超勝ち切り型: 三連単フォーメーション or 馬単
    if super_win_like and len(core) >= 3:
        nums = _as_nums(core[:3])
        return {
            'auto_bet_kind': '三連単',
            'auto_bet_form': 'フォメ',
            'auto_bet_main': f"{h_no}→{_compact_nums(nums[:2])}→{_compact_nums(nums)}",
            'auto_bet_detail': f"三連単フォーメーション：{h_no}→{_compact_nums(nums[:2])}→{_compact_nums(nums)}",
            'auto_bet_reason': '◎が勝ち切り型で抜けており、相手上位も3頭に整理できるため。'
        }

    # 2. ◎○の2頭が強い: 三連単2頭軸マルチ or 三連複2頭軸
    if two_heads_strong and len(spread_pool) >= 3:
        third_nums = _as_nums(_unique_horses([comp, ret] + extras))[:4]
        if len(third_nums) >= 2 and conf >= 95 and h_score >= 58:
            return {
                'auto_bet_kind': '三連単',
                'auto_bet_form': '2頭軸M',
                'auto_bet_main': f"{h_no},{same_no}-{_compact_nums(third_nums)}",
                'auto_bet_detail': f"三連単二頭軸マルチ：{h_no},{same_no}-{_compact_nums(third_nums)}",
                'auto_bet_reason': '◎と○の2頭が強く、3頭目だけズレる形を想定。'
            }
        return {
            'auto_bet_kind': '三連複',
            'auto_bet_form': '2頭軸',
            'auto_bet_main': f"{h_no},{same_no}-{_compact_nums(third_nums)}",
            'auto_bet_detail': f"三連複二頭軸流し：{h_no},{same_no}-{_compact_nums(third_nums)}",
            'auto_bet_reason': '◎と○の2頭が安定し、3列目を広げて拾う形。'
        }

    # 3. 勝ち切りだが三単ほどではない: 馬単一頭軸流し
    if win_like and len(core) >= 2:
        nums = _as_nums(core[:3])
        return {
            'auto_bet_kind': '馬単',
            'auto_bet_form': '1頭軸流し',
            'auto_bet_main': f"{h_no}→{_compact_nums(nums)}",
            'auto_bet_detail': f"馬単一頭軸流し：{h_no}→{_compact_nums(nums)}",
            'auto_bet_reason': '◎の勝ち切り要素が強く、相手も○▲△に整理できるため。'
        }

    # 4. 上位横並び: BOX
    if box_like:
        nums = _as_nums([honmei] + core[:3])
        if len(nums) >= 4 and core_avg >= 48:
            return {
                'auto_bet_kind': '三連複',
                'auto_bet_form': 'BOX',
                'auto_bet_main': _compact_nums(nums[:4]),
                'auto_bet_detail': f"三連複BOX：{_compact_nums(nums[:4])}",
                'auto_bet_reason': '上位4頭が横並びで、軸固定よりBOX向き。'
            }
        return {
            'auto_bet_kind': '馬連',
            'auto_bet_form': 'BOX',
            'auto_bet_main': _compact_nums(nums[:4]),
            'auto_bet_detail': f"馬連BOX：{_compact_nums(nums[:4])}",
            'auto_bet_reason': '上位が横並びで、連対の入れ替わりを想定。'
        }

    # 5. 相手ズレ型: 三連複フォーメーション
    if len(spread_pool) >= 5 and conf >= 92 and extra_good >= 1:
        second_nums = _as_nums(core[:3])
        third_nums = _as_nums(_unique_horses(core + extras))[:5]
        return {
            'auto_bet_kind': '三連複',
            'auto_bet_form': 'フォメ',
            'auto_bet_main': f"{h_no}-{_compact_nums(second_nums)}-{_compact_nums(third_nums)}",
            'auto_bet_detail': f"三連複フォーメーション：{h_no}-{_compact_nums(second_nums)}-{_compact_nums(third_nums)}",
            'auto_bet_reason': '◎は堅いが相手がズレやすく、他1・他2まで3列目に入れる形。'
        }

    # 6. 安定型: ワイド一頭軸流し
    if wide_like and len(wide_pool) >= 2:
        nums = _as_nums([same, ret]) if ret is not None else _as_nums(wide_pool[:2])
        if len(nums) < 2:
            nums = _as_nums(wide_pool[:2])
        return {
            'auto_bet_kind': 'ワイド',
            'auto_bet_form': '1頭軸流し',
            'auto_bet_main': f"{h_no}-{_compact_nums(nums[:2])}",
            'auto_bet_detail': f"ワイド一頭軸流し：{h_no}-{_compact_nums(nums[:2])}",
            'auto_bet_reason': '◎の3着内は強いが、相手の連対順がズレやすいため。'
        }

    # 7. 連軸型: 馬連一頭軸流し
    if honmei_type == '1・2着型' and len(core) >= 3:
        nums = _as_nums(core[:3])
        return {
            'auto_bet_kind': '馬連',
            'auto_bet_form': '1頭軸流し',
            'auto_bet_main': f"{h_no}-{_compact_nums(nums)}",
            'auto_bet_detail': f"馬連一頭軸流し：{h_no}-{_compact_nums(nums)}",
            'auto_bet_reason': '◎が1〜2着型で、○▲△の3頭に相手を整理できるため。'
        }

    # 8. 難解だが◎は来そう: 複勝
    if conf >= 90 and difficult_like:
        return {
            'auto_bet_kind': '複勝',
            'auto_bet_form': '',
            'auto_bet_main': f"{h_no}",
            'auto_bet_detail': f"複勝：{h_no}",
            'auto_bet_reason': '◎は馬券内候補だが、相手が散りすぎるため。'
        }

    # 9. 最後の保険: ワイド
    if len(wide_pool) >= 2:
        nums = _as_nums(wide_pool[:2])
        return {
            'auto_bet_kind': 'ワイド',
            'auto_bet_form': '1頭軸流し',
            'auto_bet_main': f"{h_no}-{_compact_nums(nums)}",
            'auto_bet_detail': f"ワイド一頭軸流し：{h_no}-{_compact_nums(nums)}",
            'auto_bet_reason': '標準の安定型として、◎から相手2頭へ。'
        }

    return {
        'auto_bet_kind': '単勝' if win_like else '複勝',
        'auto_bet_form': '',
        'auto_bet_main': f"{h_no}",
        'auto_bet_detail': f"{'単勝' if win_like else '複勝'}：{h_no}",
        'auto_bet_reason': '相手を絞り切れないため、◎単体の券種を選択。'
    }


def recommend_for_race(g):
    g = g.sort_values(['総合点', 'horseNo'], ascending=[False, True]).reset_index(drop=True)
    honmei = g.iloc[0]
    conf = confidence_from_score(honmei['総合点'])
    short_comment = f'本体{honmei["本体点"]:.1f}×展開{honmei["展開位置補正"]:.2f}×場所{honmei["前走場所直線補正"]:.2f}'
    base = {'honmei': honmei, 'confidence': conf, 'short_comment': short_comment}

    if conf < 90.0:
        return {
            **base,
            'status': '見送り',
            'honmei_type': '対象外',
            'bet_strength': '見送り',
            'bet_type': '見送り',
            'bets': [],
            'wide_bets': [],
            'umaren_bets': [],
            'trio_bets': [],
            'mates': [],
            'extra_mates': [],
            'auto_bet': {
                'auto_bet_kind': '見送り',
                'auto_bet_form': '',
                'auto_bet_main': '',
                'auto_bet_detail': '見送り',
                'auto_bet_reason': '本命信頼度が90%未満'
            },
            'reason': '本命信頼度が90%未満'
        }

    honmei_type, type_reason = judge_honmei_type(honmei, conf)
    cand = _prepare_candidates(g, honmei)
    if cand.empty:
        return {
            **base,
            'status': '見送り',
            'honmei_type': honmei_type,
            'bet_strength': '見送り',
            'bet_type': '見送り',
            'bets': [],
            'wide_bets': [],
            'umaren_bets': [],
            'trio_bets': [],
            'mates': [],
            'extra_mates': [],
            'auto_bet': {
                'auto_bet_kind': '見送り',
                'auto_bet_form': '',
                'auto_bet_main': '',
                'auto_bet_detail': '見送り',
                'auto_bet_reason': 'おすすめ馬券に必要な相手が揃わない'
            },
            'reason': 'おすすめ馬券に必要な相手が揃わない'
        }

    used = {honmei['horseNo']}
    same = _pick_same(cand, used)
    if same is not None:
        used.add(same['horseNo'])

    comp = _pick_comp(cand, used)
    if comp is not None:
        used.add(comp['horseNo'])

    ret = _pick_second_place_holdover(cand, used, honmei)
    if ret is not None:
        used.add(ret['horseNo'])

    extra_mates = _pick_extra_two(cand, used)

    mates = [m for m in [same, comp, ret] if m is not None]
    if len(mates) < 2:
        auto_bet = {
            'auto_bet_kind': '複勝',
            'auto_bet_form': '',
            'auto_bet_main': f"{_horse_no(honmei)}",
            'auto_bet_detail': f"複勝：{_horse_no(honmei)}",
            'auto_bet_reason': '相手候補が不足しているため、◎単体の安定券種を選択。'
        }
        return {
            **base,
            'status': '買い対象',
            'honmei_type': honmei_type,
            'bet_strength': '通常',
            'bet_type': auto_bet['auto_bet_detail'],
            'bets': [(auto_bet['auto_bet_detail'], '自動選択')],
            'wide_bets': [],
            'umaren_bets': [],
            'trio_bets': [],
            'mates': mates,
            'extra_mates': extra_mates,
            'auto_bet': auto_bet,
            'reason': auto_bet['auto_bet_reason']
        }

    # 3頭目が足りない場合でも自動選択は可能にする
    if len(mates) < 3:
        fill = _pick_return(cand, used, allow_d=True)
        if fill is not None:
            mates.append(fill)

    same = mates[0] if len(mates) > 0 else None
    comp = mates[1] if len(mates) > 1 else None
    ret = mates[2] if len(mates) > 2 else None

    same_reason = f'{_horse_label(same)}は本命と展開・位置取りが近い同展開相手。' if same is not None else ''
    comp_reason = f'{_horse_label(comp)}は本命と位置/展開にズレがあり、展開ズレを拾う補完相手。' if comp is not None else ''
    ret_reason = f'{_horse_label(ret)}は低評価でも2着に残る形を拾う2着残り専用枠。' if ret is not None else ''

    extra_reason = ''
    if extra_mates:
        extra_lines = []
        for role, m in extra_mates:
            extra_lines.append(f'{role}: {_horse_label(m)}')
        extra_reason = '\n追加2頭：' + ' / '.join(extra_lines)

    auto_bet = _auto_bet_recommendation(honmei, same, comp, ret, extra_mates, cand, conf, honmei_type)
    auto_bets = _auto_bet_multi_recommendations(auto_bet, honmei, same, comp, ret, extra_mates, cand, conf, honmei_type)

    reason_parts = [f'本命は{type_reason}。']
    for rtxt in [same_reason, comp_reason, ret_reason]:
        if rtxt:
            reason_parts.append(rtxt)
    base_reason = '\n'.join(reason_parts) + extra_reason
    auto_reason_lines = []
    for b in auto_bets:
        if norm_text(b.get('auto_bet_reason', '')):
            auto_reason_lines.append(f"{b.get('auto_bet_kind','')}：{b.get('auto_bet_reason','')}")
    full_reason = base_reason + "\n自動買い目理由：" + " / ".join(auto_reason_lines)

    # 表示用のbetsは複数買い目を保持
    bets = [(b['auto_bet_detail'], '自動選択') for b in auto_bets]

    return {
        **base,
        'status': '買い対象',
        'honmei_type': honmei_type,
        'bet_strength': '自動',
        'bet_type': ' / '.join([b['auto_bet_kind'] + (b['auto_bet_form'] if b.get('auto_bet_form') else '') for b in auto_bets]),
        'bets': bets,
        'wide_bets': [x for x in bets if 'ワイド' in x[0]],
        'umaren_bets': [x for x in bets if '馬連' in x[0]],
        'trio_bets': [x for x in bets if ('三連複' in x[0] or '三連単' in x[0])],
        'mates': mates,
        'extra_mates': extra_mates,
        'auto_bet': auto_bet,
        'auto_bets': auto_bets,
        'reason': full_reason
    }



def safe_race_no(row):
    try:
        v = row.get('raceNo', np.nan)
        if pd.notna(v):
            n = int(float(v))
            if n > 0:
                return n
    except Exception:
        pass
    for key in ['レース', 'raceLabel', 'レース識別ID', 'raceName', 'R', 'Ｒ', 'raceNo']:
        s = norm_text(row.get(key, ''))
        if not s:
            continue
        m = re.search(r'(\d{1,2})\s*R', s, flags=re.IGNORECASE)
        if m:
            n = int(m.group(1))
            if n > 0:
                return n
    return 0


def get_font(size, bold=False):
    paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
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
        t = t[:-1] + '…'
    draw.text((x, y), t, font=font, fill=fill)


def extract_opponent_marks(r):
    try:
        honmei_no = int(float(r.get('馬番', 0) or 0))
    except Exception:
        honmei_no = 0
    opponents = []
    for key in ['買い目1', '買い目2', '買い目3', '買い目4', '買い目5', '買い目6']:
        text = norm_text(r.get(key, ''))
        if not text:
            continue
        nums = re.findall(r'\d+', text)
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
    marks = ['◯', '▲', '△']
    return ' '.join([f'{marks[i]}{opponents[i]}' for i in range(min(3, len(opponents)))])



def bet_type_label(v):
    s = norm_text(v)
    if '馬連' in s and '三連複' in s:
        return '激アツ'
    if '馬連' in s:
        return '連系'
    if '三連複' in s:
        return '複系'
    return ''


def bet_type_label_from_row(r):
    label = bet_type_label(r.get('おすすめ馬券', ''))
    if label:
        return label

    bet_count = 0
    for key in ['買い目1', '買い目2', '買い目3', '買い目4', '買い目5', '買い目6']:
        if norm_text(r.get(key, '')):
            bet_count += 1

    if bet_count >= 6:
        return '激アツ'
    if bet_count == 3:
        honmei_type = norm_text(r.get('本命タイプ', ''))
        if '3着' in honmei_type:
            return '複系'
        return '連系'
    return ''


def bet_type_style(label):
    if label == '激アツ':
        return {'fill': (64, 41, 18), 'outline': (229, 191, 72), 'text': (255, 236, 180)}
    if label == '連系':
        return {'fill': (26, 40, 62), 'outline': (120, 152, 202), 'text': (226, 236, 248)}
    if label == '複系':
        return {'fill': (28, 48, 39), 'outline': (112, 164, 126), 'text': (225, 241, 228)}
    return {'fill': (32, 36, 45), 'outline': (95, 102, 120), 'text': (230, 230, 230)}


def draw_buy_type_pill(draw, x_right, y_top, label, font):
    if not label:
        return
    style = bet_type_style(label)
    bbox = draw.textbbox((0, 0), str(label), font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x = 18
    pad_y = 9
    w = max(100, tw + pad_x * 2)
    h = max(40, th + pad_y * 2)
    x1 = x_right - w
    y1 = y_top
    x2 = x_right
    y2 = y_top + h
    draw.rounded_rectangle((x1, y1, x2, y2), radius=16, fill=style['fill'], outline=style['outline'], width=2)
    draw_centered_text(draw, (x1, y1, x2, y2), label, font, style['text'])



def _extract_opponent_numbers(r, max_count=3):
    try:
        honmei_no = int(float(r.get('馬番', 0) or 0))
    except Exception:
        honmei_no = 0

    opponents = []
    for key in ['買い目1', '買い目2', '買い目3', '買い目4', '買い目5', '買い目6']:
        text_value = norm_text(r.get(key, ''))
        if not text_value:
            continue
        nums = re.findall(r'\d+', text_value)
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
            if len(opponents) >= max_count:
                return opponents
    return opponents


def build_standard_bet_display(r):
    """
    SNS画像右側に表示する買い目。
    v14.0では自動買い目を優先表示。
    """
    auto_display = norm_text(r.get('自動買い目表示', ''))
    if auto_display:
        return auto_display

    auto_kind = norm_text(r.get('自動券種', ''))
    auto_form = norm_text(r.get('自動形式', ''))
    auto_main = norm_text(r.get('自動買い目', ''))
    if auto_kind:
        line1 = f"推奨 {auto_kind}"
        if auto_form:
            line1 += f" {auto_form}"
        return line1 + ("\n" + auto_main if auto_main else "")

    # 古い保存データ向けの標準型フォールバック
    try:
        honmei_no = int(float(r.get('馬番', 0) or 0))
    except Exception:
        honmei_no = 0

    if honmei_no <= 0:
        return ""

    opponents = _extract_opponent_numbers(r, max_count=3)
    if not opponents:
        return ""

    maren = opponents[:3]
    wide = []
    if len(opponents) >= 1:
        wide.append(opponents[0])
    if len(opponents) >= 3:
        wide.append(opponents[2])
    elif len(opponents) >= 2:
        wide.append(opponents[1])

    maren_text = f"馬連 {honmei_no}-" + ",".join(str(x) for x in maren)
    wide_text = f"ワイド {honmei_no}-" + ",".join(str(x) for x in wide) if wide else ""
    return maren_text + ("\n" + wide_text if wide_text else "")


def draw_bet_display_box(draw, x_right, y_top, text_value, font):
    text_value = str(text_value or "").strip()
    if not text_value:
        return

    lines = [line.strip() for line in text_value.split("\n") if line.strip()]
    if not lines:
        return

    pad_x = 14
    pad_y = 8
    line_h = 30
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(bbox[2] - bbox[0])

    w = max(180, max(widths) + pad_x * 2)
    h = pad_y * 2 + line_h * len(lines)
    x1 = x_right - w
    y1 = y_top
    x2 = x_right
    y2 = y_top + h

    fill = (20, 26, 36)
    outline = (229, 191, 72)
    text_fill = (255, 236, 180)

    draw.rounded_rectangle((x1, y1, x2, y2), radius=14, fill=fill, outline=outline, width=2)

    for i, line in enumerate(lines):
        y = y1 + pad_y + i * line_h
        draw.text((x1 + pad_x, y), line, font=font, fill=text_fill)



def draw_bet_display_text(draw, x, y, text_value, font, fill, max_width):
    """
    買い目表示を馬名と被らない下段に描く。
    2行を1行に圧縮して、長い場合は省略する。
    """
    text_value = str(text_value or "").replace("\n", " / ").strip()
    if not text_value:
        return
    t = text_value
    while len(t) > 0:
        bbox = draw.textbbox((x, y), t, font=font)
        if bbox[2] - bbox[0] <= max_width:
            break
        t = t[:-1]
    if t != text_value:
        t = t[:-1] + "…"
    draw.text((x, y), t, font=font, fill=fill)


def make_sns_image(saved):
    items = [r for r in saved if float(r.get('参考信頼度', 0) or 0) >= 90.0]
    if not items:
        return None

    cleaned = []
    for r in items:
        rr = dict(r)
        rr['場所'] = norm_track(rr.get('場所', ''))
        try:
            rr['R'] = int(float(rr.get('R', 0) or 0))
        except Exception:
            rr['R'] = 0
        try:
            rr['馬番'] = int(float(rr.get('馬番', 0) or 0))
        except Exception:
            rr['馬番'] = 0
        rr['馬名'] = norm_text(rr.get('馬名', ''))
        mate_display = extract_opponent_marks(rr)
        others = []
        for k in ['他1', '他2']:
            v = rr.get(k, '')
            if str(v).strip() != '':
                try:
                    others.append(str(int(float(v))))
                except Exception:
                    others.append(str(v))
        if others:
            mate_display = (mate_display + '　他 ' + '、'.join(others)).strip()
        rr['相手表示'] = mate_display
        rr['買い目表示'] = norm_text(rr.get('買い目表示', '')) or build_standard_bet_display(rr)
        cleaned.append(rr)

    order = {'福島': 1, '東京': 2, '京都': 3, '阪神': 4, '中山': 5, '中京': 6, '新潟': 7, '小倉': 8, '札幌': 9, '函館': 10}
    cleaned = sorted(cleaned, key=lambda r: (order.get(r.get('場所', ''), 99), int(r.get('R', 0)), int(r.get('馬番', 0))))

    raw_date = str(cleaned[0].get('日付', ''))
    dt = pd.to_datetime(raw_date, errors='coerce')
    date_main = raw_date.replace('/', '.').replace('-', '.') if pd.isna(dt) else dt.strftime('%Y.%m.%d')

    W = 1080
    row_y = 360
    row_h = 155
    H = max(1920, row_y + len(cleaned) * row_h + 125)
    img = Image.new('RGB', (W, H), (10, 14, 22))
    draw = ImageDraw.Draw(img)

    white = (248, 248, 244)
    gold = (229, 191, 72)
    muted = (145, 149, 160)
    line = (42, 47, 57)
    bg = (10, 14, 22)

    title_font = get_font(70, True)
    sub_font = get_font(25, True)
    date_font = get_font(40, False)
    place_font = get_font(31, True)
    race_font = get_font(36, True)
    horse_no_font = get_font(42, True)
    horse_font = get_font(45, True)
    mate_font = get_font(25, False)
    bet_display_font = get_font(20, True)

    draw.text((80, 92), "T O D A Y ' S   P I C K S", font=sub_font, fill=gold)
    draw.text((80, 150), '本日の推奨馬', font=title_font, fill=white)
    draw.text((670, 135), date_main, font=date_font, fill=gold)
    draw.line((80, 285, 1000, 285), fill=line, width=2)

    for i, r in enumerate(cleaned):
        y = row_y + i * row_h
        if i > 0:
            draw.line((80, y - 42, 1000, y - 42), fill=(31, 36, 45), width=2)
        place = r.get('場所', '')
        race_no = int(r.get('R', 0))
        horse_no = int(r.get('馬番', 0))
        horse_name = r.get('馬名', '')
        mate_text = norm_text(r.get('相手表示', ''))
        bet_display_text = str(r.get('買い目表示', '') or '')

        draw.text((80, y - 6), place, font=place_font, fill=gold)
        draw.text((80, y + 34), f'{race_no}R', font=race_font, fill=gold)
        cx, cy, rad = 250, y + 43, 42
        draw.ellipse((cx - rad, cy - rad, cx + rad, cy + rad), fill=gold)
        draw_centered_text(draw, (cx - rad, cy - rad, cx + rad, cy + rad), str(horse_no), horse_no_font, bg)
        draw_fit_text(draw, (340, y - 6), horse_name, horse_font, white, 620)
        if mate_text:
            mate_text_spaced = mate_text.replace('◯', '○ ').replace('▲', '▲ ').replace('△', '△ ')
            draw.text((340, y + 55), mate_text_spaced, font=mate_font, fill=muted)
        if bet_display_text:
            draw_bet_display_text(draw, 340, y + 88, bet_display_text, bet_display_font, gold, 610)

    bio = io.BytesIO()
    img.save(bio, format='PNG')
    bio.seek(0)
    return bio


def add_saved_recs(new_recs):
    if 'saved_recs' not in st.session_state:
        st.session_state.saved_recs = []
    store = {}
    for r in st.session_state.saved_recs:
        rr = dict(r)
        key = f'{rr.get("日付", "")}_{norm_track(rr.get("場所", ""))}_{int(float(rr.get("R", 0) or 0))}R'
        store[key] = rr
    for r in new_recs:
        rr = dict(r)
        key = f'{rr.get("日付", "")}_{norm_track(rr.get("場所", ""))}_{int(float(rr.get("R", 0) or 0))}R'
        store[key] = rr
    st.session_state.saved_recs = list(store.values())


def saved_df():
    if 'saved_recs' not in st.session_state:
        st.session_state.saved_recs = []
    return pd.DataFrame(st.session_state.saved_recs)


GRADE_MARKERS = ['G1', 'G2', 'G3', 'Ｇ１', 'Ｇ２', 'Ｇ３', 'GI', 'GII', 'GIII', 'ＧⅠ', 'ＧⅡ', 'ＧⅢ', 'Jpn1', 'Jpn2', 'Jpn3']


def auto_grade_race_flag(row):
    s = ' '.join([norm_text(row.get(k, '')) for k in ['grade', 'グレード', '重賞', 'クラス', 'raceName', 'レース名'] if k in row.index])
    if any(m in s for m in GRADE_MARKERS) or '重賞' in s:
        return True
    race_name = norm_text(row.get('raceName', '')) or norm_text(row.get('レース名', ''))
    try:
        rno = int(float(row.get('raceNo', 0) or 0))
    except Exception:
        rno = 0
    hints = ['記念', '杯', '賞', 'ステークス', 'S', 'Ｓ', 'カップ', 'C', 'Ｃ']
    non = ['未勝利', '新馬', '1勝', '2勝', '3勝', 'オープン', 'OP', 'L']
    if any(w in race_name for w in non):
        return False
    return rno in [10, 11, 12] and any(w in race_name for w in hints)


def straight_rank_to_uratori(rank):
    rank = str(rank)
    if rank in ['S', 'A']:
        return '強い'
    if rank == 'B':
        return '普通'
    if rank == 'C':
        return '弱い'
    if rank == 'D':
        return '逆風'
    return '普通'


def build_third_app_straight_export(df, selected_race_ids):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df[df['レース識別ID'].isin(selected_race_ids)].copy()
    if out.empty:
        return pd.DataFrame()
    out['R'] = out.apply(safe_race_no, axis=1)
    out['直線ランク'] = out['トータルランク']
    out['直線点'] = out['総合点'].round(2)
    out['直線相性評価'] = out['straightEval']
    out['直線裏取り判定'] = out['直線ランク'].apply(straight_rank_to_uratori)
    out['直線コメント'] = out.apply(lambda r: f'直線ランク{r["直線ランク"]} / 総合点{float(r["総合点"]):.2f} / 本体{float(r["本体点"]):.1f} / 展開補正{float(r["展開位置補正"]):.2f} / 直線補正{float(r["前走場所直線補正"]):.2f} / 相対{r["相対評価"]}', axis=1)

    export_cols = ['date', '場所', 'R', 'raceName', 'horseNo', 'horseName', '直線ランク', '直線点', '直線相性評価', '直線裏取り判定', '直線コメント', '相対評価', 'トータルランク', '本体点', '展開位置補正', '前走場所直線補正', '総合点']
    export_df = out[export_cols].rename(columns={'date': '日付', 'raceName': 'レース名', 'horseNo': '馬番', 'horseName': '馬名'})
    export_df['馬番'] = pd.to_numeric(export_df['馬番'], errors='coerce').fillna(0).astype(int)
    export_df['R'] = pd.to_numeric(export_df['R'], errors='coerce').fillna(0).astype(int)
    return export_df.sort_values(['日付', '場所', 'R', '馬番']).reset_index(drop=True)


def render_simple_ranking(g):
    show = g[['horseNo', 'horseName', '総合点', 'トータルランク', '相対評価', '本体点', '展開位置補正', '前走場所直線補正']].copy()
    show = show.rename(columns={'horseNo': '馬番', 'horseName': '馬名'})
    st.dataframe(show, use_container_width=True, hide_index=True)


# =========================
# UI
# =========================
st.title('競馬ランクアプリ v14.1 Multi Bet AI')
st.write('◎推奨馬ロジックは維持し、○▲は従来の上位評価寄り、△だけ「2着残り専用枠」に変更した完全版です。')
st.caption('v14.1: レース型に合わせて、単勝＋馬連、馬連＋ワイドなど複数買い目も自動選択します。金額は表示しません。')

if 'saved_recs' not in st.session_state:
    st.session_state.saved_recs = []

uploaded = st.file_uploader('1会場分の予想CSVをアップロード', type=['csv'])
current_recs = []

if uploaded is None:
    st.info('まず1会場6レース分のCSVを読み込んでください。')
else:
    prev3c_stat, prev4c_stat, prevtrack_stat, prev3c_file, prev4c_file = load_stat_defaults()
    df = prepare_race_df(read_csv_any(uploaded))

    st.caption(f'3角履歴ファイル: {prev3c_file} / 4角履歴ファイル: {prev4c_file}')

    df['本体点'] = (df['prevStraight'] * 0.30 + df['prev2Straight'] * 0.20).round(2)
    df['3角履歴係数'] = df.apply(lambda r: hist_coef_prev3c(r, prev3c_stat), axis=1)
    df['4角履歴係数'] = df.apply(lambda r: hist_coef_prev4c(r, prev4c_stat), axis=1)
    df['展開履歴係数'] = ((df['3角履歴係数'] + df['4角履歴係数']) / 2).round(2)
    df['展開予想係数'] = df['paceEval'].map(EVAL_MAP).fillna(1.0)
    df['展開位置補正'] = ((df['展開履歴係数'] + df['展開予想係数']) / 2).round(2)
    df['前走場所履歴係数'] = df.apply(lambda r: hist_coef_prevtrack(r, prevtrack_stat), axis=1)
    df['直線相性係数'] = df['straightEval'].map(EVAL_MAP).fillna(1.0)
    df['前走場所直線補正'] = ((df['前走場所履歴係数'] + df['直線相性係数']) / 2).round(2)
    df['総合点'] = (df['本体点'] * df['展開位置補正'] * df['前走場所直線補正']).round(2)
    df['トータルランク'] = df['総合点'].apply(total_rank)
    df = assign_relative_ranks(df)

    tab1, tab2, tab3, tab4 = st.tabs(['ランキング', 'おすすめ買い目', '保存・SNS画像', '通過順補正確認'])

    with tab1:
        for race_id in df['レース識別ID'].unique():
            g = df[df['レース識別ID'] == race_id].sort_values(['総合点', 'horseNo'], ascending=[False, True]).reset_index(drop=True)
            st.subheader(f'{g.iloc[0]["date"]} {g.iloc[0]["レース"]} {g.iloc[0]["raceName"]} / {g.iloc[0]["距離表示"]}')
            render_simple_ranking(g)
            st.divider()

    with tab2:
        for race_id in df['レース識別ID'].unique():
            g = df[df['レース識別ID'] == race_id].sort_values(['総合点', 'horseNo'], ascending=[False, True]).reset_index(drop=True)
            st.subheader(f'{g.iloc[0]["date"]} {g.iloc[0]["レース"]} {g.iloc[0]["raceName"]}')
            rec = recommend_for_race(g)
            honmei = rec['honmei']
            conf = rec['confidence']

            st.markdown('### 単複おすすめ1')
            st.write(f'候補: {int(honmei["horseNo"])} {honmei["horseName"]}')
            st.caption(f'トータル{honmei["トータルランク"]} / 総合点 {honmei["総合点"]:.2f} / 参考信頼度 {conf:.2f}% / {rec["short_comment"]}')

            st.markdown('### おすすめ馬券')
            if rec['status'] == '見送り':
                st.write('見送り')
                st.caption(f'理由：{rec["reason"]}')
            else:
                st.write(f'本命タイプ：{rec["honmei_type"]}')
                st.write(f'勝負度：{rec["bet_strength"]}')
                st.write(f'おすすめ馬券：{rec["bet_type"]}')

                if rec.get('umaren_bets'):
                    st.write('馬連：')
                    for bet, role in rec['umaren_bets']:
                        st.write(f'{bet}　{role}')
                if rec.get('trio_bets'):
                    st.write('三連複：')
                    for bet, role in rec['trio_bets']:
                        st.write(bet)
                if rec.get('wide_bets'):
                    st.write('ワイド：')
                    for bet, role in rec['wide_bets']:
                        st.write(f'{bet}　{role}')
                st.caption('理由：\n' + rec['reason'])

            st.divider()

            extra_values = []
            for role, m in rec.get('extra_mates', []):
                extra_values.append((role, int(m['horseNo']), m['horseName']))

            if extra_values:
                st.markdown('### 三連複3列目 追加候補')
                for role, no, name in extra_values:
                    st.write(f'他 {no} {name}　{role}')

            bet_values = [b[0] for b in rec.get('bets', [])]
            current_recs.append({
                '日付': honmei['date'],
                '場所': honmei['場所'],
                'R': safe_race_no(honmei),
                '馬番': int(honmei['horseNo']),
                '馬名': honmei['horseName'],
                '単複おすすめ1': f'{int(honmei["horseNo"])} {honmei["horseName"]}',
                '参考信頼度': round(float(conf), 2),
                '短評': rec['short_comment'],
                '買い対象': 1 if rec['status'] == '買い対象' else 0,
                '本命タイプ': rec['honmei_type'],
                '勝負度': rec['bet_strength'],
                'おすすめ馬券': rec['bet_type'],
                '自動券種': ' / '.join([b.get('auto_bet_kind', '') for b in rec.get('auto_bets', [])]),
                '自動形式': ' / '.join([b.get('auto_bet_form', '') for b in rec.get('auto_bets', [])]),
                '自動買い目': ' / '.join([b.get('auto_bet_main', '') for b in rec.get('auto_bets', [])]),
                '自動買い目詳細': ' / '.join([b.get('auto_bet_detail', '') for b in rec.get('auto_bets', [])]),
                '自動買い目表示': _format_auto_bets_display(rec.get('auto_bets', [])),
                '買い目表示': _format_auto_bets_display(rec.get('auto_bets', [])),
                '買い目1': bet_values[0] if len(bet_values) > 0 else '',
                '買い目2': bet_values[1] if len(bet_values) > 1 else '',
                '買い目3': bet_values[2] if len(bet_values) > 2 else '',
                '買い目4': bet_values[3] if len(bet_values) > 3 else '',
                '買い目5': bet_values[4] if len(bet_values) > 4 else '',
                '買い目6': bet_values[5] if len(bet_values) > 5 else '',
                '他1': extra_values[0][1] if len(extra_values) > 0 else '',
                '他1馬名': extra_values[0][2] if len(extra_values) > 0 else '',
                '他1役割': extra_values[0][0] if len(extra_values) > 0 else '',
                '他2': extra_values[1][1] if len(extra_values) > 1 else '',
                '他2馬名': extra_values[1][2] if len(extra_values) > 1 else '',
                '他2役割': extra_values[1][0] if len(extra_values) > 1 else '',
                '相手選定理由': rec['reason'],
            })

    with tab3:
        st.subheader('この会場の推奨馬')
        if current_recs:
            st.dataframe(pd.DataFrame(current_recs), use_container_width=True, hide_index=True)

        if st.button('この会場の推奨馬を保存', type='primary'):
            add_saved_recs(current_recs)
            st.success('この会場の単複おすすめ1を保存しました。')

        st.subheader('保存済み推奨馬')
        sdf = saved_df()
        if sdf.empty:
            st.info('まだ保存済み推奨馬はありません。')
        else:
            st.dataframe(sdf.sort_values(['日付', '場所', 'R']), use_container_width=True, hide_index=True)
            csv_out = sdf.to_csv(index=False, encoding='utf-8-sig')
            st.download_button('保存済み推奨馬CSVをダウンロード', data=csv_out.encode('utf-8-sig'), file_name='saved_recommendations.csv', mime='text/csv')

        st.subheader('第3アプリ用 直線ロジックCSV出力')
        race_options_df = df.groupby('レース識別ID').first().reset_index()
        race_options_df['重賞候補'] = race_options_df.apply(auto_grade_race_flag, axis=1)
        race_options_df['表示名'] = race_options_df.apply(lambda r: f'{r["date"]} {r["場所"]}{int(r["raceNo"])}R {r["raceName"]}', axis=1)
        default_ids = race_options_df[race_options_df['重賞候補']]['レース識別ID'].tolist()
        selected_grade_races = st.multiselect(
            '第3アプリへ出力する重賞レースを選択',
            options=race_options_df['レース識別ID'].tolist(),
            default=default_ids,
            format_func=lambda x: race_options_df.loc[race_options_df['レース識別ID'] == x, '表示名'].iloc[0],
        )
        third_export_df = build_third_app_straight_export(df, selected_grade_races)
        if third_export_df.empty:
            st.warning('第3アプリ用に出力する重賞レースが選択されていません。')
        else:
            st.dataframe(third_export_df, use_container_width=True, hide_index=True)
            third_csv = third_export_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button('第3アプリ用 直線ロジックCSVをダウンロード', data=third_csv.encode('utf-8-sig'), file_name='third_app_straight_logic.csv', mime='text/csv')

        col1, col2 = st.columns(2)
        with col1:
            if st.button('3会場まとめSNS画像を作成'):
                img = make_sns_image(st.session_state.saved_recs)
                if img is None:
                    st.warning('信頼度90%以上の推奨馬はありません')
                else:
                    st.image(img, caption='SNS投稿用画像', use_container_width=True)
                    st.download_button('SNS画像PNGをダウンロード', data=img.getvalue(), file_name='sns_recommendations.png', mime='image/png')
        with col2:
            if st.button('保存済み推奨馬をクリア'):
                st.session_state.saved_recs = []
                st.success('保存済み推奨馬をクリアしました。')

    with tab4:
        st.subheader('通過順補正確認')
        check_cols = ['date', 'レース', 'horseNo', 'horseName', 'prevFieldSize', 'prev3cPos', 'prev4cPos', 'prev3cRel_final', 'prev4cRel_final', 'prev3cCat_original', 'prev4cCat_original', 'prev3cCat', 'prev4cCat', '3角履歴係数', '4角履歴係数', '通過順補正']
        show_df = df[check_cols].rename(columns={
            'date': '日付', 'horseNo': '馬番', 'horseName': '馬名',
            'prevFieldSize': '前走頭数', 'prev3cPos': '前3角通過順', 'prev4cPos': '前4角通過順',
            'prev3cRel_final': '前3角相対位置', 'prev4cRel_final': '前4角相対位置',
            'prev3cCat_original': '前3角元カテゴリ', 'prev4cCat_original': '前4角元カテゴリ',
            'prev3cCat': '前3角補正後カテゴリ', 'prev4cCat': '前4角補正後カテゴリ',
        })
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        export_cols = ['date', '場所', 'レース', 'raceName', 'horseNo', 'horseName', 'prevFieldSize', 'prev3cPos', 'prev4cPos', 'prev3cRel_final', 'prev4cRel_final', 'prev3cCat', 'prev4cCat', '通過順補正', '相対評価', 'トータルランク', '本体点', '展開位置補正', '前走場所直線補正', '総合点']
        export_df = df[export_cols].rename(columns={
            'date': '日付', 'raceName': 'レース名', 'horseNo': '馬番', 'horseName': '馬名',
            'prevFieldSize': '前走頭数', 'prev3cPos': '前3角通過順', 'prev4cPos': '前4角通過順',
            'prev3cRel_final': '前3角相対位置', 'prev4cRel_final': '前4角相対位置',
            'prev3cCat': '前3角位置カテゴリ', 'prev4cCat': '前4角位置カテゴリ',
        })
        result_csv = export_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button('補正確認付き予想結果CSVをダウンロード', data=result_csv.encode('utf-8-sig'), file_name='keiba_rank_v128_second_place_results.csv', mime='text/csv')
