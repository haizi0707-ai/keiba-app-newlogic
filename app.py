import os
import unicodedata
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='競馬ロジック検証アプリ v1', layout='centered')

# ---------- Utility ----------

def read_csv_any(file_obj_or_path):
    encodings = ['utf-8-sig', 'cp932', 'shift_jis', 'utf-8']
    last_err = None
    for enc in encodings:
        try:
            if hasattr(file_obj_or_path, 'seek'):
                file_obj_or_path.seek(0)
            return pd.read_csv(file_obj_or_path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def norm_text(v) -> str:
    if pd.isna(v):
        return ''
    s = unicodedata.normalize('NFKC', str(v)).strip()
    return ' '.join(s.split())


def norm_track(v) -> str:
    s = norm_text(v)
    replace_map = {
        '東京競馬場': '東京', '中山競馬場': '中山', '中京競馬場': '中京', '阪神競馬場': '阪神',
        '京都競馬場': '京都', '新潟競馬場': '新潟', '福島競馬場': '福島', '小倉競馬場': '小倉',
        '札幌競馬場': '札幌', '函館競馬場': '函館',
    }
    return replace_map.get(s, s)


def norm_surface(v) -> str:
    s = norm_text(v)
    if s.startswith('芝'):
        return '芝'
    if s.startswith('ダ') or s.startswith('ダート'):
        return 'ダ'
    if s.startswith('障'):
        return '障'
    return {'ダート': 'ダ'}.get(s, s)


def norm_style(v) -> str:
    s = norm_text(v)
    mp = {
        '逃': '逃げ', '逃げ': '逃げ',
        '先': '先行', '先行': '先行',
        '差': '差し', '差し': '差し',
        '追': '追込', '追込': '追込', '追い込み': '追込',
    }
    return mp.get(s, s)


def norm_winstyle(v) -> str:
    s = norm_text(v)
    mp = {
        '逃げ切り': '逃げ切り',
        '先行押し切り': '先行押し切り',
        '好位差し': '好位差し',
        '差し': '差し',
        '追い込み': '追い込み',
        '追込': '追い込み',
        'まくり': 'まくり',
    }
    return mp.get(s, s)


def to_int(v):
    if pd.isna(v):
        return np.nan
    s = ''.join(ch for ch in norm_text(v) if ch.isdigit())
    return pd.to_numeric(s, errors='coerce')


def rename_first_match(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    out = df.copy()
    cols_norm = {norm_text(c): c for c in out.columns}
    for target, candidates in mapping.items():
        if target in out.columns:
            continue
        for cand in candidates:
            actual = cols_norm.get(norm_text(cand))
            if actual is not None:
                out = out.rename(columns={actual: target})
                break
    return out


MASTER_MAP = {
    '日付': ['日付', '日付S', 'date'],
    '場所': ['場所', 'track'],
    '芝ダ': ['芝ダ', '芝・ダ', 'surface'],
    '距離': ['距離', 'distance'],
    'R': ['R', 'raceNo', 'レース番号'],
    'レース名': ['レース名', 'raceName', 'レース'],
    '馬番': ['馬番', 'horseNo'],
    '馬名': ['馬名', 'horseName'],
    '騎手': ['騎手', 'jockey'],
    '調教師': ['調教師', 'trainer'],
    '血統': ['血統', '種牡馬', 'sire'],
    '母父馬': ['母父馬', 'damSire'],
    '馬場状態': ['馬場状態', 'going', '馬場'],
    '着順数値': ['着順数値', 'finish_rank', '着順'],
    'win_flag': ['win_flag'],
    'place_flag': ['place_flag'],
    '脚質': ['脚質', 'style', '脚質タグ'],
    '前走場所': ['前走場所', 'prevTrack', '前開催', '前走競馬場', '前走場所タグ'],
    '勝ち方': ['勝ち方', '勝ち方タグ', 'winStyle'],
}

RUNNER_MAP = {
    '場所': ['場所', 'track'],
    '芝ダ': ['芝ダ', '芝・ダ', 'surface'],
    '距離': ['距離', 'distance'],
    'レース': ['レース', 'raceName', 'レース名'],
    'raceNo': ['raceNo', 'R', 'レース番号'],
    '馬番': ['馬番', 'horseNo'],
    '馬名': ['馬名', 'horseName'],
    '血統': ['血統', '種牡馬', 'sire'],
    '調教師': ['調教師', 'trainer'],
    '脚質': ['脚質', 'style'],
    '前走場所': ['前走場所', 'prevTrack', '前開催'],
    '勝ち方': ['勝ち方', 'winStyle'],
}

RESULT_MAP = {
    'レース': ['レース', 'raceName', 'レース名'],
    '馬名': ['馬名', 'horseName'],
    '着順': ['着順', 'finish_rank', '実着順'],
}

SPECS = [
    ('血統', '血統', '血統点', '血統母数'),
    ('調教師', '調教師', '調教師点', '調教師母数'),
    ('脚質', '脚質', '脚質点', '脚質母数'),
    ('前走場所', '前走場所', '前走場所点', '前走場所母数'),
    ('勝ち方', '勝ち方', '勝ち方点', '勝ち方母数'),
]


def prepare_master(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_first_match(df, MASTER_MAP)
    for col in MASTER_MAP:
        if col not in df.columns:
            df[col] = ''

    df['場所'] = df['場所'].apply(norm_track)
    df['芝ダ'] = df['芝ダ'].apply(norm_surface)
    df['距離'] = pd.to_numeric(df['距離'], errors='coerce')
    df['馬名'] = df['馬名'].apply(norm_text)
    df['調教師'] = df['調教師'].apply(norm_text)
    df['血統'] = df['血統'].apply(norm_text)
    df['脚質'] = df['脚質'].apply(norm_style)
    df['前走場所'] = df['前走場所'].apply(norm_track)
    df['勝ち方'] = df['勝ち方'].apply(norm_winstyle)
    df['着順数値'] = pd.to_numeric(df['着順数値'], errors='coerce')

    if 'win_flag' not in df.columns or df['win_flag'].astype(str).eq('').all():
        df['win_flag'] = (df['着順数値'] == 1).astype(int)
    else:
        df['win_flag'] = pd.to_numeric(df['win_flag'], errors='coerce').fillna(0).astype(int)

    if 'place_flag' not in df.columns or df['place_flag'].astype(str).eq('').all():
        df['place_flag'] = df['着順数値'].between(1, 3, inclusive='both').fillna(False).astype(int)
    else:
        df['place_flag'] = pd.to_numeric(df['place_flag'], errors='coerce').fillna(0).astype(int)

    return df


def prepare_runners(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_first_match(df, RUNNER_MAP)
    for col in RUNNER_MAP:
        if col not in df.columns:
            df[col] = ''

    df['場所'] = df['場所'].apply(norm_track)
    df['芝ダ'] = df['芝ダ'].apply(norm_surface)
    # Allow old format distance like 芝1200 only if 芝ダ/距離 missing
    if 'distance' in df.columns:
        old_dist = df['distance'].apply(norm_text)
        fill_surface = old_dist.apply(norm_surface)
        fill_dist = old_dist.apply(to_int)
        df['芝ダ'] = np.where(df['芝ダ'].astype(str).str.strip() != '', df['芝ダ'], fill_surface)
        df['距離'] = np.where(pd.to_numeric(df['距離'], errors='coerce').notna(), pd.to_numeric(df['距離'], errors='coerce'), fill_dist)
    else:
        df['距離'] = pd.to_numeric(df['距離'], errors='coerce')
    df['距離'] = pd.to_numeric(df['距離'], errors='coerce')

    for col in ['レース', '馬名', '血統', '調教師']:
        df[col] = df[col].apply(norm_text)
    df['脚質'] = df['脚質'].apply(norm_style)
    df['前走場所'] = df['前走場所'].apply(norm_track)
    df['勝ち方'] = df['勝ち方'].apply(norm_winstyle)
    return df


def prepare_results(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_first_match(df, RESULT_MAP)
    for col in RESULT_MAP:
        if col not in df.columns:
            df[col] = ''
    df['レース'] = df['レース'].apply(norm_text)
    df['馬名'] = df['馬名'].apply(norm_text)
    df['着順'] = pd.to_numeric(df['着順'], errors='coerce')
    return df[['レース', '馬名', '着順']].copy()


@st.cache_data(show_spinner=False)
def build_stats(master_df: pd.DataFrame, attr_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_cols = ['場所', '芝ダ', '距離']
    attr = attr_col
    use = master_df[base_cols + [attr, 'win_flag', 'place_flag']].copy()
    use[attr] = use[attr].apply(norm_text)
    if attr in ['前走場所']:
        use[attr] = use[attr].apply(norm_track)
    if attr in ['脚質']:
        use[attr] = use[attr].apply(norm_style)
    if attr in ['勝ち方']:
        use[attr] = use[attr].apply(norm_winstyle)

    detailed = (
        use[use[attr] != '']
        .groupby(base_cols + [attr], dropna=False)
        .agg(母数=('place_flag', 'size'), 勝利数=('win_flag', 'sum'), 複勝数=('place_flag', 'sum'))
        .reset_index()
    )
    detailed['単勝率'] = detailed['勝利数'] / detailed['母数']
    detailed['複勝率'] = detailed['複勝数'] / detailed['母数']
    detailed['補正複勝率'] = (detailed['複勝数'] + 1) / (detailed['母数'] + 3)

    fallback = (
        use.groupby(base_cols, dropna=False)
        .agg(条件母数=('place_flag', 'size'), 条件勝利数=('win_flag', 'sum'), 条件複勝数=('place_flag', 'sum'))
        .reset_index()
    )
    fallback['条件平均複勝率'] = fallback['条件複勝数'] / fallback['条件母数']
    fallback['条件補正複勝率'] = (fallback['条件複勝数'] + 1) / (fallback['条件母数'] + 3)
    return detailed, fallback


def score_runners(runners: pd.DataFrame, master_df: pd.DataFrame, weights: Dict[str, int]) -> pd.DataFrame:
    out = runners.copy()
    contribution_cols = []

    for attr, key_col, score_col, count_col in SPECS:
        detail, fallback = build_stats(master_df, key_col)
        detail = detail.rename(columns={key_col: f'{key_col}_key'})
        fallback_cols = ['場所', '芝ダ', '距離', '条件補正複勝率', '条件母数']
        merged = out.merge(
            detail[['場所', '芝ダ', '距離', f'{key_col}_key', '補正複勝率', '母数']],
            left_on=['場所', '芝ダ', '距離', key_col],
            right_on=['場所', '芝ダ', '距離', f'{key_col}_key'],
            how='left'
        ).merge(
            fallback[fallback_cols],
            on=['場所', '芝ダ', '距離'],
            how='left'
        )
        rate = merged['補正複勝率'].where(merged['補正複勝率'].notna(), merged['条件補正複勝率'])
        mothers = merged['母数'].where(merged['母数'].notna(), merged['条件母数'])
        out[count_col] = mothers.fillna(0).astype(float).round(0).astype(int)
        out[score_col] = (rate.fillna(0) * weights[attr]).round(2)
        contribution_cols.append(score_col)

    out['総合点'] = out[contribution_cols].sum(axis=1).round(2)
    out['最大加点項目'] = out[contribution_cols].idxmax(axis=1).str.replace('点', '', regex=False)
    out['最大加点値'] = out[contribution_cols].max(axis=1).round(2)
    out['最小加点項目'] = out[contribution_cols].idxmin(axis=1).str.replace('点', '', regex=False)
    out['最小加点値'] = out[contribution_cols].min(axis=1).round(2)

    race_keys = ['場所', '芝ダ', '距離']
    if 'レース' in out.columns and out['レース'].astype(str).str.strip().ne('').any():
        race_keys = ['場所', '芝ダ', '距離', 'レース']
    out['順位'] = out.groupby(race_keys)['総合点'].rank(method='min', ascending=False).astype(int)
    return out


def add_results(scored: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if results is None or results.empty:
        return scored
    out = scored.copy()
    merge_keys = ['馬名']
    if 'レース' in out.columns and out['レース'].astype(str).str.strip().ne('').any() and results['レース'].astype(str).str.strip().ne('').any():
        merge_keys = ['レース', '馬名']
    out = out.merge(results, on=merge_keys, how='left')
    out['複勝圏'] = np.where(out['着順'].between(1, 3, inclusive='both'), '○', '')
    out['1着'] = np.where(out['着順'] == 1, '○', '')
    return out


def summary_by_race(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ['場所', '芝ダ', '距離']
    if 'レース' in df.columns and df['レース'].astype(str).str.strip().ne('').any():
        group_cols.append('レース')
    agg = df.groupby(group_cols, dropna=False).agg(
        頭数=('馬名', 'size'),
        予想1位複勝=('複勝圏', lambda s: int((s == '○').sum()) if len(s) else 0),
        予想1位勝利=('1着', lambda s: int((s == '○').sum()) if len(s) else 0),
    ).reset_index()
    return agg


# ---------- UI ----------
st.title('競馬ロジック検証アプリ v1')
st.caption('3年分元データから全頭診断・項目寄与確認・結果照合を行う検証専用版')

with st.expander('使い方', expanded=False):
    st.markdown(
        '- 元データCSVを読み込みます。\n'
        '- 出走馬CSVを読み込みます。\n'
        '- 重みを調整すると総合点が即時再計算されます。\n'
        '- 結果CSVを入れると着順照合ができます。'
    )

col1, col2 = st.columns(2)
with col1:
    master_file = st.file_uploader('① 元データCSV（3年分）', type=['csv'], key='master')
    runners_file = st.file_uploader('② 出走馬CSV', type=['csv'], key='runners')
with col2:
    results_file = st.file_uploader('③ 結果CSV（任意）', type=['csv'], key='results')
    st.markdown('**重み設定（合計100推奨）**')
    weights = {
        '血統': st.slider('血統', 0, 100, 20, 1),
        '調教師': st.slider('調教師', 0, 100, 20, 1),
        '脚質': st.slider('脚質', 0, 100, 20, 1),
        '前走場所': st.slider('前走場所', 0, 100, 20, 1),
        '勝ち方': st.slider('勝ち方', 0, 100, 20, 1),
    }
    st.caption(f"現在の合計: {sum(weights.values())}")

if not master_file or not runners_file:
    st.info('元データCSVと出走馬CSVをアップロードしてください。')
    st.stop()

try:
    master_raw = read_csv_any(master_file)
    runners_raw = read_csv_any(runners_file)
    results_raw = read_csv_any(results_file) if results_file else None
except Exception as e:
    st.error(f'CSVの読み込みに失敗しました: {e}')
    st.stop()

master_df = prepare_master(master_raw)
runners_df = prepare_runners(runners_raw)
results_df = prepare_results(results_raw) if results_raw is not None else None

with st.expander('読み込み確認', expanded=False):
    st.write('元データ件数:', len(master_df))
    st.write('出走馬件数:', len(runners_df))
    if results_df is not None:
        st.write('結果件数:', len(results_df))
    st.write('元データ列:', list(master_raw.columns))
    st.write('出走馬列:', list(runners_raw.columns))

scored = score_runners(runners_df, master_df, weights)
scored = add_results(scored, results_df)

# Tabs
main_tab, effect_tab, result_tab, export_tab = st.tabs(['全頭診断', '効いた項目', '結果照合', 'CSV出力'])

with main_tab:
    show_cols = [
        c for c in ['順位', 'レース', '馬番', '馬名', '血統点', '調教師点', '脚質点', '前走場所点', '勝ち方点',
                    '総合点', '血統母数', '調教師母数', '脚質母数', '前走場所母数', '勝ち方母数'] if c in scored.columns
    ]
    display_df = scored.sort_values(['順位', '総合点'], ascending=[True, False])[show_cols]
    st.dataframe(display_df, use_container_width=True, height=700)

with effect_tab:
    effect_cols = [c for c in ['順位', 'レース', '馬名', '最大加点項目', '最大加点値', '最小加点項目', '最小加点値',
                               '血統点', '調教師点', '脚質点', '前走場所点', '勝ち方点', '総合点'] if c in scored.columns]
    st.dataframe(scored.sort_values(['順位', '総合点'], ascending=[True, False])[effect_cols], use_container_width=True, height=700)
    avg_cols = ['血統点', '調教師点', '脚質点', '前走場所点', '勝ち方点', '総合点']
    st.subheader('レース全体の平均点')
    st.dataframe(scored[avg_cols].mean().round(2).rename('平均').to_frame().T, use_container_width=True)

with result_tab:
    if results_df is None or results_df.empty:
        st.info('結果CSVを入れると実着順との照合が表示されます。')
    else:
        result_cols = [c for c in ['順位', 'レース', '馬名', '総合点', '着順', '複勝圏', '1着', '最大加点項目', '最大加点値'] if c in scored.columns]
        st.dataframe(scored.sort_values(['順位', '着順'], ascending=[True, True])[result_cols], use_container_width=True, height=700)
        top1 = scored[scored['順位'] == 1].copy()
        if not top1.empty:
            top1_summary = {
                '予想1位頭数': len(top1),
                '予想1位の勝利数': int((top1['着順'] == 1).sum()),
                '予想1位の複勝数': int(top1['着順'].between(1, 3, inclusive='both').sum()),
                '予想1位単勝率': round((top1['着順'] == 1).mean(), 4) if len(top1) else 0,
                '予想1位複勝率': round(top1['着順'].between(1, 3, inclusive='both').mean(), 4) if len(top1) else 0,
            }
            st.subheader('予想1位の成績')
            st.json(top1_summary)

with export_tab:
    export_df = scored.copy()
    st.dataframe(export_df, use_container_width=True, height=500)
    csv_bytes = export_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button('診断結果CSVをダウンロード', data=csv_bytes, file_name='validation_scored_output.csv', mime='text/csv')
