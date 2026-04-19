import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Keiba Rank App v6.8 New Logic', layout='wide')

DEFAULT_THRESHOLDS = pd.DataFrame([
    {'rank': 'S', 'min_score': 28.0, 'max_score': np.nan, 'label': '50%+'},
    {'rank': 'A', 'min_score': 24.0, 'max_score': 28.0, 'label': '30-50%'},
    {'rank': 'B', 'min_score': 22.0, 'max_score': 24.0, 'label': '20-30%'},
    {'rank': 'C', 'min_score': 20.0, 'max_score': 22.0, 'label': '10-20%'},
    {'rank': 'D', 'min_score': -999.0, 'max_score': 20.0, 'label': '<10%'},
])

REQUIRED_HIST_COLS = {
    'date', 'date_key', 'track', 'race_name', 'horse_name', 'sire', 'dam_sire', 'trainer',
    'surface', 'distance', 'distance_band', 'going_group', 'interval_category',
    'distance_change', 'track_change', 'jockey_change', 'vertical_score',
    'horizontal_score', 'total_score', 'rank', 'finish_position', 'placed'
}

REQUIRED_PRED_COLS = {
    'date_key', 'track', 'race_name', 'horse_name', 'sire', 'dam_sire', 'trainer',
    'surface', 'distance', 'distance_band', 'going_group', 'interval_category',
    'distance_change', 'track_change', 'jockey_change'
}


def read_csv_any(uploaded_file):
    data = uploaded_file.getvalue()
    for enc in ('utf-8-sig', 'cp932', 'utf-8'):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            continue
    raise ValueError('CSVを読み込めませんでした。UTF-8 または CP932 で保存してください。')


def normalize_prediction_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = REQUIRED_PRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f'予想CSVに不足している列: {sorted(missing)}')
    return df


def normalize_hist_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = REQUIRED_HIST_COLS - set(df.columns)
    if missing:
        raise ValueError(f'過去データCSVに不足している列: {sorted(missing)}')
    return df


def build_stats(hist: pd.DataFrame):
    hist = hist.copy()
    hist['placed'] = pd.to_numeric(hist['placed'], errors='coerce').fillna(0).astype(int)

    def agg(keys):
        out = hist.groupby(keys, dropna=False).agg(
            sample=('placed', 'size'),
            place_rate=('placed', 'mean')
        ).reset_index()
        return out

    stats = {
        'vt_track': agg(['sire', 'track']),
        'vt_dist': agg(['sire', 'distance_band']),
        'vt_going': agg(['sire', 'going_group']),
        'hz_track': agg(['trainer', 'track']),
        'hz_dist': agg(['trainer', 'distance_band']),
        'hz_interval': agg(['trainer', 'interval_category']),
        'hz_dchg': agg(['trainer', 'distance_change']),
        'hz_tchg': agg(['trainer', 'track_change']),
        'hz_jchg': agg(['trainer', 'jockey_change']),
    }

    base_place = hist['placed'].mean()
    return stats, float(base_place)


def smooth_score(rate, sample, base_rate, strength=12):
    if pd.isna(rate) or pd.isna(sample):
        return base_rate * 100
    return ((rate * sample) + (base_rate * strength)) / (sample + strength) * 100


def merge_feature(df, stat_df, keys, value_name):
    tmp = stat_df.copy()
    tmp[value_name] = [smooth_score(r, s, st.session_state.base_place_rate) for r, s in zip(tmp['place_rate'], tmp['sample'])]
    cols = keys + [value_name, 'sample']
    renamed = tmp[cols].rename(columns={'sample': value_name + '_sample'})
    return df.merge(renamed, on=keys, how='left')


def score_predictions(pred: pd.DataFrame, stats: dict, thresholds: pd.DataFrame):
    df = pred.copy()

    df = merge_feature(df, stats['vt_track'], ['sire', 'track'], 'vt_track_score')
    df = merge_feature(df, stats['vt_dist'], ['sire', 'distance_band'], 'vt_dist_score')
    df = merge_feature(df, stats['vt_going'], ['sire', 'going_group'], 'vt_going_score')
    df = merge_feature(df, stats['hz_track'], ['trainer', 'track'], 'hz_track_score')
    df = merge_feature(df, stats['hz_dist'], ['trainer', 'distance_band'], 'hz_dist_score')
    df = merge_feature(df, stats['hz_interval'], ['trainer', 'interval_category'], 'hz_interval_score')
    df = merge_feature(df, stats['hz_dchg'], ['trainer', 'distance_change'], 'hz_dchg_score')
    df = merge_feature(df, stats['hz_tchg'], ['trainer', 'track_change'], 'hz_tchg_score')
    df = merge_feature(df, stats['hz_jchg'], ['trainer', 'jockey_change'], 'hz_jchg_score')

    fill_score = st.session_state.base_place_rate * 100
    score_cols = [c for c in df.columns if c.endswith('_score')]
    for c in score_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(fill_score)

    df['vertical_score'] = df[['vt_track_score', 'vt_dist_score', 'vt_going_score']].mean(axis=1)
    df['horizontal_score'] = df[['hz_track_score', 'hz_dist_score', 'hz_interval_score', 'hz_dchg_score', 'hz_tchg_score', 'hz_jchg_score']].mean(axis=1)
    df['total_score'] = (df['vertical_score'] + df['horizontal_score']) / 2

    def assign_rank(x):
        for _, row in thresholds.iterrows():
            lo = row['min_score']
            hi = row['max_score']
            if pd.isna(hi):
                if x >= lo:
                    return row['rank']
            else:
                if x >= lo and x < hi:
                    return row['rank']
        return 'D'

    df['rank'] = df['total_score'].apply(assign_rank)
    df = df.sort_values(['date_key', 'track', 'race_name', 'total_score'], ascending=[True, True, True, False]).reset_index(drop=True)
    return df


def to_download_bytes(df):
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


st.title('競馬ランクアプリ v6.8 New Logic')
st.caption('見た目は6.8v系の想定で、内部ロジックを新2軸版に差し替えた再構成版です。')

with st.sidebar:
    st.header('データ')
    hist_file = st.file_uploader('過去データCSV（keiba_historical_ranked.csv）', type=['csv'], key='hist')
    pred_file = st.file_uploader('予想データCSV（テンプレ形式）', type=['csv'], key='pred')
    th_file = st.file_uploader('ランク基準CSV（任意）', type=['csv'], key='thr')

    st.header('補正')
    strength = st.slider('母数スムージング強度', min_value=1, max_value=50, value=12)
    st.session_state['strength'] = strength

    st.markdown('---')
    st.write('同梱サンプルファイルを使う場合は、GitHub配置後に `sample_data` フォルダを参照してください。')

sample_dir = Path(__file__).parent / 'sample_data'
default_hist_path = sample_dir / 'keiba_historical_ranked.csv'
default_thr_path = sample_dir / 'keiba_rank_thresholds.csv'
default_pred_path = sample_dir / 'keiba_prediction_input_template.csv'

hist_df = None
thr_df = DEFAULT_THRESHOLDS.copy()
pred_df = None

if hist_file is not None:
    hist_df = normalize_hist_df(read_csv_any(hist_file))
elif default_hist_path.exists():
    hist_df = normalize_hist_df(pd.read_csv(default_hist_path))

if th_file is not None:
    raw_thr = read_csv_any(th_file)
    col_map = {'ランク': 'rank', '下限点': 'min_score', '上限点': 'max_score'}
    raw_thr = raw_thr.rename(columns=col_map)
    if {'rank', 'min_score', 'max_score'}.issubset(raw_thr.columns):
        thr_df = raw_thr[['rank', 'min_score', 'max_score']].copy()
        thr_df['label'] = ''
    else:
        st.warning('ランク基準CSVの列が想定と違うため、既定値を使用します。')
elif default_thr_path.exists():
    raw_thr = pd.read_csv(default_thr_path)
    raw_thr = raw_thr.rename(columns={'ランク': 'rank', '下限点': 'min_score', '上限点': 'max_score'})
    thr_df = raw_thr[['rank', 'min_score', 'max_score']].copy()
    thr_df['label'] = ''

if pred_file is not None:
    pred_df = normalize_prediction_df(read_csv_any(pred_file))
elif default_pred_path.exists():
    pred_df = pd.read_csv(default_pred_path)

if hist_df is None:
    st.info('左のサイドバーから過去データCSVを読み込んでください。')
    st.stop()

stats, base_place = build_stats(hist_df)
st.session_state.base_place_rate = base_place

# re-run score smoothing with selected strength by overriding helper closure use

def merge_feature(df, stat_df, keys, value_name):
    tmp = stat_df.copy()
    tmp[value_name] = [smooth_score(r, s, st.session_state.base_place_rate, st.session_state['strength']) for r, s in zip(tmp['place_rate'], tmp['sample'])]
    cols = keys + [value_name, 'sample']
    renamed = tmp[cols].rename(columns={'sample': value_name + '_sample'})
    return df.merge(renamed, on=keys, how='left')

# Monkey patch local function used below
locals()['merge_feature'] = merge_feature

hist_tab, pred_tab, guide_tab = st.tabs(['過去データ', '予想ランク', '使い方'])

with hist_tab:
    c1, c2, c3 = st.columns(3)
    c1.metric('過去データ件数', f"{len(hist_df):,}")
    c2.metric('基準複勝率', f"{base_place*100:.1f}%")
    c3.metric('登録ランク種類', str(hist_df['rank'].nunique()))

    st.subheader('過去データプレビュー')
    st.dataframe(hist_df.head(50), use_container_width=True)

    if 'rank' in hist_df.columns:
        summary = hist_df.groupby('rank', dropna=False).agg(
            件数=('placed', 'size'),
            複勝率=('placed', 'mean'),
            平均総合点=('total_score', 'mean')
        ).reset_index().sort_values('平均総合点', ascending=False)
        summary['複勝率'] = (summary['複勝率'] * 100).round(1)
        st.subheader('過去データのランク別実績')
        st.dataframe(summary, use_container_width=True)

with pred_tab:
    st.subheader('ランク基準')
    edit_thr = st.data_editor(thr_df, num_rows='fixed', use_container_width=True)

    if pred_df is None or pred_df.empty:
        st.info('予想データCSVを読み込むと、ここにランク結果が表示されます。')
    else:
        scored = score_predictions(pred_df, stats, edit_thr)
        st.success('予想データを新2軸ロジックでランク付けしました。')

        left, right = st.columns([2, 1])
        with left:
            st.dataframe(scored, use_container_width=True, height=520)
        with right:
            rank_summary = scored.groupby('rank', dropna=False).size().reset_index(name='頭数').sort_values('rank')
            st.dataframe(rank_summary, use_container_width=True)
            st.download_button(
                'ランク付け済みCSVをダウンロード',
                data=to_download_bytes(scored),
                file_name='keiba_prediction_ranked.csv',
                mime='text/csv'
            )

with guide_tab:
    st.markdown('''
### 使い方
1. 過去データCSVとして `keiba_historical_ranked.csv` を読み込みます。  
2. 予想データCSVとして `keiba_prediction_input_template.csv` 形式のファイルを読み込みます。  
3. 必要ならランク基準CSVを読み込むか、画面上で閾値を編集します。  
4. 自動で `vertical_score` `horizontal_score` `total_score` `rank` を計算します。

### 予想データCSVの必須列
- date_key
- track
- race_name
- horse_name
- sire
- dam_sire
- trainer
- surface
- distance
- distance_band
- going_group
- interval_category
- distance_change
- track_change
- jockey_change

### メモ
- 前回6.8vの見た目を意識しつつ、中身は直線ロジックではなく新2軸ロジックに置き換えています。
- 既定では、過去データの複勝率を母数でスムージングしてスコア化しています。
''')
