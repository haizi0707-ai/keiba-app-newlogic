# 競馬ランクアプリ v6.8 New Logic

前回の6.8v系アプリの見た目を踏襲しつつ、内部ロジックを新2軸版へ差し替えた Streamlit アプリです。

## 新ロジック
- 縦軸: 血統 × 競馬場 / 距離帯 / 馬場
- 横軸: 厩舎 × 競馬場 / 距離帯 / 間隔 / 距離変更 / 場替わり / 騎手継続
- 合成点から S〜D ランク付け

## 起動方法
```bash
pip install -r requirements.txt
streamlit run app.py
```

## GitHub / Streamlit Community Cloud
- そのまま GitHub にアップロード可能です。
- `sample_data` フォルダにサンプルCSVを同梱しています。
- Streamlit Community Cloud ではリポジトリを指定してデプロイしてください。

## 同梱ファイル
- `sample_data/keiba_historical_ranked.csv`
- `sample_data/keiba_rank_thresholds.csv`
- `sample_data/keiba_prediction_input_template.csv`

## 予想データ必須列
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
