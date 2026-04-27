import os
import re
import io
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageDraw, ImageFont

# --- 1. Streamlit 設定 ---
st.set_page_config(page_title="競馬ランクアプリ v13.0 AI-Multiplier", layout="wide")

# サイドバーでAPIキーを取得
st.sidebar.title("🛠 AI設定")
# Secretsから取得を試み、なければ入力させる
api_key_input = st.sidebar.text_input("Gemini API Keyを入力", type="password", value=st.secrets.get("GEMINI_API_KEY", ""))

if api_key_input:
    genai.configure(api_key=api_key_input)
else:
    st.sidebar.warning("APIキーを入力してください。")

# --- 2. 定数・初期設定 ---
EVAL_MAP = {"かなり向く": 1.25, "向く": 1.10, "普通": 1.00, "やや不向き": 0.85, "不向き": 0.70}

# --- 3. AIデータ取得ロジック (指定プロンプト統合) ---
def fetch_race_data_ai(date_str, place_name, race_no, baba, tb, mode="作成"):
    if not api_key_input:
        st.error("APIキーが設定されていません。")
        return None
    
    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash',
        tools=[{"google_search": {}}]
    )
    
    if mode == "作成":
        prompt = f'''
🔥あなたは競馬予想データ作成AIです。

ユーザーが指定するレースについて、
画像は使わず、ネット検索で取得できる最新の公開情報をもとに、
「競馬ランクアプリ v12.0 Multiplier Logic」にそのまま読み込める
高精度な出走馬CSVを作成してください。

【最重要ルール】
・必ずネット検索で必要情報を集めること
・画像は使わないこと
・最終出力は必ず .csvファイル にすること
・オッズ、人気による補正はしないこと
・推測で雑に埋めないこと
・ただし、主要列は複数ページを横断してでも最大限埋めること
・1つのページで足りない場合は、別ページを検索して補完すること
・提出前に空欄チェックを行い、空欄が残る列は再検索して補完すること
・最後の成果物は必ず .csvファイル にすること

【対象レース】
{date_str} {place_name} {race_no}R
想定条件: 馬場={{baba}}, 傾向={{tb}}

【出力CSVの列】
日付,場所,芝ダ,距離,レース,レース名,馬番,馬名,前走競馬場,前走芝ダ,前走距離数値,前3角位置カテゴリ,前4角位置カテゴリ,前走場所,前走直線ロジック点,前々走直線ロジック点,展開予想評価,直線相性評価

【必須方針】
この作業は、必ず以下の4段階で行うこと。
① 出馬表ページでレース基本情報を全頭取得
② 各馬近走成績ページで前走・前々走内容を確認
③ 前走の競馬場・芝ダ・距離・3角位置カテゴリ・4角位置カテゴリ・前走場所を埋める
④ 直線ロジック点、展開予想評価、直線相性評価を判定する

【評価ロジック】
・本体点 = 前走直線ロジック点×0.30 + 前々走直線ロジック点×0.20
・最終点 = 本体点 × 展開位置補正 × 前走場所直線補正

【直線ロジック点のルール】
0〜100点で数値化。90-100(特注)、75-89(強)、60-74(良)、45-59(並)、30-44(弱)、0-29(極弱)

【展開・直線相性評価のルール】
「かなり向く」「向く」「普通」「やや不向き」「不向き」の5段階。

【出力ルール】
・対象レースの全頭を1行ずつCSV化すること。
・CSVテキストデータのみを返し、解説文や ```csv 等の装飾は一切不要です。
'''
    else:
        prompt = f'''
🔥あなたは競馬当日補正AIです。
前日版の評価をベースにして、当日の馬場傾向・トラックバイアス（TB）を加味して補正値だけ微調整してください。

【対象レース】
{date_str} {place_name} {race_no}R
当日の条件：馬場={{baba}}, トラックバイアス={{tb}}

【更新対象】
・展開予想評価
・直線相性評価
(5段階：かなり向く、向く、普通、やや不向き、不向き)

【出力ルール】
・既存のCSV列順を維持し、全頭分出力してください。
・CSVテキストデータのみを返し、解説文や ```csv 等の装飾は一切不要です。
'''

    try:
        response = model.generate_content(prompt)
        res_text = response.text
        # クリーンアップ
        res_text = re.sub(r'^```(csv)?\s*', '', res_text, flags=re.MULTILINE)
        res_text = re.sub(r'^```\s*', '', res_text, flags=re.MULTILINE)
        return res_text.strip()
    except Exception as e:
        st.error(f"AI処理中にエラーが発生しました: {{e}}")
        return None

# --- 4. 計算ロジック ---
def calculate_logic(df):
    # 数値変換
    df["前走直線ロジック点"] = pd.to_numeric(df["前走直線ロジック点"], errors='coerce').fillna(0)
    df["前々走直線ロジック点"] = pd.to_numeric(df["前々走直線ロジック点"], errors='coerce').fillna(0)
    
    # 計算
    df["本体点"] = (df["前走直線ロジック点"] * 0.30) + (df["前々走直線ロジック点"] * 0.20)
    df["展開係数"] = df["展開予想評価"].map(EVAL_MAP).fillna(1.0)
    df["相性係数"] = df["直線相性評価"].map(EVAL_MAP).fillna(1.0)
    
    df["最終スコア"] = df["本体点"] * df["展開係数"] * df["相性係数"]
    return df.sort_values("最終スコア", ascending=False)

# --- 5. UI設定 ---
def main():
    st.title("🏇 AI-Multiplier Logic v13.0")
    
    tab1, tab2 = st.tabs(["AI自動生成モード", "CSV手動モード"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        date_str = col1.text_input("開催日", "2026.4.25")
        place_str = col2.selectbox("場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "福島", "小倉", "札幌", "函館"])
        race_no = col3.number_input("レース番号", 1, 12, 11)
        
        baba = st.radio("馬場状態", ["良", "稍重", "重", "不良"], horizontal=True)
        tb = st.text_input("当日の傾向/TB", "先行有利・内枠有利")
        
        mode_choice = st.radio("AI動作モード", ["前日：新規データ作成", "当日：馬場/TB補正"], horizontal=True)
        
        if st.button("AI予想・補正を実行", type="primary"):
            mode_tag = "作成" if "新規" in mode_choice else "補正"
            with st.spinner("AIが情報を収集・分析中..."):
                csv_data = fetch_race_data_ai(date_str, place_str, race_no, baba, tb, mode_tag)
                if csv_data:
                    try:
                        df = pd.read_csv(io.StringIO(csv_data))
                        res_df = calculate_logic(df)
                        
                        st.subheader("📊 予想ランキング")
                        st.dataframe(res_df[["馬番", "馬名", "最終スコア", "本体点", "展開予想評価", "直線相性評価"]])
                        
                        # ダウンロードボタン
                        st.download_button("結果CSVをダウンロード", csv_data, file_name=f"race_{place_str}{race_no}R.csv", mime="text/csv")
                    except Exception as e:
                        st.error("CSVの解析に失敗しました。AIの出力を確認してください。")
                        st.text(csv_data)

    with tab2:
        up = st.file_uploader("作成済みCSVをアップロード", type="csv")
        if up:
            df_up = pd.read_csv(up)
            res_up = calculate_logic(df_up)
            st.dataframe(res_up)

if __name__ == "__main__":
    main()
