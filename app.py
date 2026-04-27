import os
import re
import io
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai

# --- 1. 設定 ---
st.set_page_config(page_title="AI-Multiplier v13.0 全自動版", layout="wide")

# APIキー設定（サイドバー、またはStreamlit Secretsから）
api_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.secrets.get("GEMINI_API_KEY", ""))
if api_key:
    genai.configure(api_key=api_key)

EVAL_MAP = {"かなり向く": 1.25, "向く": 1.10, "普通": 1.00, "やや不向き": 0.85, "不向き": 0.70}

# --- 2. AI検索・データ生成関数 ---
def fetch_full_auto_data(date_str, place_name, race_no):
    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash',
        tools=[{"google_search": {}}]
    )
    
    # AIへの全自動指示プロンプト
    prompt = f"""
あなたは競馬分析エキスパートです。以下のステップで作業してください。

【ステップ1：情報収集】
ネット検索を駆使して、{date_str}の{place_name}競馬場の「現在の馬場状態（良・稍重など）」と、当日ここまでのレース結果から見える「トラックバイアス（内有利、差し有利など）」の最新情報を取得してください。

【ステップ2：CSV生成】
取得した馬場・TB情報を踏まえ、{race_no}Rの出走馬について、あなたのMultiplier Logic（直線ロジック点、展開・相性評価）を適用したCSVを作成してください。

【必須ルール】
・画像は使わずネット検索の結果のみを使うこと。
・出力は以下のCSV形式のみ。解説文は一切不要です。
・列順：日付,場所,芝ダ,距離,レース,レース名,馬番,馬名,前走競馬場,前走芝ダ,前走距離数値,前3角位置カテゴリ,前4角位置カテゴリ,前走場所,前走直線ロジック点,前々走直線ロジック点,展開予想評価,直線相性評価

【対象レース】
{date_str} {place_name} {race_no}R
"""

    with st.spinner("AIが今日の馬場とTBをネット検索中..."):
        response = model.generate_content(prompt)
        res_text = response.text
        # 不要な装飾を削除
        res_text = re.sub(r'^```(csv)?\s*', '', res_text, flags=re.MULTILINE)
        res_text = re.sub(r'^```\s*', '', res_text, flags=re.MULTILINE).strip()
        return res_text

# --- 3. 計算ロジック ---
def calculate_multiplier(df):
    df["前走直線ロジック点"] = pd.to_numeric(df["前走直線ロジック点"], errors='coerce').fillna(50)
    df["前々走直線ロジック点"] = pd.to_numeric(df["前々走直線ロジック点"], errors='coerce').fillna(50)
    df["本体点"] = (df["前走直線ロジック点"] * 0.30) + (df["前々走直線ロジック点"] * 0.20)
    df["展開係数"] = df["展開予想評価"].map(EVAL_MAP).fillna(1.0)
    df["相性係数"] = df["直線相性評価"].map(EVAL_MAP).fillna(1.0)
    df["最終点"] = df["本体点"] * df["展開係数"] * df["相性係数"]
    return df.sort_values("最終点", ascending=False)

# --- 4. メインUI ---
st.title("🏇 AI-Multiplier Logic v13.0 (全自動検索版)")

col1, col2, col3 = st.columns(3)
d_val = col1.text_input("開催日", "2026.4.27")
p_val = col2.selectbox("場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "福島", "小倉", "札幌", "函館"])
r_val = col3.number_input("レース番号", 1, 12, 11)

if st.button("全自動予想（馬場・TB検索込み）を実行", type="primary"):
    if not api_key:
        st.error("左側のサイドバーにAPIキーを入力してください。")
    else:
        csv_data = fetch_full_auto_data(d_val, p_val, r_val)
        try:
            df = pd.read_csv(io.StringIO(csv_data))
            res_df = calculate_multiplier(df)
            
            st.success(f"AIが当日の{p_val}の傾向を読み取り、分析を完了しました。")
            st.dataframe(res_df[["馬番", "馬名", "最終点", "本体点", "展開予想評価", "直線相性評価"]])
            
            st.download_button("分析済みCSVを保存", csv_data, f"{p_val}{r_val}R_result.csv")
        except Exception as e:
            st.error("AIのデータ生成でエラーが発生しました。もう一度実行してください。")
            st.text(f"AI出力内容: {csv_data}")
