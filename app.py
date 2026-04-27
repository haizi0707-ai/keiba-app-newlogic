import os
import re
import io
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai

# --- 1. 基本設定 ---
st.set_page_config(page_title="AI-Multiplier v13.0 全自動版", layout="wide")

# APIキー設定（サイドバーから入力）
api_key = st.sidebar.text_input("Gemini API Key", type="password", value=st.secrets.get("GEMINI_API_KEY", ""))
if api_key:
    genai.configure(api_key=api_key)

EVAL_MAP = {"かなり向く": 1.25, "向く": 1.10, "普通": 1.00, "やや不向き": 0.85, "不向き": 0.70}

# --- 2. AI検索・データ生成関数 ---
def fetch_full_auto_data(date_str, place_name, race_no):
    if not api_key:
        st.error("左側のサイドバーにAPIキーを入力してください。")
        return None
    
    model = genai.GenerativeModel(
        model_name='gemini-2.0-flash',
        tools=[{"google_search": {}}]
    )
    
    # 完全に自動で情報を拾わせる指示（あなたのプロンプトを統合）
    prompt = f"""
あなたは競馬予想データ作成AIです。

【指令】
1. ネット検索で、{date_str}の{place_name}競馬場の「最新の馬場状態」と「トラックバイアス（有利不利）」を調べてください。
2. その結果を反映させつつ、{race_no}Rの出走馬全頭について、以下のCSVを作成してください。

【CSVの列順】
日付,場所,芝ダ,距離,レース,レース名,馬番,馬名,前走競馬場,前走芝ダ,前走距離数値,前3角位置カテゴリ,前4角位置カテゴリ,前走場所,前走直線ロジック点,前々走直線ロジック点,展開予想評価,直線相性評価

【ルール】
・CSVテキストデータのみ出力すること（装飾不要）。
・直線ロジック点は0-100の数値。
・展開/相性評価は「かなり向く」「向く」「普通」「やや不向き」「不向き」の5段階。
・必ずネット検索の結果（馬場・TB）を評価に反映させること。
"""

    with st.spinner("AIが当日の馬場・TBを検索して分析中...（約30秒）"):
        try:
            response = model.generate_content(prompt)
            res_text = response.text
            # Markdown除去
            res_text = re.sub(r'^```(csv)?\s*', '', res_text, flags=re.MULTILINE)
            res_text = re.sub(r'^```\s*', '', res_text, flags=re.MULTILINE).strip()
            return res_text
        except Exception as e:
            st.error(f"AIとの通信に失敗しました: {e}")
            return None

# --- 3. 計算ロジック ---
def calculate_multiplier(df):
    for col in ["前走直線ロジック点", "前々走直線ロジック点"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(50)
    
    # 倍率計算
    df["本体点"] = (df["前走直線ロジック点"] * 0.30) + (df["前々走直線ロジック点"] * 0.20)
    df["展開係数"] = df["展開予想評価"].map(EVAL_MAP).fillna(1.0)
    df["相性係数"] = df["直線相性評価"].map(EVAL_MAP).fillna(1.0)
    df["最終点"] = df["本体点"] * df["展開係数"] * df["相性係数"]
    return df.sort_values("最終点", ascending=False)

# --- 4. メインUI ---
st.title("🏇 AI-Multiplier Logic (全自動版)")

c1, c2, c3 = st.columns(3)
d_val = c1.text_input("開催日 (例: 2026.4.27)", "2026.4.27")
p_val = c2.selectbox("場所", ["東京", "中山", "京都", "阪神", "中京", "新潟", "福島", "小倉", "札幌", "函館"])
r_val = c3.number_input("レース番号", 1, 12, 11)

if st.button("馬場・TBを自動取得して分析開始", type="primary"):
    csv_data = fetch_full_auto_data(d_val, p_val, r_val)
    if csv_data:
        try:
            df = pd.read_csv(io.StringIO(csv_data))
            res_df = calculate_multiplier(df)
            st.success("AIが情報を取得し、分析を完了しました！")
            
            # 結果表示
            st.subheader(f"🏆 {p_val}{r_val}R 予想ランキング")
            st.dataframe(res_df[["馬番", "馬名", "最終点", "本体点", "展開予想評価", "直線相性評価"]], use_container_width=True)
            
            # CSVダウンロード
            st.download_button("分析結果(CSV)を保存", csv_data, f"{p_val}{r_val}R_auto.csv")
        except:
            st.error("データの形式が合いませんでした。もう一度実行してください。")
            st.text_area("AIの出力内容（確認用）", csv_data)
