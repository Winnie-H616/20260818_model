from __future__ import annotations
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from common import load_data, order_facts

st.set_page_config(page_title="Random Forest 迴歸入門", page_icon="🌳", layout="wide")

NUMERIC_FEATURES = ["order_month", "is_weekend", "customer_days"]
CATEGORICAL_FEATURES = [
    "payment_type",
    "segment",
    "city",
    "acquisition_channel",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "order_revenue"

@st.cache_data
def prepare_data() -> pd.DataFrame:
    """把 common.py 的訂單金額資料和客戶資料合併成模型資料。"""
    data = load_data()
    df = order_facts(data).rename(columns={"line_revenue": TARGET})
    customers = data["customers"].copy()
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    df = df.merge(customers, on="customer_id", how="left", validate="many_to_one")
    df["order_month"] = df["order_date"].dt.month
    df["is_weekend"] = (df["order_date"].dt.dayofweek >= 5).astype(int)
    df["customer_days"] = (df["order_date"] - df["signup_date"]).dt.days.clip(lower=0)
    return df[FEATURES + [TARGET]].dropna().reset_index(drop=True)

@st.cache_resource
def train_model(df: pd.DataFrame, trees: int, max_depth: int, min_leaf: int):
    """切分資料、建立 Pipeline、訓練模型並回傳評估結果。"""
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=42
    )
    preprocess = ColumnTransformer(
        [
            ("number", "passthrough", NUMERIC_FEATURES),
            ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=trees,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    result = pd.DataFrame({"實際金額": y_test, "預測金額": prediction}).reset_index(drop=True)
    return model, result, mean_absolute_error(y_test, prediction), r2_score(y_test, prediction)

df = prepare_data()

st.title("🌳 Random Forest 迴歸系統")
st.caption("目標：根據客戶與訂單資訊，預測完成訂單的金額。")

with st.sidebar:
    st.header("模型設定")
    trees = st.slider("樹的數量 n_estimators", 50, 300, 100, 50)
    max_depth = st.slider("每棵樹最大深度 max_depth", 2, 20, 8)
    min_leaf = st.slider("葉節點最少樣本 min_samples_leaf", 1, 20, 5)
    st.info("課堂上只調這三個參數，觀察 MAE 與 R² 如何變化。")

model, result, mae, r2 = train_model(df, trees, max_depth, min_leaf)

st.subheader("1. 資料")
st.write(f"共 {len(df):,} 筆完成訂單；80% 訓練、20% 測試。")
with st.expander("查看模型資料與欄位"):
    st.dataframe(df.head(20), width="stretch", hide_index=True)
    st.markdown(
        "**X（特徵）**：月份、是否週末、成為客戶天數、付款方式、客群、城市、獲客通路  "
        "\n**y（答案）**：訂單金額"
    )

st.subheader("2. 訓練與評估")
m1, m2, m3 = st.columns(3)
m1.metric("測試資料筆數", f"{len(result):,}")
m2.metric("MAE 平均絕對誤差", f"NT$ {mae:,.0f}")
m3.metric("R² 決定係數", f"{r2:.3f}")
st.caption("MAE 越小越好；R² 越接近 1 越好，若小於 0 代表模型比直接猜平均值還差。")

chart_data = result.sample(min(100, len(result)), random_state=42).sort_values("實際金額")
st.line_chart(chart_data, y=["實際金額", "預測金額"])

st.subheader("3. 輸入一筆新資料")
left, right = st.columns(2)
with left:
    order_month = st.slider("訂單月份", 1, 12, 8)
    is_weekend = st.checkbox("週末下單")
    customer_days = st.number_input("成為客戶天數", 0, 3000, 180, step=30)
    payment_type = st.selectbox("付款方式", sorted(df["payment_type"].unique()))
with right:
    segment = st.selectbox("客群", sorted(df["segment"].unique()))
    city = st.selectbox("城市", sorted(df["city"].unique()))
    acquisition_channel = st.selectbox("獲客通路", sorted(df["acquisition_channel"].unique()))

new_order = pd.DataFrame(
    [
        {
            "order_month": order_month,
            "is_weekend": int(is_weekend),
            "customer_days": customer_days,
            "payment_type": payment_type,
            "segment": segment,
            "city": city,
            "acquisition_channel": acquisition_channel,
        }
    ]
)

if st.button("預測訂單金額", type="primary"):
    predicted_revenue = model.predict(new_order)[0]
    st.success(f"預測訂單金額：NT$ {predicted_revenue:,.0f}")