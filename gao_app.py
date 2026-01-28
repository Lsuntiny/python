import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# 页面配置
st.set_page_config(page_title="投标博弈分析-红区版", layout="wide")

# --- 左侧侧边栏：输入设置 ---
with st.sidebar:
    st.header("📋 参数设置")
    limit_price = st.number_input("最高限价 (万元)", value=10000.0, step=100.0)
    cost_price = st.number_input("成本价 (万元)", value=8300.0, step=50.0)
    c_discount = (1 - cost_price / limit_price) * 100
    st.caption(f"📊 成本下浮率底线: **{c_discount:.2f}%**")
    
    num_competitors = st.number_input("投标单位数量 (不含自己)", value=10, min_value=1)
    step_size = st.number_input("模拟步长 (万元)", value=1.0, min_value=0.1, step=0.5)
    
    st.divider()
    run_button = st.button("🚀 执行全概率演算", use_container_width=True)

# --- 右侧主界面 ---
if run_button:
    start_time = time.time()
    n_values = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    sim_per_n = 2000 
    
    test_bids = np.arange(cost_price, limit_price + step_size, step_size)
    total_samples = len(test_bids) * len(n_values) * sim_per_n
    
    all_scores = []
    n_detail_matrix = []

    with st.spinner('正在计算红色最优区间...'):
        market_mu = limit_price * 0.92  
        market_sigma = limit_price * 0.015 
        
        for my_bid in test_bids:
            bid_n_scores = []
            for n in n_values:
                others = np.random.normal(market_mu, market_sigma, (sim_per_n, num_competitors))
                p_values = (np.sum(others, axis=1) + my_bid) / (num_competitors + 1)
                d_values = p_values * (100 - n) / 100
                
                deviations = np.round((my_bid - d_values) / d_values * 100, 2)
                scores = np.where(deviations >= 0, 100 - deviations * 5, 100 - np.abs(deviations) * 3)
                bid_n_scores.append(np.mean(np.maximum(scores, 0)))
            
            all_scores.append(np.mean(bid_n_scores))
            n_detail_matrix.append(bid_n_scores)

    duration = time.time() - start_time
    all_scores = np.array(all_scores)
    best_idx = np.argmax(all_scores)
    best_bid = test_bids[best_idx]
    best_discount = (1 - best_bid / limit_price) * 100
    
    # --- 计算最优段（得分在峰值 99.9% 以上的范围） ---
    threshold = all_scores[best_idx] * 0.999
    range_idx = np.where(all_scores >= threshold)[0]
    min_range_bid, max_range_bid = test_bids[range_idx[0]], test_bids[range_idx[-1]]
    min_range_disc = (1 - max_range_bid / limit_price) * 100
    max_range_disc = (1 - min_range_bid / limit_price) * 100

    # --- 第一排：核心决策指标 ---
    st.subheader("🔴 最优段决策分析")
    c1, c2, c3 = st.columns(3)
    c1.metric("最优建议点", f"{best_bid:,.2f} 万", f"下浮 {best_discount:.2f}%")
    c2.metric("红色最优段 (金额)", f"{min_range_bid:,.0f} - {max_range_bid:,.0f} 万")
    c3.metric("红色最优段 (下浮率)", f"{min_range_disc:.2f}% - {max_range_disc:.2f}%")

    # --- 第二排：模拟运行指标 ---
    s1, s2, s3 = st.columns(3)
    s1.metric("计算规模", f"{total_samples:,} 次")
    s2.metric("计算耗时", f"{duration:.2f} 秒")
    s3.metric("期望得分", f"{all_scores[best_idx]:.2f} 分")

    st.divider()

    # --- 第三排：图表与数据明细 ---
    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        fig = go.Figure()
        # 基础曲线
        fig.add_trace(go.Scatter(x=test_bids, y=all_scores, mode='lines', 
                                 line=dict(width=3, color='#444'), name="期望得分"))
        
        # 标注红色最优段
        fig.add_vrect(x0=min_range_bid, x1=max_range_bid, 
                      fillcolor="red", opacity=0.3, line_width=0,
                      annotation_text="最优段 (红区)", annotation_position="top left")
        
        # 标注最优点
        fig.add_vline(x=best_bid, line_dash="dash", line_color="red", annotation_text="最优点")

        fig.update_layout(title="报价博弈曲线（红区为最优报价区间）", xaxis_title="报价 (万元)", 
                          yaxis_title="加权得分", template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.write("**最优价下 N 值得分分布**")
        detail_df = pd.DataFrame({
            "现场 N 值": [f"{n}%" for n in n_values],
            "预测得分": [f"{x:.2f}" for x in n_detail_matrix[best_idx]]
        })
        st.table(detail_df)

else:
    st.info("👈 请设置参数后点击“执行全概率演算”。红区将代表数学上的最优报价区间。")