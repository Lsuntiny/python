import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# 页面配置
st.set_page_config(page_title="投标博弈-精准约束版", layout="wide")

# --- 左侧侧边栏 ---
with st.sidebar:
    st.header("📋 参数设置")
    limit_price = st.number_input("最高限价 (万元)", value=10000.0, step=100.0)
    cost_price = st.number_input("成本价 (万元)", value=8300.0, step=50.0)
    c_discount = (1 - cost_price / limit_price) * 100
    st.markdown(f"📊 成本下浮率底线: **{c_discount:.2f}%**")
    
    num_competitors = st.number_input("投标单位数量 (不含自己)", value=10, min_value=1)
    step_size = st.number_input("模拟步长 (万元)", value=1.0, min_value=0.1, step=0.5)
    
    st.divider()
    st.subheader("🛡️ 废标判定规则")
    st.info("报价 < 限价90% **且** < 均值95% 时判定为 0 分")
    
    run_button = st.button("🚀 执行演算", use_container_width=True)

# --- 右侧主界面 ---
if run_button:
    start_time = time.time()
    n_values = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    sim_per_n = 2000 
    
    # 搜索范围从成本价开始，不再强制截断于限价90%
    test_bids = np.arange(cost_price, limit_price + step_size, step_size)
    
    all_scores = []
    n_detail_matrix = []

    with st.spinner('正在模拟复合约束下的最优红区...'):
        market_mu = limit_price * 0.92  
        market_sigma = limit_price * 0.015 
        
        static_threshold = limit_price * 0.90
        
        for my_bid in test_bids:
            bid_n_scores = []
            for n in n_values:
                others = np.random.normal(market_mu, market_sigma, (sim_per_n, num_competitors))
                p_values = (np.sum(others, axis=1) + my_bid) / (num_competitors + 1)
                
                # --- 核心逻辑修正：双重条件同时满足才废标 ---
                dynamic_threshold = p_values * 0.95
                # 条件1：低于限价90%； 条件2：低于均值95%
                is_invalid = (my_bid < static_threshold) & (my_bid < dynamic_threshold)
                
                d_values = p_values * (100 - n) / 100
                deviations = np.round((my_bid - d_values) / d_values * 100, 2)
                
                # 评分逻辑
                scores = np.where(deviations >= 0, 100 - deviations * 5, 100 - np.abs(deviations) * 3)
                # 判定废标
                final_scores = np.where(is_invalid, 0, np.maximum(scores, 0))
                bid_n_scores.append(np.mean(final_scores))
            
            all_scores.append(np.mean(bid_n_scores))
            n_detail_matrix.append(bid_n_scores)

    duration = time.time() - start_time
    all_scores = np.array(all_scores)
    best_idx = np.argmax(all_scores)
    best_bid = test_bids[best_idx]
    best_discount = (1 - best_bid / limit_price) * 100
    
    # 锁定红色最优段
    threshold = all_scores[best_idx] * 0.999
    range_idx = np.where(all_scores >= threshold)[0]
    min_range_bid, max_range_bid = test_bids[range_idx[0]], test_bids[range_idx[-1]]

    # --- 结果呈现 ---
    st.subheader("🎯 “与逻辑”约束决策分析")
    c1, c2, c3 = st.columns(3)
    c1.metric("建议最优点", f"{best_bid:,.2f} 万", f"下浮 {best_discount:.2f}%")
    c2.metric("红色最优段 (金额)", f"{min_range_bid:,.0f} - {max_range_bid:,.0f} 万")
    c3.metric("红色最优段 (下浮率)", f"{(1-max_range_bid/limit_price)*100:.2f}% - {(1-min_range_bid/limit_price)*100:.2f}%")

    st.divider()

    # 指标行
    s1, s2, s3 = st.columns(3)
    s1.metric("模拟次数", f"{len(test_bids)*9*sim_per_n:,} 次")
    s2.metric("演算耗时", f"{duration:.2f} 秒")
    s3.metric("期望得分", f"{all_scores[best_idx]:.2f} 分")

    st.divider()

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_bids, y=all_scores, mode='lines', line=dict(width=3, color='#333')))
        fig.add_vrect(x0=min_range_bid, x1=max_range_bid, fillcolor="red", opacity=0.3, line_width=0, annotation_text="最优红区")
        fig.add_vline(x=best_bid, line_dash="dash", line_color="red")
        fig.update_layout(title="复合约束博弈曲线", xaxis_title="报价 (万元)", yaxis_title="加权得分", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.write("**各 N 值得分明细**")
        detail_df = pd.DataFrame({"N 值": [f"{n}%" for n in n_values], "得分": [f"{x:.2f}" for x in n_detail_matrix[best_idx]]})
        st.table(detail_df)