import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# 页面配置
st.set_page_config(page_title="投标博弈-实战版", layout="wide")

# --- 左侧侧边栏：输入设置 ---
with st.sidebar:
    st.header("📋 参数设置")
    limit_price = st.number_input("最高限价 (万元)", value=10000.0, step=100.0)
    
    # 恢复成本价下浮率实时换算
    cost_price = st.number_input("成本价 (万元)", value=8300.0, step=50.0)
    cost_discount_percent = (1 - cost_price / limit_price) * 100
    st.markdown(f"📊 成本下浮率底线: **{cost_discount_percent:.2f}%**")
    
    st.markdown("---")
    st.subheader("🛡️ 约束条件设定")
    st.caption("1. 报价 ≥ 限价的 90%")
    st.caption("2. 报价 ≥ P值的 95%")
    
    num_competitors = st.number_input("投标单位数量 (不含自己)", value=10, min_value=1)
    step_size = st.number_input("模拟步长 (万元)", value=1.0, min_value=0.1, step=0.5)
    
    st.divider()
    run_button = st.button("🚀 执行全概率演算", use_container_width=True)

# --- 右侧主界面 ---
if run_button:
    start_time = time.time()
    n_values = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    sim_per_n = 2000 
    
    # 逻辑门槛：报价不得低于限价90%
    abs_lower_limit = limit_price * 0.90
    # 搜索范围从成本价和限价90%的较大值开始
    test_start = max(cost_price, abs_lower_limit)
    test_bids = np.arange(test_start, limit_price + step_size, step_size)
    
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
                
                # 双重约束判定
                # 1. 静态：90%限价 (已在test_bids控制)
                # 2. 动态：95% P值
                is_valid = my_bid >= (p_values * 0.95)
                
                d_values = p_values * (100 - n) / 100
                deviations = np.round((my_bid - d_values) / d_values * 100, 2)
                
                # 评分逻辑
                scores = np.where(deviations >= 0, 100 - deviations * 5, 100 - np.abs(deviations) * 3)
                # 约束失败则判定为0分
                final_scores = np.where(is_valid, np.maximum(scores, 0), 0)
                bid_n_scores.append(np.mean(final_scores))
            
            all_scores.append(np.mean(bid_n_scores))
            n_detail_matrix.append(bid_n_scores)

    duration = time.time() - start_time
    all_scores = np.array(all_scores)
    best_idx = np.argmax(all_scores)
    best_bid = test_bids[best_idx]
    best_discount = (1 - best_bid / limit_price) * 100
    
    # 计算红色最优段 (期望得分 99.9% 覆盖区)
    threshold = all_scores[best_idx] * 0.999
    range_idx = np.where(all_scores >= threshold)[0]
    min_range_bid, max_range_bid = test_bids[range_idx[0]], test_bids[range_idx[-1]]
    min_range_disc = (1 - max_range_bid / limit_price) * 100
    max_range_disc = (1 - min_range_bid / limit_price) * 100

    # --- 数据看板 ---
    st.subheader("🔴 最优报价与红色区间分布")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("建议最优点", f"{best_bid:,.2f} 万", f"下浮 {best_discount:.2f}%")
    c2.metric("红色最优段 (金额)", f"{min_range_bid:,.0f} - {max_range_bid:,.0f} 万")
    c3.metric("红色最优段 (下浮率)", f"{min_range_disc:.2f}% - {max_range_disc:.2f}%")

    st.divider()

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_bids, y=all_scores, mode='lines', 
                                 line=dict(width=3, color='#444'), name="得分曲线"))
        
        # 红色最优段标注
        fig.add_vrect(x0=min_range_bid, x1=max_range_bid, 
                      fillcolor="red", opacity=0.3, line_width=0,
                      annotation_text="最优段", annotation_position="top left")
        
        fig.add_vline(x=best_bid, line_dash="dash", line_color="red")

        fig.update_layout(title="双重约束下的博弈曲线", xaxis_title="报价 (万元)", 
                          yaxis_title="加权得分", template="plotly_white", height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.write("**实时统计**")
        st.write(f"模拟样本: {len(test_bids)*9*sim_per_n:,} 次")
        st.write(f"演算耗时: {duration:.2f} 秒")
        
        st.write("**最优价下 N 值得分明细**")
        detail_df = pd.DataFrame({
            "现场 N 值": [f"{n}%" for n in n_values],
            "预测得分": [f"{x:.2f}" for x in n_detail_matrix[best_idx]]
        })
        st.table(detail_df)

else:
    st.info("👈 请设置参数。成本下浮率将根据您的输入实时更新。")