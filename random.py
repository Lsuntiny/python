import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go

# 页面配置
st.set_page_config(layout="wide", page_title="五法博弈动态演化决策系统")

# --- 辅助工具：获取方法名称 ---
def get_method_name(idx):
    names = {1: "二次平均法", 2: "随机平均法", 3: "随机权重法", 4: "随机步距法", 5: "随机低价法"}
    return names.get(idx, "未知方法")

# --- 核心算法函数 ---
def calculate_score(my_price, base_price, E1=1.5, E2=1.0):
    if base_price == 0: return 0
    deviation = (my_price - base_price) / base_price
    score = 100 - deviation * 100 * E1 if my_price > base_price else 100 + deviation * 100 * E2
    return max(0.0, round(score, 4))

def get_base_price_detailed(method_idx, all_bids, N):
    if not all_bids: return 0, {}
    
    all_bids = sorted(all_bids)
    M = int(N / 5)
    n1 = random.randint(1, M) if N >= 6 else 0
    n2 = random.randint(1, M) if N >= 6 else 0
    valid_bids = all_bids[n2 : N-n1] if (n1+n2) < N else all_bids
    if not valid_bids: valid_bids = all_bids # 防止剔除后为空
    
    details = {"n1(高剔)": n1, "n2(低剔)": n2}

    if method_idx == 1:
        avg1 = np.mean(valid_bids)
        second_list = [x for x in valid_bids if x <= avg1]
        p = np.mean(second_list) if second_list else avg1
    elif method_idx == 2:
        p = np.mean(random.sample(valid_bids, min(3, len(valid_bids))))
    elif method_idx == 3:
        X, Y = random.randint(0, 4), random.randint(0, 9)
        K = (X + Y/10) / 10
        # 修正：确保索引在范围内
        idx_A = min(max(0, N-n1-1), len(all_bids)-1)
        p = all_bids[idx_A] * K + np.mean(valid_bids) * (1 - K)
        details.update({"K": K})
    elif method_idx == 4:
        a, Z = random.choice([3, 4, 5]), random.choice([0.96, 0.97, 0.98, 0.99])
        p = np.mean(valid_bids[::a]) * Z
        details.update({"步距a": a, "系数Z": Z})
    elif method_idx == 5:
        m = random.randint(int(N*0.2), int(N*0.7))
        idx_m = min(max(0, m), len(all_bids)-1)
        p = all_bids[idx_m]
        details.update({"抽取排名m": m})
    else:
        p = np.mean(all_bids)
    
    return round(p, 2), details

# --- 侧边栏：参数输入 ---
with st.sidebar:
    st.header("🛠 基础参数配置")
    limit_p = st.number_input("1. 招标最高限价", value=10000.0)
    n_count = st.number_input("2. 投标单位数量(N)", value=30)
    init_high = st.number_input("3. 投标理性高限价", value=9000.0)
    init_cost = st.number_input("4. 成本价", value=8000.0)
    scan_step = st.number_input("5. 扫描步长", value=5.0)
    evolve_rounds = st.slider("6. 演化轮数 (外层循环)", 1, 15, 5)
    st.divider()
    run_calc = st.button("🚀 启动全量演化模拟")

# --- 主界面布局 ---
st.title("🏗️ 投标决策演化专家系统")

# 第一部分：单次模拟展示
st.header("第一部分：现场开标随机计算模拟 (单次)")
if st.button("执行单次随机开标演示"):
    sample_bids = [random.uniform(init_cost, init_high) for _ in range(n_count)]
    single_res = []
    for m in range(1, 6):
        if m in [4, 5] and n_count < 8: continue
        bp, info = get_base_price_detailed(m, sample_bids, n_count)
        single_res.append({"办法名称": get_method_name(m), "模拟基准价": bp, **info})
    st.table(pd.DataFrame(single_res).fillna("-"))

st.divider()

# 第二部分：动态演化展示
st.header("第二部分：五法动态演化收敛分析")

if run_calc:
    curr_low, curr_high = init_cost, init_high
    all_history = []
    boundary_history = []
    progress_bar = st.progress(0)
    
    for r in range(evolve_rounds):
        round_prices = []
        # 记录本轮边界
        boundary_history.append({"轮次": r+1, "下限(成本)": curr_low, "上限(理性)": curr_high})
        
        # 针对五种办法分别扫描
        for m_idx in range(1, 6):
            if m_idx in [4, 5] and n_count < 8: continue
            
            prices = np.arange(curr_low, curr_high + 0.1, scan_step)
            best_p_m, max_s_m = curr_low, -1.0
            
            for p_test in prices:
                # 蒙特卡洛采样 (20次)
                m_scores = []
                for _ in range(20):
                    others = [random.uniform(curr_low, curr_high) for _ in range(n_count-1)]
                    bp, _ = get_base_price_detailed(m_idx, others + [p_test], n_count)
                    m_scores.append(calculate_score(p_test, bp))
                
                avg_s = np.mean(m_scores)
                if avg_s > max_s_m:
                    max_s_m, best_p_m = avg_s, p_test
            
            round_prices.append(best_p_m)
            all_history.append({
                "轮次": r+1, 
                "办法名称": get_method_name(m_idx), 
                "最优报价": best_p_m, 
                "预期分": round(max_s_m, 2)
            })
        
        # --- 演化逻辑：Min-Max 迭代 ---
        curr_low = min(round_prices)
        curr_high = max(round_prices)
        progress_bar.progress((r + 1) / evolve_rounds)

    # 结果展示展示
    df_history = pd.DataFrame(all_history)
    df_bounds = pd.DataFrame(boundary_history)

    col_res, col_chart = st.columns([1, 2])
    
    with col_res:
        st.subheader("📋 最终演化建议")
        final_round = df_history[df_history['轮次'] == evolve_rounds]
        final_round['下浮率'] = final_round['最优报价'].apply(lambda x: f"{round((1-x/limit_p)*100, 2)}%")
        st.dataframe(final_round[['办法名称', '最优报价', '下浮率', '预期分']], use_container_width=True)
        
        st.subheader("📉 边界演化记录")
        st.dataframe(df_bounds, use_container_width=True)

    with col_chart:
        st.subheader("📈 五法收敛轨迹图")
        fig = go.Figure()
        for m in df_history['办法名称'].unique():
            m_df = df_history[df_history['办法名称'] == m]
            fig.add_trace(go.Scatter(x=m_df['轮次'], y=m_df['报价' if '报价' in m_df else '最优报价'], name=m, mode='lines+markers'))
        
        # 阴影填充
        fig.add_trace(go.Scatter(x=df_bounds['轮次'], y=df_bounds['上限(理性)'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df_bounds['轮次'], y=df_bounds['下限(成本)'], line=dict(width=0), fill='tonexty', 
                                 fillcolor='rgba(0, 100, 255, 0.1)', name='动态演化区间'))
        
        fig.update_layout(xaxis_title="演化轮次", yaxis_title="金额", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 请在左侧配置参数并启动模拟。")