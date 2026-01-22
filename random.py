import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go

# 页面配置
st.set_page_config(layout="wide", page_title="投标决策专家系统")

# --- 核心逻辑函数 ---
def calculate_score(my_price, base_price, E1=1.5, E2=1.0):
    deviation_rate = (my_price - base_price) / base_price
    score = 100 - deviation_rate * 100 * E1 if my_price > base_price else 100 + deviation_rate * 100 * E2
    return max(0.0, round(score, 4))

def get_base_price_detailed(method_idx, all_bids, N):
    """返回基准价及该次随机抽取的详细参数"""
    all_bids = sorted(all_bids)
    M = int(N / 5)
    n1 = random.randint(1, M) if N >= 6 else 0
    n2 = random.randint(1, M) if N >= 6 else 0
    valid_bids = all_bids[n2 : N-n1] if (n1+n2) < N else all_bids
    
    details = {"n1(最高剔除)": n1, "n2(最低剔除)": n2}
    p = 0
    
    if method_idx == 1:
        avg1 = np.mean(valid_bids)
        second_list = [x for x in valid_bids if x <= avg1]
        p = np.mean(second_list) if second_list else avg1
        details["过程"] = "二次平均"
    elif method_idx == 2:
        samples = random.sample(valid_bids, min(3, len(valid_bids)))
        p = np.mean(samples)
        details["随机抽取价"] = str([round(x, 2) for x in samples])
    elif method_idx == 3:
        X, Y = random.randint(0, 4), random.randint(0, 9)
        K = (X + Y/10) / 10
        A, B = all_bids[N-n1-1], np.mean(valid_bids)
        p = A * K + B * (1 - K)
        details.update({"X": X, "Y": Y, "权重K": K})
    elif method_idx == 4:
        a, Z = random.choice([3, 4, 5]), random.choice([0.96, 0.97, 0.98, 0.99])
        p = np.mean(valid_bids[::a]) * Z
        details.update({"步距a": a, "系数Z": Z})
    elif method_idx == 5:
        m = random.randint(int(N*0.2), int(N*0.7))
        p = all_bids[m]
        details["抽取排名m"] = m
        
    return round(p, 2), details

# --- 侧边栏参数 ---
st.sidebar.header("⚙️ 招标基础参数")
limit_p = st.sidebar.number_input("招标最高限价", value=10000.0)
n_count = st.sidebar.number_input("投标单位数量(N)", value=30)
rat_high = st.sidebar.number_input("理性报价上限", value=9000.0)
cost_p = st.sidebar.number_input("成本价下限", value=8000.0)
user_step = st.sidebar.number_input("初始扫描步长", value=10.0)
iter_count = st.sidebar.slider("收敛迭代次数", 5, 100, 50)

# --- 主界面 ---
st.title("🎯 投标报价模拟与动态决策系统")

# 部分一：现场开标随机模拟
st.header("第一部分：现场随机参数模拟")
st.write("点击按钮，模拟一次真实的开标现场随机抽取过程（以你的理性高限价作为我的报价）。")

if st.button("模拟一次随机开标"):
    sim_others = [random.uniform(cost_p, rat_high) for _ in range(n_count-1)]
    sim_data = []
    for m in range(1, 6):
        if m in [4, 5] and n_count < 8: continue
        bp, dt = get_base_price_detailed(m, sim_others + [rat_high], n_count)
        score = calculate_score(rat_high, bp)
        sim_data.append({"办法": f"方法 {m}", "计算基准价": bp, "我的得分": score, **dt})
    st.table(pd.DataFrame(sim_data).fillna("-"))

st.divider()

# 部分二：报价动态收敛展示
st.header("第二部分：报价动态收敛优化")
st.write("系统将通过步长扫描和区间折半算法，在多次随机波动中寻找“期望得分最高”的最优报价。")

if st.sidebar.button("开始收敛计算"):
    results = []
    converge_plots = {}
    
    # 创建进度条
    progress_bar = st.progress(0)
    
    # 针对每种办法进行独立收敛
    for m_idx in range(1, 6):
        if m_idx in [4, 5] and n_count < 8: continue
        
        low, high = cost_p, rat_high
        step = user_step
        history = []
        
        for i in range(iter_count):
            # 1. 步长扫描
            scan_prices = np.arange(low, high + 0.1, max(0.1, step))
            best_p_iter, max_s = low, -1.0
            
            for p_test in scan_prices:
                # 蒙特卡洛抽样：每个报价点模拟20次开标求平均分
                m_scores = []
                for _ in range(20):
                    others = [random.uniform(cost_p, rat_high) for _ in range(n_count-1)]
                    bp, _ = get_base_price_detailed(m_idx, others + [p_test], n_count)
                    m_scores.append(calculate_score(p_test, bp))
                
                avg_s = np.mean(m_scores)
                if avg_s > max_s:
                    max_s, best_p_iter = avg_s, p_test
            
            history.append(best_p_iter)
            
            # 2. 动态收缩区间 (展示过程的关键)
            range_w = (high - low) * 0.4
            low = max(cost_p, best_p_iter - range_w/2)
            high = min(rat_high, best_p_iter + range_w/2)
            step = (high - low) / 10 # 步长随区间变细
            
            if (high - low) < 0.05: break
            
        converge_plots[f"方法 {m_idx}"] = history
        results.append({
            "评标办法": f"方法 {m_idx}",
            "最优报价建议": round(best_p_iter, 2),
            "建议下浮率": f"{round((1 - best_p_iter/limit_p)*100, 2)}%",
            "预期最高得分": round(max_s, 2)
        })
        progress_bar.progress(m_idx * 20)

    # 展示收敛结果
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("最终决策表")
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    
    with c2:
        st.subheader("收敛过程可视化")
        fig = go.Figure()
        for m_name, h in converge_plots.items():
            fig.add_trace(go.Scatter(y=h, mode='lines+markers', name=m_name))
        fig.update_layout(xaxis_title="迭代步数", yaxis_title="搜索报价", height=450, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 请点击左侧边栏底部的 '开始收敛计算' 按钮启动全量模拟。")