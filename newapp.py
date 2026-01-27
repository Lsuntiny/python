import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
import time

# --- 核心算法 (保持高精度逻辑) ---
def calculate_score(my_price, base_price, E1, E2):
    if base_price <= 0: return 0
    deviation = round((my_price - base_price) / base_price, 9)
    score = 100.0 - deviation * 100 * E1 if my_price > base_price else 100.0 + deviation * 100 * E2
    return max(0.0, round(score, 2))

def get_base_price_by_method(method_idx, my_price, others, N):
    all_bids = sorted(others + [my_price])
    if method_idx <= 4:
        if N < 6: n1, n2 = 0, 0
        else:
            M = int(N / 5)
            n1, n2 = random.randint(1, M), random.randint(1, M)
        valid_bids = all_bids[n2 : N-n1]
        if method_idx == 1: # 二次平均
            avg1 = np.mean(valid_bids)
            second_list = [x for x in valid_bids if x <= avg1]
            p = np.mean(second_list) if second_list else avg1
        elif method_idx == 2: # 随机平均
            p = np.mean(random.sample(valid_bids, 3)) if len(valid_bids) >= 3 else np.mean(valid_bids)
        elif method_idx == 3: # 随机权重
            A, B = all_bids[N-n1-1], np.mean(valid_bids)
            X, Y = random.choice([0,1,2,3,4]), random.choice(range(10))
            K = (X + Y/10) / 10
            p = A * K + B * (1 - K)
        elif method_idx == 4: # 随机步距
            if N < 8: return np.mean(valid_bids)
            a, Z = random.choice([3, 4, 5]), random.choice([0.96, 0.97, 0.98, 0.99])
            sampled = [valid_bids[i] for i in range(0, len(valid_bids), a)]
            p = np.mean(sampled) * Z
    else: # 方法 5
        m = random.randint(int(N * 0.2), int(N * 0.7))
        p = all_bids[m]
    return round(p, 2)

# --- 界面布局 ---
st.set_page_config(layout="wide", page_title="投标博弈系统")

# 移除冗余边距
st.markdown("""<style> .main .block-container {padding-top: 1rem; padding-bottom: 1rem;} </style>""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🛡️ 参数中心")
    col_a, col_b = st.columns(2)
    limit_p = col_a.number_input("最高限价", value=10000.0)
    n_count = col_b.number_input("单位数量(N)", value=30)
    e1_val = col_a.number_input("高标E1", value=1.5)
    e2_val = col_b.number_input("低标E2", value=1.0)
    init_low = col_a.number_input("成本价", value=8200.0)
    init_high = col_b.number_input("理性上限", value=9200.0)
    
    st.divider()
    step_val = st.number_input("扫描步长", value=1.0, step=0.1, format="%.3f")
    evolve_rounds = st.slider("演化轮数", 1, 10, 5)
    sample_size = st.slider("每点采样", 50, 500, 150)
    risk_lambda = st.slider("风险系数", 0.0, 2.0, 0.8)
    run_btn = st.button("🚀 启动演化", use_container_width=True)

if run_btn:
    curr_l, curr_h = init_low, init_high
    history = []
    total_sim = 0
    start_t = time.time()
    
    p_bar = st.progress(0)
    for r in range(evolve_rounds):
        prices = np.arange(curr_l, curr_h + (step_val/2), step_val)
        round_data = []
        for p_test in prices:
            scores = []
            for _ in range(sample_size):
                m_idx = random.choice([1, 2, 3, 4, 5])
                others = [random.uniform(curr_l, curr_h) for _ in range(n_count-1)]
                bp = get_base_price_by_method(m_idx, p_test, others, n_count)
                scores.append(calculate_score(p_test, bp, e1_val, e2_val))
                total_sim += 1
            round_data.append({"p": p_test, "avg": np.mean(scores), "std": np.std(scores)})
        
        # 定义适应度并提取结果
        for d in round_data: d['fit'] = d['avg'] - risk_lambda * d['std']
        agg = max(round_data, key=lambda x: x['avg'])
        rob = max(round_data, key=lambda x: x['fit'])
        
        history.append({
            "轮次": r+1, "激进价": agg['p'], "激进下浮": f"{(1-agg['p']/limit_p)*100:.3f}%", "激进得分": round(agg['avg'], 2), "激进风险": round(agg['std'], 3),
            "稳健价": rob['p'], "稳健下浮": f"{(1-rob['p']/limit_p)*100:.3f}%", "稳健得分": round(rob['avg'], 2), "稳健风险": round(rob['std'], 3)
        })
        # 收敛
        margin = (curr_h - curr_l) * 0.3
        curr_l, curr_h = max(init_low, rob['p']-margin), min(init_high, rob['p']+margin)
        p_bar.progress((r+1)/evolve_rounds)

    df = pd.DataFrame(history)

    # --- 结果展示区 ---
    c1, c2, c3 = st.columns([1.5, 1.5, 1])
    with c1:
        st.success("### 🚩 激进型结果 (均值导向)")
        st.metric("报价方案", f"{df.iloc[-1]['激进价']:.2f}", f"下浮 {df.iloc[-1]['激进下浮']}")
        st.write(f"**预期均分：** {df.iloc[-1]['激进得分']}")
        st.write(f"**波动风险：** {df.iloc[-1]['激进风险']} (Std)")

    with c2:
        st.info("### 🛡️ 稳健型结果 (对冲导向)")
        st.metric("报价方案", f"{df.iloc[-1]['稳健价']:.2f}", f"下浮 {df.iloc[-1]['稳健下浮']}", delta_color="inverse")
        st.write(f"**预期均分：** {df.iloc[-1]['稳健得分']}")
        st.write(f"**波动风险：** {df.iloc[-1]['稳健风险']} (Std)")
        
    with c3:
        st.write("### 📊 模拟摘要")
        st.write(f"总模拟次数: {total_sim:,}")
        st.write(f"计算耗时: {time.time()-start_t:.2f}s")
        st.write(f"收敛步长: {step_val}")

    # --- 深度说明 Tab 页 ---
    st.divider()
    tab1, tab2, tab3 = st.tabs(["💡 策略风险说明", "📈 演化轨迹图", "📑 详细数据日志"])
    
    with tab1:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
            **1. 激进型（波动性解析）：**
            * **核心逻辑**：追求五种办法随机下总分的最大化。
            * **风险说明**：其风险值（Std）通常偏高，意味着该报价对现场抽签结果**极度敏感**。如果抽中方法三可能得满分，但抽中方法五可能出现较大偏差。
            * **波动表现**：在多次模拟中分数值离散度大，适合“非赢即输”的激进投标环境。
            """)
        with col_r:
            st.markdown("""
            **2. 稳健型（风险对冲解析）：**
            * **核心逻辑**：在确保高分的同时，极力规避可能出现的“极端低分”。
            * **风险说明**：通过风险系数（λ）对波动率进行了惩罚。其报价点通常位于**概率分布的最稠密区域**。
            * **波动表现**：无论现场抽中哪种办法，得分表现都趋于稳定，能有效对抗方法五（随机低价法）带来的不确定性。
            """)
        

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['轮次'], y=df['激进价'], name="激进轨迹", line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=df['轮次'], y=df['稳健价'], name="稳健轨迹", line=dict(color='green', width=3)))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.dataframe(df, use_container_width=True, height=300)