import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go
import time

# --- 1. 核心评标算法：严格适配招标文件 ---

def calculate_score(my_price, base_price, E1=1.5, E2=1.0):
    """偏差率保留9位，分值保留2位"""
    if base_price <= 0: return 0
    deviation = round((my_price - base_price) / base_price, 9)
    if my_price > base_price:
        score = 100.0 - deviation * 100 * E1
    else:
        score = 100.0 + deviation * 100 * E2
    return max(0.0, round(score, 2))

def get_base_price_by_method(method_idx, my_price, others, N):
    """按照招标文件描述计算基准价"""
    all_bids = sorted(others + [my_price])
    
    # 方法 1, 2, 3, 4 执行 n1, n2 剔除
    if method_idx <= 4:
        if N < 6:
            n1, n2 = 0, 0
        else:
            M = int(N / 5)
            n1, n2 = random.randint(1, M), random.randint(1, M)
        
        valid_bids = all_bids[n2 : N-n1] # 参与计算的其余评标价
        
        if method_idx == 1: # 二次平均法
            avg1 = np.mean(valid_bids)
            second_list = [x for x in valid_bids if x <= avg1]
            p = np.mean(second_list) if second_list else avg1
            
        elif method_idx == 2: # 随机平均法
            p = np.mean(random.sample(valid_bids, 3)) if len(valid_bids) >= 3 else np.mean(valid_bids)
            
        elif method_idx == 3: # 随机权重法
            A = all_bids[N-n1-1] # 去掉n1个最高后的最大值
            B = np.mean(valid_bids)
            X, Y = random.choice([0,1,2,3,4]), random.choice(range(10))
            K = (X + Y/10) / 10
            p = A * K + B * (1 - K)
            
        elif method_idx == 4: # 随机步距法
            if N < 8: return np.mean(valid_bids)
            a = random.choice([3, 4, 5])
            Z = random.choice([0.96, 0.97, 0.98, 0.99])
            sampled = [valid_bids[i] for i in range(0, len(valid_bids), a)]
            p = np.mean(sampled) * Z
            
    else: # 方法 5：随机低价法 (独立逻辑)
        if N < 8: return np.mean(all_bids)
        n3, n4 = int(N * 0.2), int(N * 0.7)
        m = random.randint(n3, n4)
        p = all_bids[m] # 去掉m个最低后的最低价
        
    return round(p, 2)

# --- 2. 界面与交互 ---

st.title("🛡️ 投标决策全概率全量演化系统")

with st.sidebar:
    st.header("⚙️ 参数配置")
    limit_p = st.number_input("招标最高限价", value=10000.0)
    n_count = st.number_input("单位数量(N)", value=30)
    init_low = st.number_input("成本价", value=8200.0)
    init_high = st.number_input("初始理性高限", value=9200.0)
    
    st.divider()
    st.header("🧪 模拟控制")
    step_val = st.selectbox("扫描步长", [10.0, 5.0, 2.0, 1.0, 0.5], index=2)
    evolve_rounds = st.slider("演化轮数", 1, 10, 5)
    sample_size = st.slider("全概率采样数", 50, 500, 150)
    risk_lambda = st.slider("风险权重系数", 0.0, 2.0, 0.8)
    
    run_btn = st.button("🚀 启动全概率深度模拟")

if run_btn:
    curr_l, curr_h = init_low, init_high
    history = []
    total_sim_count = 0
    start_time = time.time()
    progress_bar = st.progress(0)
    
    # 演化主循环
    for r in range(evolve_rounds):
        # 根据当前区间生成步长阵列
        prices = np.arange(curr_l, curr_h + 0.01, step_val)
        round_data = []
        
        for p_test in prices:
            scores = []
            # 全概率采样：模拟 150 次开标，办法和参数全随机
            for _ in range(sample_size):
                m_idx = random.choice([1, 2, 3, 4, 5])
                # 竞争对手报价在当前博弈区间内随机分布
                others = [random.uniform(curr_l, curr_h) for _ in range(n_count-1)]
                bp = get_base_price_by_method(m_idx, p_test, others, n_count)
                scores.append(calculate_score(p_test, bp))
                total_sim_count += 1
            
            avg_s, std_s = np.mean(scores), np.std(scores)
            round_data.append({"p": p_test, "avg": avg_s, "std": std_s, "fit": avg_s - risk_lambda * std_s})
        
        # 提取本轮双策略点
        agg = max(round_data, key=lambda x: x['avg']) # 纯分值最高
        rob = max(round_data, key=lambda x: x['fit']) # 稳健性最高
        
        history.append({
            "轮次": r+1,
            "激进报价方案": agg['p'], "激进下浮率": f"{(1-agg['p']/limit_p)*100:.3f}%", "激进风险": round(agg['std'], 3), "激进均分": round(agg['avg'], 2),
            "稳健报价方案": rob['p'], "稳健下浮率": f"{(1-rob['p']/limit_p)*100:.3f}%", "稳健风险": round(rob['std'], 3), "稳健均分": round(rob['avg'], 2),
            "L": curr_l, "H": curr_h
        })
        
        # 边界收敛逻辑：以稳健报价方案为中心收缩搜索区间
        margin = (curr_h - curr_l) * 0.35
        curr_l = max(init_low, rob['p'] - margin)
        curr_h = min(init_high, rob['p'] + margin)
        progress_bar.progress((r+1)/evolve_rounds)

    df = pd.DataFrame(history)
    
    # --- 3. 结果标示 ---
    st.info(f"📊 统计：全概率模拟总计计算 **{total_sim_count:,}** 次 | 耗时 **{time.time()-start_time:.2f}** 秒")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🚩 激进方案（期望得分最高）")
        st.metric("方案数值", f"{df.iloc[-1]['激进报价方案']:.2f}", f"下浮 {df.iloc[-1]['激进下浮率']}")
        st.write(f"全概率预期得分：{df.iloc[-1]['激进均分']}")
        st.write(f"风险波动：{df.iloc[-1]['激进风险']}")

    with c2:
        st.subheader("🛡️ 稳健方案（系统综合推荐）")
        st.metric("方案数值", f"{df.iloc[-1]['稳健报价方案']:.2f}", f"下浮 {df.iloc[-1]['稳健下浮率']}", delta_color="inverse")
        st.write(f"全概率预期得分：{df.iloc[-1]['稳健均分']}")
        st.write(f"受控风险：{df.iloc[-1]['稳健风险']}")

    # --- 4. 可视化图表 ---
    st.divider()
    st.subheader("📈 策略演化收敛轨迹")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['轮次'], y=df['激进报价方案'], name="激进轨迹", line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df['轮次'], y=df['稳健报价方案'], name="稳健轨迹", line=dict(color='green', width=4)))
    fig.add_trace(go.Scatter(x=df['轮次'], y=df['H'], line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df['轮次'], y=df['L'], line=dict(width=0), fill='tonexty', fillcolor='rgba(0,255,0,0.05)', name='搜索范围'))
    fig.update_layout(xaxis_title="演化轮次", yaxis_title="报价数值", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 详细演化数据表")
    st.table(df.drop(columns=['L', 'H']))