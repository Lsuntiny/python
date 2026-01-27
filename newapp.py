import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="全概率抗风险决策系统")

# --- 核心算法：增加了稳定性计算 ---
def calculate_score(my_price, base_price, E1=1.5, E2=1.0):
    if base_price == 0: return 0
    deviation = (my_price - base_price) / base_price
    score = 100 - deviation * 100 * E1 if my_price > base_price else 100 + deviation * 100 * E2
    return max(0.0, round(score, 4))

def get_base_price_by_method(method_idx, all_bids, N):
    all_bids = sorted(all_bids)
    M = int(N / 5)
    n1, n2 = random.randint(1, M) if N >= 6 else 0, random.randint(1, M) if N >= 6 else 0
    valid_bids = all_bids[n2 : N-n1] if (n1+n2) < N else all_bids
    
    if method_idx == 1: # 二次平均
        avg1 = np.mean(valid_bids)
        second_list = [x for x in valid_bids if x <= avg1]
        p = np.mean(second_list) if second_list else avg1
    elif method_idx == 2: # 随机平均
        p = np.mean(random.sample(valid_bids, min(3, len(valid_bids))))
    elif method_idx == 3: # 随机权重
        K = (random.randint(0, 4) + random.randint(0, 9)/10) / 10
        p = all_bids[min(max(0, N-n1-1), len(all_bids)-1)] * K + np.mean(valid_bids) * (1 - K)
    elif method_idx == 4: # 随机步距
        a, Z = random.choice([3, 4, 5]), random.choice([0.96, 0.97, 0.98, 0.99])
        p = np.mean(valid_bids[::a]) * Z
    elif method_idx == 5: # 随机低价
        m = random.randint(int(N*0.2), int(N*0.7))
        p = all_bids[min(max(0, m), len(all_bids)-1)]
    return round(p, 2)

# --- 主界面 ---
st.title("🛡️ 全概率抗风险投标决策专家 (Pro Max)")
st.sidebar.header("⚙️ 核心演化参数")

# 输入参数
limit_p = st.sidebar.number_input("最高限价", value=10000.0)
n_count = st.sidebar.number_input("单位数量(N)", value=30)
c_low = st.sidebar.number_input("成本价", value=8000.0)
c_high = st.sidebar.number_input("理性高限价", value=9000.0)
evolve_rounds = st.sidebar.slider("演化轮数", 1, 15, 5)
risk_lambda = st.sidebar.slider("风险厌恶系数 (越大越求稳)", 0.0, 2.0, 0.5)

if st.sidebar.button("🚀 启动深度演化模拟"):
    curr_l, curr_h = c_low, c_high
    evolution_history = []
    
    # 模拟进度
    bar = st.progress(0)
    
    for r in range(evolve_rounds):
        # 扫描步长随轮次变细
        step = (curr_h - curr_l) / 20
        prices = np.arange(curr_l, curr_h + 0.1, max(0.1, step))
        
        round_results = []
        for p_test in prices:
            scores = []
            # 增加样本量到 100 次以获得更稳定的标准差
            for _ in range(100):
                m_idx = random.choice([1, 2, 3, 4, 5])
                others = [random.uniform(curr_l, curr_h) for _ in range(n_count-1)]
                bp = get_base_price_by_method(m_idx, others + [p_test], n_count)
                scores.append(calculate_score(p_test, bp))
            
            avg_s = np.mean(scores)
            std_s = np.std(scores)
            # 适应度 = 期望得分 - 风险系数 * 标准差
            fitness = avg_s - (risk_lambda * std_s)
            round_results.append({
                "price": p_test, 
                "avg": avg_s, 
                "std": std_s, 
                "fitness": fitness
            })

        # 找到本轮“最稳且最高分”的点
        best_entry = max(round_results, key=lambda x: x['fitness'])
        best_p = best_entry['price']
        
        evolution_history.append({
            "轮次": r+1,
            "建议报价": best_p,
            "预期均分": round(best_entry['avg'], 2),
            "风险波动": round(best_entry['std'], 2),
            "搜索下限": curr_l,
            "搜索上限": curr_h
        })

        # 演化收缩边界
        margin = (curr_h - curr_l) * 0.25
        curr_l = max(c_low, best_p - margin)
        curr_h = min(c_high, best_p + margin)
        bar.progress((r+1)/evolve_rounds)

    # --- 深度展示 ---
    res_df = pd.DataFrame(evolution_history)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 策略演化明细")
        st.dataframe(res_df.style.highlight_max(subset=['预期均分'], color='lightgreen'))
        
        st.metric("最终黄金报价", f"{best_p}", f"下浮 {round((1-best_p/limit_p)*100, 2)}%")
        st.info(f"风险提示：最终风险波动为 {res_df.iloc[-1]['风险波动']}，数值越小报价越稳。")

    with col2:
        st.subheader("📈 报价与风险边界演化")
        fig = go.Figure()
        # 报价线
        fig.add_trace(go.Scatter(x=res_df['轮次'], y=res_df['建议报价'], name="黄金报价点", mode='lines+markers', line=dict(color='#FF4B4B', width=3)))
        # 绘制边界
        fig.add_trace(go.Scatter(x=res_df['轮次'], y=res_df['搜索上限'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=res_df['轮次'], y=res_df['搜索下限'], line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 75, 75, 0.1)', name='动态演化区间'))
        
        fig.update_layout(xaxis_title="演化轮次", yaxis_title="报价金额")
        st.plotly_chart(fig, use_container_width=True)

    # 绘制得分敏感度图
    st.divider()
    st.subheader("🎯 最终报价敏感度分析 (最后一轮快照)")
    sens_df = pd.DataFrame(round_results)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sens_df['price'], y=sens_df['avg'], name="期望得分", line=dict(color='green')))
    fig2.add_trace(go.Scatter(x=sens_df['price'], y=sens_df['avg']-sens_df['std'], name="风险下限", line=dict(dash='dot', color='orange')))
    fig2.update_layout(xaxis_title="报价", yaxis_title="得分")
    st.plotly_chart(fig2, use_container_width=True)