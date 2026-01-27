import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go

# --- 页面配置 ---
st.set_page_config(layout="wide", page_title="投标博弈全概率决策系统")

# --- 核心算法函数库 ---

def calculate_score(my_price, base_price, E1=1.5, E2=1.0):
    """计算得分：高于基准价扣分多(E1)，低于基准价扣分少(E2)"""
    if base_price == 0: return 0
    deviation = (my_price - base_price) / base_price
    score = 100 - deviation * 100 * E1 if my_price > base_price else 100 + deviation * 100 * E2
    return max(0.0, round(score, 4))

def get_base_price_by_method(method_idx, all_bids, N):
    """根据随机抽取的办法索引，计算基准价"""
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

# --- Streamlit 界面 ---

st.title("🛡️ 投标全概率决策专家系统")
st.markdown("""
本系统模拟现场**随机抽取一种办法**进行评标的情况。
- **综合得分（均值）**：代表该价格在所有可能情况下的平均分数。数值**越大**越好。
- **风险值（波动）**：代表得分的不稳定性。数值**越小**越稳。
- **推荐指数**：结合了得分和风险的最终评价指标。数值**越大**代表综合决策价值越高。
""")

with st.sidebar:
    st.header("⚙️ 1. 输入初始参数")
    limit_p = st.number_input("招标最高限价", value=10000.0)
    n_count = st.number_input("投标单位家数(N)", value=30)
    init_cost = st.number_input("企业初始成本价", value=8000.0)
    init_high = st.number_input("初始理性最高价", value=9000.0)
    
    st.header("⚙️ 2. 模拟设置")
    evolve_rounds = st.slider("演化轮数 (逐步锁定范围)", 1, 15, 5)
    risk_weight = st.slider("风险厌恶权重 (越大则越避开波动点)", 0.0, 2.0, 0.5)
    scan_step = st.number_input("初始扫描步长", value=10.0)
    
    run_btn = st.button("🚀 启动深度演化模拟")

if run_btn:
    curr_low, curr_high = init_cost, init_high
    evolution_log = []
    
    progress_bar = st.progress(0)
    
    # --- 演化主循环 ---
    for r in range(evolve_rounds):
        # 建立当前区间的扫描阵列
        prices = np.arange(curr_low, curr_high + 0.1, max(0.1, scan_step))
        round_results = []
        
        for p_test in prices:
            scores = []
            # 对每个测试点进行 100 次全概率模拟（抽随机办法）
            for _ in range(100):
                m_random = random.choice([1, 2, 3, 4, 5]) # 现场随机抽签模拟
                others = [random.uniform(curr_low, curr_high) for _ in range(n_count-1)]
                bp = get_base_price_by_method(m_random, others + [p_test], n_count)
                scores.append(calculate_score(p_test, bp))
            
            avg_s = np.mean(scores)
            std_s = np.std(scores)
            # 计算推荐指数（适应度）：得分越高、波动越小，指数越高
            fitness = avg_s - (risk_weight * std_s)
            
            round_results.append({"price": p_test, "avg": avg_s, "std": std_s, "fitness": fitness})
        
        # 寻找本轮推荐指数最高的价格点
        best_entry = max(round_results, key=lambda x: x['fitness'])
        best_p = best_entry['price']
        
        # 记录本轮数据
        evolution_log.append({
            "轮次": r + 1,
            "推荐报价": round(best_p, 2),
            "综合得分(均值)": round(best_entry['avg'], 2),
            "风险值(波动)": round(best_entry['std'], 2),
            "推荐指数(综合)": round(best_entry['fitness'], 2),
            "下限": round(curr_low, 2),
            "上限": round(curr_high, 2)
        })
        
        # --- 演化边界更新逻辑 ---
        # 下一轮的范围基于本轮最佳报价进行合理缩减
        width = (curr_high - curr_low) * 0.3
        curr_low = max(init_cost, best_p - width/2)
        curr_high = min(init_high, best_p + width/2)
        # 步长随轮次微调，越往后越精细
        scan_step = max(0.1, scan_step * 0.7)
        
        progress_bar.progress((r + 1) / evolve_rounds)

    # --- 结果展示与分析 ---
    df = pd.DataFrame(evolution_log)
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("📋 决策演化报表")
        st.dataframe(df.iloc[:, :-2]) # 隐藏上限下限列，保持整洁
        
        st.metric("最终黄金报价", f"{best_p:.2f}", f"下浮率 {round((1-best_p/limit_p)*100, 2)}%")
        st.warning(f"最终风险波动为 {df.iloc[-1]['风险值(波动)']} (数值越小代表开标现场越平稳)")

    with c2:
        st.subheader("📈 报价收敛与决策安全区")
        fig = go.Figure()
        
        # 绘制主报价演化线
        fig.add_trace(go.Scatter(x=df['轮次'], y=df['推荐报价'], name="推荐报价点", 
                                 line=dict(color='green', width=4), mode='lines+markers'))
        
        # 绘制演化边界阴影
        fig.add_trace(go.Scatter(x=df['轮次'], y=df['上限'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=df['轮次'], y=df['下限'], line=dict(width=0), 
                                 fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)', name='博弈平衡区'))
        
        fig.update_layout(xaxis_title="演化轮次", yaxis_title="报价金额")
        st.plotly_chart(fig, use_container_width=True)

    # 深度风险分析
    st.divider()
    st.subheader("🎯 最终轮次风险敏感度分布")
    st.markdown("该图展示了在最终区间内，不同报价对应的得分期望值及风险下浮动量。")
    
    sens_df = pd.DataFrame(round_results)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sens_df['price'], y=sens_df['avg'], name="综合得分均值", line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=sens_df['price'], y=sens_df['avg'] - sens_df['std'], 
                              name="风险波动下限", line=dict(dash='dot', color='red')))
    
    fig2.update_layout(xaxis_title="候选报价", yaxis_title="得分", hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👈 请在左侧配置参数，并点击【启动深度演化模拟】开始寻找黄金报价。")