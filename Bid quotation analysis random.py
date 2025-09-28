import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
matplotlib.use('Agg')  # 避免GUI警告
import warnings
warnings.filterwarnings('ignore')
import os

font_path = os.path.join(os.path.dirname(__file__), "NotoSerifCJKsc-Regular.otf")
myfont = fm.FontProperties(fname=font_path)


# 确保负号显示正常
matplotlib.rcParams['axes.unicode_minus'] = False


# 设置页面配置（必须在所有Streamlit命令之前）
st.set_page_config(
    page_title="投标报价优化分析", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题
st.title("📊 投标报价优化分析工具")
st.markdown("基于概率论分析两种摇珠方法下的最优投标报价策略")

# 侧边栏输入参数
st.sidebar.header("🎯 参数设置")

with st.sidebar.expander("基本参数", expanded=True):
    x = st.number_input("下浮率下限 X (%)", min_value=0.0, max_value=50.0, value=5.0, step=0.1, help="摇珠下浮率的最小值")
    y = st.number_input("下浮率上限 Y (%)", min_value=0.0, max_value=50.0, value=8.0, step=0.1, help="摇珠下浮率的最大值")
    d1 = st.number_input("高价惩罚系数 D1", min_value=0.1, max_value=10.0, value=2.0, step=0.1, help="报价高于基准价时的扣分系数")
    d2 = st.number_input("低价惩罚系数 D2", min_value=0.1, max_value=10.0, value=1.0, step=0.1, help="报价低于基准价时的扣分系数")

# 设置样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# 计算得分函数
def calculate_score(bid_price, benchmark_price, d1, d2):
    """计算评标价得分"""
    if benchmark_price == 0:
        return 0
    
    deviation = abs(bid_price - benchmark_price) / benchmark_price
    
    if bid_price > benchmark_price:
        score = 95 - deviation * 100 * d1
    else:
        score = 95 - deviation * 100 * d2
    
    # 确保得分不会为负数
    return max(0, round(score, 4))

# 方法一：单球摇珠的概率计算（优化版本）
def method1_expected_score(bid_price, min_discount, max_discount, d1, d2, num_points=200):
    """计算方法一的期望得分（单球摇珠）"""
    discount_rates = np.linspace(min_discount, max_discount, num_points)
    benchmark_prices = 1 - discount_rates
    
    # 使用向量化计算提高性能
    deviations = np.abs(bid_price - benchmark_prices) / benchmark_prices
    scores = np.where(bid_price > benchmark_prices, 
                     95 - deviations * 100 * d1, 
                     95 - deviations * 100 * d2)
    
    # 确保得分非负
    scores = np.maximum(scores, 0)
    return np.mean(scores)

# 方法二：三球平均值的概率计算（优化版本）
def method2_expected_score(bid_price, min_discount, max_discount, d1, d2, num_samples=2000):
    """计算方法二的期望得分（三球平均值）"""
    # 使用向量化蒙特卡洛模拟提高性能
    n = num_samples
    # 一次性生成所有样本
    three_discounts = np.random.uniform(min_discount, max_discount, (n, 3))
    avg_discounts = np.mean(three_discounts, axis=1)
    # 四舍五入到0.001%
    avg_discounts = np.round(avg_discounts, 5)
    benchmark_prices = 1 - avg_discounts
    
    # 向量化计算得分
    deviations = np.abs(bid_price - benchmark_prices) / benchmark_prices
    scores = np.where(bid_price > benchmark_prices, 
                     95 - deviations * 100 * d1, 
                     95 - deviations * 100 * d2)
    
    # 确保得分非负
    scores = np.maximum(scores, 0)
    return np.mean(scores)

# 主计算函数（优化性能）
def calculate_optimal_bid(x, y, d1, d2):
    """计算最优投标报价"""
    # 转换百分比为小数
    min_discount = x / 100
    max_discount = y / 100
    
    # 优化：减少计算点数，使用智能采样
    if (max_discount - min_discount) <= 0.05:  # 范围较小时使用更密的采样
        discount_step = 1 / 5000  # 0.02%
    else:
        discount_step = 1 / 2000  # 0.05%
    
    bid_discounts = np.arange(min_discount, max_discount + discount_step, discount_step)
    # 确保不超过范围
    bid_discounts = bid_discounts[bid_discounts <= max_discount]
    bid_prices = 1 - bid_discounts
    
    results = []
    total_points = len(bid_discounts)
    
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (bid_discount, bid_price) in enumerate(zip(bid_discounts, bid_prices)):
        # 更新进度
        progress = (i + 1) / total_points
        progress_bar.progress(progress)
        if i % 10 == 0:  # 每10次更新一次状态，减少频繁更新
            status_text.text(f"计算进度: {i+1}/{total_points} ({progress*100:.1f}%)")
        
        try:
            # 计算方法一的期望得分
            method1_score = method1_expected_score(bid_price, min_discount, max_discount, d1, d2)
            
            # 计算方法二的期望得分
            method2_score = method2_expected_score(bid_price, min_discount, max_discount, d1, d2)
            
            # 总期望得分（两种方法各50%概率）
            expected_score = 0.5 * method1_score + 0.5 * method2_score
            
            results.append({
                'bid_discount': bid_discount,
                'bid_discount_pct': bid_discount * 100,
                'bid_price': bid_price,
                'bid_price_pct': bid_price * 100,
                'method1_score': method1_score,
                'method2_score': method2_score,
                'expected_score': expected_score
            })
        except Exception as e:
            # 跳过错误点继续计算
            continue
    
    # 清除进度条
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        st.error("计算过程中出现错误，请调整参数后重试")
        return None, None
    
    df = pd.DataFrame(results)
    
    # 找出最优报价
    optimal_idx = df['expected_score'].idxmax()
    optimal_bid = df.loc[optimal_idx]
    
    return df, optimal_bid

# 绘制图表函数
def plot_results(df, optimal_bid, x, y, d1, d2):
    """绘制结果图表"""
    # 设置中文字体
    print("✅ 成功加载中文字体:", font_path)
    plt.rcParams['font.sans-serif'] = [myfont.get_name()]
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 图1：总期望得分
    ax1.plot(df['bid_price_pct'], df['expected_score'], 
             label='总期望得分', color='#3498db', linewidth=2.5)
    ax1.axvline(x=optimal_bid['bid_price_pct'], color='red', linestyle='--', 
                label=f'最优报价: {optimal_bid["bid_price_pct"]:.3f}%', linewidth=2)
    ax1.set_xlabel('投标报价（相对于最高限价的百分比）', fontsize=12)
    ax1.set_ylabel('期望得分', fontsize=12)
    ax1.set_title('投标报价与期望得分关系', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(85, 95)
    
    # 图2：两种方法得分对比
    ax2.plot(df['bid_price_pct'], df['method1_score'], 
             label='方法一得分', color='#2ecc71', linewidth=2)
    ax2.plot(df['bid_price_pct'], df['method2_score'], 
             label='方法二得分', color='#e74c3c', linewidth=2)
    ax2.axvline(x=optimal_bid['bid_price_pct'], color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('投标报价（相对于最高限价的百分比）', fontsize=12)
    ax2.set_ylabel('得分', fontsize=12)
    ax2.set_title('两种摇珠方法得分对比', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(85, 95)
    
    # 图3：得分分布直方图
    ax3.hist(df['expected_score'], bins=30, alpha=0.7, color='#3498db', edgecolor='black')
    ax3.axvline(x=optimal_bid['expected_score'], color='red', linestyle='--', 
                label=f'最优得分: {optimal_bid["expected_score"]:.4f}', linewidth=2)
    ax3.set_xlabel('期望得分', fontsize=12)
    ax3.set_ylabel('频数', fontsize=12)
    ax3.set_title('期望得分分布', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 图4：下浮率与得分关系
    ax4.plot(df['bid_discount_pct'], df['expected_score'], 
             color='#9b59b6', linewidth=2.5)
    ax4.axvline(x=optimal_bid['bid_discount_pct'], color='red', linestyle='--',
                label=f'最优下浮率: {optimal_bid["bid_discount_pct"]:.3f}%', linewidth=2)
    ax4.set_xlabel('投标下浮率 (%)', fontsize=12)
    ax4.set_ylabel('期望得分', fontsize=12)
    ax4.set_title('下浮率与期望得分关系', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(85, 95)
    
    plt.tight_layout()
    return fig

# 主程序
def main():
    if st.sidebar.button("🚀 开始计算最优报价", type="primary", use_container_width=True):
        # 验证输入
        if x >= y:
            st.error("❌ 错误：下浮率下限X必须小于上限Y")
            return
        
        # 显示参数摘要
        st.sidebar.info(f"""
        **参数摘要:**
        - 下浮率范围: {x}% - {y}%
        - 惩罚系数: D1={d1}, D2={d2}
        """)
        
        # 计算最优报价
        with st.spinner("🔍 正在计算最优投标报价，请稍候..."):
            df, optimal_bid = calculate_optimal_bid(x, y, d1, d2)
        
        if df is not None and optimal_bid is not None:
            st.success("✅ 计算完成！")
            
            # 显示最优结果
            st.subheader("🎯 最优报价结果")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="最优下浮率",
                    value=f"{optimal_bid['bid_discount_pct']:.3f}%",
                    delta=f"范围: {x}%-{y}%"
                )
            
            with col2:
                st.metric(
                    label="最优报价",
                    value=f"{optimal_bid['bid_price_pct']:.3f}%",
                    delta="相对于最高限价"
                )
            
            with col3:
                st.metric(
                    label="预期得分",
                    value=f"{optimal_bid['expected_score']:.4f}",
                    delta="最高得分"
                )
            
            with col4:
                st.metric(
                    label="方法得分",
                    value=f"{optimal_bid['method1_score']:.2f}|{optimal_bid['method2_score']:.2f}",
                    delta="方法一|方法二"
                )
            
            # 显示图表
            st.subheader("📈 分析图表")
            fig = plot_results(df, optimal_bid, x, y, d1, d2)
            st.pyplot(fig)
            
            # 显示详细数据
            st.subheader("📊 详细数据")
            
            # 前10个最佳报价
            st.write("**前10个最佳报价方案：**")
            top_10 = df.nlargest(10, 'expected_score')[['bid_discount_pct', 'bid_price_pct', 
                                                       'method1_score', 'method2_score', 'expected_score']]
            top_10.columns = ['下浮率(%)', '报价(%)', '方法一得分', '方法二得分', '总期望得分']
            top_10.index = range(1, 11)
            
            # 格式化显示
            styled_df = top_10.style.format({
                '下浮率(%)': '{:.3f}',
                '报价(%)': '{:.3f}',
                '方法一得分': '{:.4f}',
                '方法二得分': '{:.4f}',
                '总期望得分': '{:.4f}'
            }).background_gradient(subset=['总期望得分'], cmap='YlOrRd')
            
            st.dataframe(styled_df, use_container_width=True)
            
            # 分析说明
            with st.expander("💡 分析说明和建议", expanded=True):
                st.markdown(f"""
                ### 分析结果说明
                
                根据当前参数设置，最优投标策略为：
                - **下浮率**: {optimal_bid['bid_discount_pct']:.3f}%
                - **报价**: {optimal_bid['bid_price_pct']:.3f}% (最高限价的{optimal_bid['bid_price_pct']:.3f}%)
                - **预期得分**: {optimal_bid['expected_score']:.4f}
                
                ### 策略建议
                1. **报价定位**: 当前最优报价位于下浮率范围的{((optimal_bid['bid_discount_pct'] - x) / (y - x) * 100):.1f}%位置
                2. **风险分析**: 方法二（三球平均）的得分{optimal_bid['method2_score']:.2f} vs 方法一（单球）的得分{optimal_bid['method1_score']:.2f}
                3. **稳定性**: 得分曲线在最优值附近{'较为平缓' if np.std(df['expected_score'].tail(10)) < 0.1 else '变化较大'}
                
                ### 参数敏感性
                - D1/D2比值: {d1/d2:.2f} ({'高价惩罚较重' if d1 > d2 else '低价惩罚较重' if d1 < d2 else '惩罚对称'})
                - 下浮率范围宽度: {y-x:.1f}% ({'范围较宽' if (y-x) > 3 else '范围适中' if (y-x) > 1 else '范围较窄'})
                """)

    else:
        # 默认显示说明
        st.info("👈 请在左侧设置参数并点击'开始计算最优报价'")
        
        # 显示示例说明
        with st.expander("📖 使用说明和示例", expanded=True):
            st.markdown("""
            ### 使用说明
            
            1. **设置参数**: 在左侧输入下浮率范围和惩罚系数
            2. **开始计算**: 点击"开始计算最优报价"按钮
            3. **查看结果**: 系统将显示最优报价策略和详细分析
            
            ### 参数含义
            - **X, Y**: 摇珠下浮率的范围，如5%-8%表示下浮率在5%到8%之间随机产生
            - **D1**: 报价高于基准价时的惩罚系数，值越大扣分越多
            - **D2**: 报价低于基准价时的惩罚系数，值越大扣分越多
            
            ### 摇珠规则
            - **方法一**: 摇取1个球，直接确定下浮率
            - **方法二**: 摇取3个球，取平均值作为下浮率
            - **选择机制**: 开标时随机选择方法一或方法二（各50%概率）
            
            ### 评分规则
            得分 = 95 - 偏差率 × 100 × 惩罚系数
            - 偏差率 = |报价 - 基准价| / 基准价
            - 基准价 = 最高限价 × (1 - 下浮率)
            """)

# 运行主程序
if __name__ == "__main__":
    main()

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "基于概率论和统计分析的投标报价优化工具 • 使用Streamlit构建"
    "</div>", 
    unsafe_allow_html=True
)
