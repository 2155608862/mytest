import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from joblib import load  # 仅替换模型加载方式，其余保留
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 基础配置（整合必要依赖） --------------------------
# 设置中文字体（避免图表中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 页面基础配置
st.set_page_config(
    page_title="学生成绩分析与预测系统",
    page_icon=":graduation_cap:",
    layout='wide'
)

# 路径配置（仅修改模型文件后缀为joblib，其余保留）
CONFIG = {
    "model_path": "rfr_model.joblib",  # 仅改这里：pkl→joblib
    "feature_names_path": "feature_names.pkl",
    "unique_values_path": "unique_values.pkl",
    "csv_path": "student_data_adjusted_rounded.csv"
}

# 加载模型和关键数据
@st.cache_resource
def load_resources():
    # 1. 加载训练好的模型和配置文件（仅修改模型加载为joblib，其余保留）
    model = load(CONFIG["model_path"])  # 替换pickle.load为joblib.load
    with open(CONFIG["feature_names_path"], 'rb') as f:
        feature_names = pickle.load(f)
    with open(CONFIG["unique_values_path"], 'rb') as f:
        unique_values = pickle.load(f)
    
    # 2. 加载CSV数据（完全保留你的原有逻辑）
    df = pd.read_csv(
        CONFIG["csv_path"],
        encoding='utf-8-sig',
        dtype={
            '学号': str,
            '性别': 'category',
            '专业': 'category'
        }
    ).dropna()
    
    return model, feature_names, unique_values, df

# 执行模型加载（全局仅加载一次）
model, feature_names, unique_values, df = load_resources()

# -------------------------- 1. 样式设置（完全保留你的原有美化效果） --------------------------
def set_normal_theme():
    st.markdown("""
    <style>
    /* 输入区域卡片优化 */
    .input-container {
        padding: 1.2rem;
        border-radius: 8px;
        background-color: #f0f2f6;
        margin-bottom: 1.5rem;
    }
    /* 标题样式优化 */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
    }
    .section-title::before {
        content: "📋";
        margin-right: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------- 2. 数据读取（完全保留你的原有兼容逻辑） --------------------------
def get_dataframe_from_csv():
    csv_path = "student_data_adjusted_rounded.csv"
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="gbk")
    
    core_cols = [
        "性别", "专业", "每周学习时长（小时）", 
        "上课出勤率", "期中考试分数", "期末考试分数"
    ]
    valid_cols = [col for col in core_cols if col in df.columns]
    return df[valid_cols].dropna() if valid_cols else pd.DataFrame()

# -------------------------- 3. 界面1：项目介绍页面（完全保留原功能+图片展示） --------------------------
def page1_project_intro():
    st.title("学生成绩分析与预测系统")
    
    # 项目概述
    with st.container():
        st.subheader("📋 项目概述")
        st.write("""
        本项目是一个基于Streamlit的学生成绩分析平台，通过该平台可可视化同学学习状态，帮助教育工作者和学生深入了解学习表现，并预测期末考试成绩。
        """)
        
        # 主要特点
        st.subheader("✨ 主要特点")
        st.markdown("""
        - **数据可视化**：多维度展示学生学业数据
        - **专业分析**：多维度的专业统计分析
        - **智能预测**：基于学习维度建模的成绩预测
        - **学习建议**：根据预测结果提供个性化反馈
        """)
    
    # 项目目标
    with st.container():
        st.subheader("🎯 项目目标")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 目标一：分析维度覆盖")
            st.write("- 识别关键学习指标\n- 探索维度相关性\n- 维度密度及分布")
        with col2:
            st.markdown("#### 目标二：可视化展示")
            st.write("- 专业对比分析\n- 性别差异分析\n- 学习习惯识别")
        with col3:
            st.markdown("#### 目标三：成绩预测")
            st.write("- 机器学习模型\n- 个性化反馈\n- 及时干预预警")
    
    # 技术架构
    with st.container():
        st.subheader("🔧 技术架构")
        arch_cols = st.columns(4)
        with arch_cols[0]:
            st.markdown("#### 前端框架\nStreamlit")
        with arch_cols[1]:
            st.markdown("#### 数据处理\nPandas\nNumPy")
        with arch_cols[2]:
            st.markdown("#### 可视化\nPlotly\nMatplotlib")
        with arch_cols[3]:
            st.markdown("#### 机器学习\nScikit-Learn")
    
    # 界面截图展示（恢复你原有图片展示代码）
    with st.container():
        st.subheader("🖼️ 系统界面预览")
        try:
            st.image("专业数据分析截图.png", caption="专业数据分析界面", use_container_width=True)
        except:
            st.warning("预览图片未找到，不影响功能使用")

# -------------------------- 4. 界面2：专业数据分析页面（完全保留原功能） --------------------------
def page2_major_analysis(df):
    st.title("专业数据分析")
    st.divider()

    # 需求1：核心指标表格
    st.subheader("📋 各专业核心学习指标")
    table_data = df.groupby("专业").agg({
        "每周学习时长（小时）": "mean",
        "期中考试分数": "mean",
        "期末考试分数": "mean"
    }).round(2).rename(
        columns={
            "每周学习时长（小时）": "每周平均学时（小时）",
            "期中考试分数": "期中考试平均分",
            "期末考试分数": "期末考试平均分"
        }
    ).reset_index()
    st.dataframe(table_data, use_container_width=True)
    st.divider()

    # 需求2：双层柱状图-性别比例
    st.subheader("📊 各专业男女性别比例")
    gender_count = df.groupby(["专业", "性别"]).size().reset_index(name="人数")
    fig_gender = px.bar(
        gender_count, x="专业", y="人数", color="性别", barmode="group",
        color_discrete_map={"男": "#1E88E5", "女": "#90CAF9"}
    )
    st.plotly_chart(fig_gender, use_container_width=True)
    st.divider()

    # 需求3：折线图-期中/期末分数
    st.subheader("📈 各专业期中/期末分数对比")
    exam_data = df.groupby("专业").agg({
        "期中考试分数": "mean", "期末考试分数": "mean"
    }).round(2).reset_index()
    exam_long = pd.melt(exam_data, id_vars="专业", 
                        value_vars=["期中考试分数", "期末考试分数"],
                        var_name="考试类型", value_name="平均分")
    fig_exam = px.line(
        exam_long, x="专业", y="平均分", color="考试类型", markers=True
    )
    st.plotly_chart(fig_exam, use_container_width=True)
    st.divider()

    # 需求4：单层柱状图-出勤率
    st.subheader("📊 各专业平均上课出勤率")
    attendance_data = df.groupby("专业")["上课出勤率"].mean().round(2).reset_index()
    fig_att = px.bar(
        attendance_data, x="专业", y="上课出勤率",
        color_discrete_sequence=["#4CAF50"]
    )
    st.plotly_chart(fig_att, use_container_width=True)
    st.divider()

    # 需求5：大数据管理专业专项
    st.subheader("🎯 大数据管理专业专项指标")
    bigdata_df = df[df["专业"] == "大数据管理"]
    if not bigdata_df.empty:
        bigdata_stats = bigdata_df.agg({
            "上课出勤率": "mean", "期末考试分数": "mean", "每周学习时长（小时）": "mean"
        }).round(2)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均出勤率", f"{bigdata_stats['上课出勤率']*100:.1f}%")
        with col2:
            st.metric("期末平均分", f"{bigdata_stats['期末考试分数']:.1f}分")
        with col3:
            st.metric("每周学习时长", f"{bigdata_stats['每周学习时长（小时）']:.1f}小时")
        fig_bigdata = px.bar(bigdata_df, x="性别", y="期末考试分数")
        st.plotly_chart(fig_bigdata, use_container_width=True)
    else:
        st.warning("未找到大数据管理专业数据")

# -------------------------- 5. 界面3：成绩预测页面（仅修改模型加载，完全保留图片展示） --------------------------
def page3_score_prediction():
    st.title("期末成绩预测")
    st.write("请输入学生的学习信息，系统将基于机器学习模型预测期末成绩并提供学习建议")
    st.divider()

    # 输入区域
    with st.container():
        st.markdown('<div class="section-title">学生信息输入</div>', unsafe_allow_html=True)
        col_left, col_right = st.columns([1, 1.5])  # 左窄右宽比例

        # 左侧：文本输入+下拉框（完全保留原有逻辑）
        with col_left:
            student_id = st.text_input("学号", placeholder="请输入学号（如2023001）")
            gender = st.selectbox("性别", options=unique_values['性别'], index=0)
            major = st.selectbox("专业", options=unique_values['专业'], index=0)
            # 预测按钮（左侧底部，宽按钮样式）
            predict_btn = st.button("预测期末成绩", type="primary", use_container_width=True)

        # 右侧：滑块组（完全保留原有逻辑）
        with col_right:
            study_hour = st.slider(
                "每周学习时长（小时）", 
                min_value=0.0, max_value=50.0, value=15.0, step=0.01
            )
            attendance = st.slider(
                "上课出勤率（%）", 
                min_value=0, max_value=100, value=90, step=1
            ) / 100  # 转换为小数（匹配模型训练格式）
            mid_score = st.slider(
                "期中考试分数", 
                min_value=0.0, max_value=100.0, value=60.0, step=0.01
            )
            homework_rate = st.slider(
                "作业完成率（%）", 
                min_value=0, max_value=100, value=80, step=1
            ) / 100  # 转换为小数（匹配模型训练格式）

    # 预测结果展示（完全保留你的原有美化+图片展示逻辑）
    if predict_btn:
        # 验证必填项（学号可选，核心特征必填）
        if study_hour == 0 or attendance == 0 or mid_score == 0 or homework_rate == 0:
            st.error("请完善学习数据输入（学习时长、出勤率、期中分数、作业完成率不能为空）")
            return

        st.divider()
        st.subheader("📊 预测结果")
        
        # 构造模型输入数据（仅保留逻辑，未删减）
        input_data = {feat: 0 for feat in feature_names}
        # 填充数值型特征
        input_data['每周学习时长（小时）'] = study_hour
        input_data['上课出勤率'] = attendance
        input_data['期中考试分数'] = mid_score
        input_data['作业完成率'] = homework_rate
        # 填充独热编码的分类特征
        gender_feat = f"性别_{gender}"
        major_feat = f"专业_{major}"
        if gender_feat in input_data:
            input_data[gender_feat] = 1
        if major_feat in input_data:
            input_data[major_feat] = 1
        
        # 转换为DataFrame（保证列顺序与训练时一致）
        input_df = pd.DataFrame([input_data], columns=feature_names)
        # 模型预测（仅用joblib加载的模型，逻辑不变）
        final_score = model.predict(input_df)[0]
        final_score = round(final_score, 1)

        # 结果展示（完全保留metric+图片展示）
        st.metric("预测期末成绩", f"{final_score}分", delta=None)

        # 结果提示+图片（完全恢复你原有图片展示代码）
        if final_score >= 60:
            st.success("🎉 恭喜！预测成绩及格啦！继续保持优秀表现~")
            try:
                st.image("恭喜.png", caption="成绩优秀！", width=250)
            except:
                st.markdown("📌 建议：保持当前学习节奏，重点巩固薄弱知识点")
        else:
            st.warning("💪 没关系！预测成绩暂未及格，针对性提升后可显著进步")
            try:
                st.image("鼓励.png", caption="继续努力！", width=250)
            except:
                st.markdown("📌 建议：参考下方学习建议，重点优化薄弱环节")

        # 个性化学习建议（完全保留原有逻辑）
        st.subheader("📌 个性化学习建议")
        advice_list = []
        if study_hour < 15:
            advice_list.append("- 每周学习时长不足15小时，建议增加至15-25小时，分时段高效学习")
        if attendance < 0.9:
            advice_list.append("- 上课出勤率低于90%，建议提高出勤，紧跟老师教学节奏，及时答疑")
        if homework_rate < 0.8:
            advice_list.append("- 作业完成率低于80%，建议按时完成作业，通过练习巩固知识点")
        if mid_score < 60:
            advice_list.append("- 期中考试分数偏低，建议复盘错题，针对性补强核心知识点")
        if mid_score >= 80 and final_score < 70:
            advice_list.append("- 期中成绩优秀但期末预测偏低，建议加强知识综合应用训练")
        
        if advice_list:
            for advice in advice_list:
                st.markdown(advice)
        else:
            st.markdown("- 当前学习状态良好，保持现有节奏，重点提升知识深度和应用能力")

# -------------------------- 主函数：导航+页面切换（完全保留原逻辑） --------------------------
def main():
    # 设置美化样式
    set_normal_theme()

    # 左侧导航菜单
    with st.sidebar:
        st.title("导航菜单")
        st.write("选择功能页面")
        selected_page = st.radio(
            " ",
            ["项目介绍", "专业数据分析", "成绩预测"],
            index=2  # 默认选中“成绩预测”页
        )

    # 页面切换逻辑
    if selected_page == "项目介绍":
        page1_project_intro()
    elif selected_page == "专业数据分析":
        df = get_dataframe_from_csv()
        if df.empty:
            st.error("未读取到有效数据，请核对CSV路径和列名")
        else:
            page2_major_analysis(df)
    elif selected_page == "成绩预测":
        page3_score_prediction()

if __name__ == "__main__":
    main()
