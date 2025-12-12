# 导入Streamlit库，用于快速构建交互式Web应用
import streamlit as st

# ===================== 页面基础配置 =====================
# 设置页面基础属性：标题、图标、布局（宽屏），契合宇宙风格的陈奕迅档案主题
st.set_page_config(
    page_title="陈奕迅档案",  # 浏览器标签页标题
    page_icon="🎤",          # 页面图标（麦克风，贴合歌手主题）
    layout="wide"            # 宽屏布局，适配大屏展示星际档案
)

# ===================== 自定义CSS样式（宇宙风格） =====================
# 通过markdown注入自定义CSS，实现宇宙星际风格的视觉效果
st.markdown("""
    <style>
    /* 全局应用背景：深空蓝紫渐变，模拟宇宙星云质感 */
    .stApp {
        background-color: #0a0e27;
        background-image: linear-gradient(160deg, #0a0e27 0%, #1a2240 100%);
        color: #e0e7ff;  /* 全局文字主色：浅蓝白，模拟星光 */
    }
    /* 标题样式：幻彩紫+霓虹光晕，模拟星云/星轨的荧光效果 */
    .stTitle {
        color: #8b5cf6;  /* 主色调：幻彩紫 */
        text-shadow: 0 0 15px #8b5cf6, 0 0 30px #8b5cf680;  /* 光晕效果增强宇宙感 */
    }
    /* 子标题样式：科技青蓝+半透明下划线，模拟星际科技界面 */
    .stHeader {
        color: #06b6d4;  /* 子标题主色：科技青蓝 */
        border-bottom: 1px solid #06b6d480;  /* 半透明下划线分隔 */
        padding-bottom: 5px;  /* 底部内边距，优化视觉间距 */
    }
    /* 指标标签样式：浅灰蓝，作为辅助文字不抢视觉焦点 */
    .metric-label {
        color: #94a3b8;
    }
    /* 指标数值样式：玫红荧光+光晕，模拟超新星的高亮效果 */
    .metric-value {
        color: #ec4899;          /* 数值主色：玫红荧光 */
        font-size: 1.5rem;       /* 字号放大，突出核心数据 */
        text-shadow: 0 0 8px #ec4899, 0 0 20px #ec489980;  /* 光晕增强视觉冲击 */
    }
    /* 表格表头样式：深色背景+科技青蓝文字，强化星际科技感 */
    .dataframe thead tr th {
        background-color: #1e293b !important;  /* 表头背景：深空灰 */
        color: #06b6d4 !important;             /* 表头文字：科技青蓝 */
        font-weight: bold;                     /* 文字加粗，突出表头 */
        border: 1px solid #06b6d450;           /* 半透明边框，增加层次感 */
    }
    /* 表格内容样式：半透明深色背景+浅紫边框，贴合宇宙渐变风格 */
    .dataframe tbody tr td {
        color: #e0e7ff !important;             /* 内容文字：浅蓝白 */
        background-color: #1e293b80 !important;/* 半透明背景，增加通透感 */
        border: 1px solid #8b5cf630;           /* 浅紫边框，呼应标题色调 */
    }
    /* 代码块样式优化：深色背景+浅紫边框，统一宇宙风格 */
    .stCodeBlock {
        background-color: #1e293b !important;  /* 代码块背景：深空灰 */
        border: 1px solid #8b5cf650;           /* 浅紫边框，增加科技感 */
    }
    </style>
""", unsafe_allow_html=True)  # 允许HTML渲染，使CSS生效

# ===================== 页面标题区域 =====================
# 主标题：结合麦克风、行星表情，突出“星际档案”的宇宙主题
st.title("🎤 陈奕迅 星际档案 🌌", anchor="top")
# 分隔线：视觉分隔标题与内容，优化页面布局
st.markdown("---")

# ===================== 基础信息板块 =====================
# 使用container容器包裹，模块化管理页面元素，便于样式控制和布局
with st.container():
    # 子标题：🆔表情+文字，标识“基础信息”板块，anchor用于锚点定位
    st.header("🆔 星际档案基础信息", anchor="basic", help="艺人核心档案")
    # 基础信息展示：使用markdown支持富文本（颜色、表情），贴合宇宙风格
    st.markdown("""
    **艺人名**：陈奕迅 🎙️  <!-- 麦克风表情，贴合歌手身份 -->  
    **档案ID**：Singer-2000-001 🪐  <!-- 行星表情，呼应宇宙主题 -->  
    **活跃状态**：<span style='color:#06b6d4;'>星际级巅峰 🚀</span>  <!-- 科技青蓝+火箭，突出巅峰状态 -->  
    **音色标签**：<span style='color:#ec4899;'>磁性/沙哑/治愈 ✨</span>  <!-- 玫红荧光+星光，突出音色特质 -->
    """, unsafe_allow_html=True)  # 允许HTML渲染，使颜色样式生效

# ===================== 歌唱能力矩阵（星图）板块 =====================
with st.container():
    # 子标题：🎚️表情+“歌唱能力星图”，替换“矩阵”更贴合宇宙风格
    st.header("🎚️ 歌唱能力星图", anchor="skills")
    # 分3列布局，平均展示3个核心能力指标，优化大屏视觉效果
    col1, col2, col3 = st.columns(3)
    
    # 第一列：音域跨度指标
    with col1:
        # 指标标签：音域跨度+🎶表情，贴合音乐属性
        st.markdown("<p class='metric-label'>音域跨度 🎶</p>", unsafe_allow_html=True)
        # 指标数值：核心数据，使用自定义样式高亮
        st.markdown("<p class='metric-value'>3.8个八度</p>", unsafe_allow_html=True)
        # 涨幅提示：科技青蓝+📈表情，展示数据增长
        st.markdown("<span style='color:#06b6d4;'>↑ 12% 📈</span>", unsafe_allow_html=True)
    
    # 第二列：声音迷人程度指标
    with col2:
        st.markdown("<p class='metric-label'>声音迷人程度 💖</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-value'>98%</p>", unsafe_allow_html=True)
        st.markdown("<span style='color:#06b6d4;'>↑ 5% 📈</span>", unsafe_allow_html=True)
    
    # 第三列：情感传达力指标
    with col3:
        st.markdown("<p class='metric-label'>情感传达力 🫶</p>", unsafe_allow_html=True)
        st.markdown("<p class='metric-value'>99%</p>", unsafe_allow_html=True)
        st.markdown("<span style='color:#06b6d4;'>↑ 8% 📈</span>", unsafe_allow_html=True)

# ===================== 高热度作品星表板块 =====================
with st.container():
    # 子标题：📀表情+“高热度作品星表”，替换“日志”更贴合宇宙风格
    st.header("📀 高热度作品星表", anchor="works")
    # 构建陈奕迅全网热度TOP作品数据，贴合宇宙风格命名状态
    work_data = {
        "发行日期 📅": ["2021-11", "2003-04", "2006-11", "2010-03", "2007-04"],  # 日期列+日历表情
        "作品名 🎵": ["《孤勇者》", "《十年》", "《爱情转移》", "《浮夸》", "《富士山下》"],  # 作品列+音符表情
        "热度状态 📊": ["超新星级", "星系级封神", "星系级封神", "星系级封神", "星系级封神"],  # 状态列+图表表情
        "热度评级 ⭐": ["⭐⭐⭐⭐⭐+", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]  # 评级列+星星表情
    }
    # 渲染数据表格：隐藏索引、自适应宽度，应用自定义宇宙风格样式
    st.dataframe(work_data, hide_index=True, use_container_width=True)

# ===================== 演唱核心算法板块 =====================
with st.container():
    # 子标题：🔧表情+“演唱核心星核算法”，替换“核心算法”更贴合宇宙风格
    st.header("🔧 演唱核心星核算法", anchor="code")
    # 定义演唱算法代码块，添加宇宙风格注释，模拟星际科技逻辑
    sing_code = """
def eason_sing(song, emotion_intensity=10):
    # 音色调制：磁性+沙哑星际融合 🎛️
    voice = magnetic_filter(original_voice) + hoarse_effect(voice)
    # 情感渲染：星际级强度放大 💓
    voice = emotion_amplifier(voice, emotion_intensity)
    # 音准锁定：误差≤0.01Hz（星际精度）🎯
    voice = pitch_lock(voice, precision=0.01)
    return voice

# 执行星际演唱 🎤
final_performance = eason_sing("孤勇者", emotion_intensity=12)
    """
    # 渲染代码块：指定Python语言，应用自定义宇宙风格样式
    st.code(sing_code, language="python")

# ===================== 系统状态信息板块 =====================
# 分隔线：视觉分隔内容与页脚状态信息
st.markdown("---")
# 系统状态展示：使用markdown+表情，贴合宇宙风格的状态描述
st.markdown("""
**系统状态**：星际在线 🟢  <!-- 绿色圆点，标识在线状态 -->  
**当前模式**：超新星功率输出 ⚡  <!-- 闪电表情，突出高功率 -->  
**最后更新**：2025-12-11 15:30:00 🕒  <!-- 时钟表情，标识时间 -->  
**档案等级**：银河系S级 🏆  <!-- 奖杯表情，突出顶级档案 -->
""")
