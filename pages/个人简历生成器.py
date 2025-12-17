import streamlit as st
import datetime

# 页面基础配置
st.set_page_config(page_title="个人简历生成器", page_icon="📝", layout="wide")

# 自定义CSS样式（新增大标题样式，移除头像美化）
st.markdown("""
    <style>
    /* 全局样式 */
    body {
        font-family: "Microsoft YaHei", sans-serif;
        background-color: #f8f9fa;
    }
    /* 页面大标题样式 */
    .main-title {
        color: #2c3e50;
        font-size: 36px;
        font-weight: 800;
        text-align: center;
        margin: 20px 0 30px 0;
        padding-bottom: 15px;
        border-bottom: 3px solid #3498db;
    }
    /* 卡片容器样式 */
    .card {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        padding: 20px;
        margin-bottom: 20px;
    }
    /* 表单标题样式 */
    .form-title {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 20px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    /* 预览标题样式 */
    .preview-title {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 20px;
        border-bottom: 2px solid #2ecc71;
        padding-bottom: 10px;
    }
    /* 输入框/选择器样式优化 */
    div[data-testid="stTextInput"],
    div[data-testid="stNumberInput"],
    div[data-testid="stDateInput"],
    div[data-testid="stSelectbox"],
    div[data-testid="stRadio"],
    div[data-testid="stMultiselect"],
    div[data-testid="stSlider"],
    div[data-testid="stTextArea"],
    div[data-testid="stFileUploader"] {
        margin-bottom: 15px;
    }
    /* 按钮样式优化 */
    button {
        border-radius: 8px !important;
    }
    /* 预览模块样式 */
    .preview-module {
        margin-bottom: 20px;
        padding: 15px;
        background-color: #f1f9ff;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# 页面顶部大标题
st.markdown('<h1 class="main-title">个人简历生成器</h1>', unsafe_allow_html=True)

# 分栏：左侧表单、右侧预览（保持原有比例）
c1, c2 = st.columns([1, 2])

with c1:
    # 左侧表单卡片容器
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="form-title">个人信息表单</h3>', unsafe_allow_html=True)
    
    # 原有表单逻辑（仅调整排版，功能不变）
    user_name = st.text_input('姓名', placeholder='请输入您的姓名')
    user_zhiwei = st.text_input('职位', placeholder='请输入意向职位')
    user_iphone = st.text_input('电话', placeholder='请输入联系电话')
    user_youxiang = st.text_input('邮箱', placeholder='请输入邮箱地址')
    user_dizhi = st.text_input('地址', placeholder='请输入居住地址')
    user_shengri = st.date_input(
        "出生日期",
        value=datetime.date(2003, 1, 1),  # 设置一个示例日期（唤起日历选择器）
        format="YYYY-MM-DD",  # 保持日期格式
        min_value=datetime.date(1990, 1, 1)  # 允许选择更早的日期（避免限制）
    )
    user_xingbie = st.radio('性别', ['男', '女', '其他'], horizontal=True)
    user_xueli = st.selectbox('学历', ['大专', '本科', '硕士', '博士'], placeholder='选择学历')
    
    user_jineng = st.multiselect(
        label="技能",
        options=["HTML/CSS", "JavaScript", "React", "Vue", "TypeScript", "Python", "Git", "Webpack"],
        default=[],
        placeholder='可多选，点击选择技能'
    )
    
    user_yuyan_nengli = st.multiselect(
        label="语言能力",
        options=["中文", "英语（CET-4）", "英语（CET-6）", "英语（雅思6.5+）", "英语（托福90+）", "日语（N1）", "日语（N2）", "韩语（TOPIK中级）", "德语（B1）"],
        default=[],
        placeholder='可多选，点击选择语言能力'
    )
    
    user_gongzuojingyan = st.slider('工作经验（年）', min_value=0, max_value=10, value=1, step=1)
    
    user_xinzi_range = st.slider(
        label="期望薪资范围（元）",
        min_value=5800,
        max_value=58000,
        value=(19123, 29390),
        step=100
    )
    
    user_gerenjianjie = st.text_area('个人简介', placeholder='请简要介绍自己（如工作经历、个人优势等）', height=100)
    user_touxiang = st.file_uploader('上传头像', type=['jpg', 'png', 'jpeg'], help='支持jpg/png/jpeg格式，建议尺寸200x200')
    
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    # 右侧预览卡片容器
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3 class="preview-title">简历实时预览</h3>', unsafe_allow_html=True)
    
    # 头像预览（恢复原生样式，移除所有美化）
    col_avatar, col_name = st.columns([1, 4])
    with col_avatar:
        if user_touxiang:
            # 原生图片渲染，移除美化样式和弃用参数
            st.image(user_touxiang, width=100, output_format='PNG', caption='头像', clamp=True)
        else:
            # 原生占位提示，移除圆形/虚线边框美化
            st.markdown('<div style="width:100px;height:100px;border:1px solid #ccc;display:flex;align-items:center;justify-content:center;color:#999;">上传头像</div>', unsafe_allow_html=True)
    
    with col_name:
        # 姓名美化（保留，不影响头像）
        st.write(f"<h2 style='color:#2c3e50;margin:0;'>{user_name if user_name else '请填写姓名'}</h2>", unsafe_allow_html=True)
        st.write(f"<p style='color:#7f8c8d;margin:5px 0;'>{user_zhiwei if user_zhiwei else '意向职位待填写'}</p>", unsafe_allow_html=True)
    
    # 基本信息模块（美化：独立卡片）
    st.markdown('<div class="preview-module">', unsafe_allow_html=True)
    st.write("<h4 style='color:#3498db;margin:0 0 10px 0;'>基本信息</h4>", unsafe_allow_html=True)
    
    # 基本信息排版优化（两列展示）
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.write(f"🗓 出生日期：{user_shengri.strftime('%Y-%m-%d') if user_shengri else '待填写'}")
        st.write(f"🚻 性别：{user_xingbie}")
        st.write(f"🎓 学历：{user_xueli}")
        st.write(f"📞 电话：{user_iphone if user_iphone else '待填写'}")
    with info_col2:
        st.write(f"📧 邮箱：{user_youxiang if user_youxiang else '待填写'}")
        st.write(f"📍 地址：{user_dizhi if user_dizhi else '待填写'}")
        st.write(f"💼 工作经验：{user_gongzuojingyan}年")
        st.write(f"💰 期望薪资：{user_xinzi_range[0]}-{user_xinzi_range[1]}元")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 技能模块
    st.markdown('<div class="preview-module">', unsafe_allow_html=True)
    st.write("<h4 style='color:#2ecc71;margin:0 0 10px 0;'>技能与语言能力</h4>", unsafe_allow_html=True)
    st.write(f"🛠 掌握技能：{', '.join(user_jineng) if user_jineng else '待选择技能'}")
    st.write(f"🗣 语言能力：{', '.join(user_yuyan_nengli) if user_yuyan_nengli else '待选择语言能力'}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 个人简介模块
    st.markdown('<div class="preview-module">', unsafe_allow_html=True)
    st.write("<h4 style='color:#9b59b6;margin:0 0 10px 0;'>个人简介</h4>", unsafe_allow_html=True)
    st.write(user_gerenjianjie if user_gerenjianjie else '<p style="color:#999;">待填写个人简介内容</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
