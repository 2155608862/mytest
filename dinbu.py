import streamlit as st
import pandas as pd
import datetime

# 设置页面基础配置
st.set_page_config(
    page_title="综合应用平台",
    page_icon="📱",
    layout="wide"
)

# 全局CSS样式：设置白色背景、黑色字体，调整选项卡（导航栏）样式
st.markdown("""
    <style>
    /* 全局背景和字体 */
    .stApp {
        background-color: white !important;
        color: black !important;
    }
    /* 选项卡（导航栏）标签字体颜色为黑色 */
    button[data-baseweb="tab"] > div {
        color: black !important;
        font-weight: 500;
    }
    /* 选项卡选中状态的下划线颜色（可选，可根据需要调整） */
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom-color: #165dff !important;
    }
    /* 所有标题字体颜色为黑色 */
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }
    /* 普通文本、p标签、span标签字体颜色为黑色 */
    p, span, div {
        color: black !important;
    }
    /* 表格文字颜色为黑色 */
    .dataframe tbody tr td,
    .dataframe thead tr th {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

# 初始化会话状态，用于记录各页面所需索引
if 'video_ind' not in st.session_state:
    st.session_state['video_ind'] = 0
if 'album_ind' not in st.session_state:
    st.session_state['album_ind'] = 0
if 'music_ind' not in st.session_state:
    st.session_state['music_ind'] = 0

# 选项卡导航
st.title("综合应用平台")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["视频播放", "简历生成器", "相册", "音乐播放器", "美食数据", "陈奕迅档案"]
)

# 视频播放选项卡
with tab1:
    # 自定义样式：美化简介卡片+按钮
    st.markdown("""
        <style>
        /* 视频简介卡片样式 */
        .intro-card {
            background: linear-gradient(120deg, #f5f7fa 0%, #e4eaf5 100%);
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin: 15px 0;
            border: 1px solid #e0e6f0;
        }
        /* 当前集数标题 */
        .current-ep {
            font-size: 20px;
            font-weight: 700;
            color: #165dff;
            margin: 0 0 12px 0;
        }
        /* 搞笑简介文本 */
        .ep-desc {
            font-size: 15px;
            color: #333;
            line-height: 1.8;
            margin: 0;
            white-space: pre-line;
        }
        /* 按钮间距优化 */
        div[data-testid="column"] > button {
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title('兄弟们，开武魂！')

    # 视频数据
    video_arr = [
        {
            'url':'https://www.w3school.com.cn/example/html5/mov_bbb.mp4',
            'episode':1,
            'title':'第一集：武魂觉醒',
            'funny_desc': '''
唐三觉醒蓝银草武魂，当场蚌埠住了：这玩意儿能打过谁？
大师反手一个大逼兜：小子，这可是顶级武魂！
唐三：？？？（主打一个差生文具多，菜但装备牛）
觉醒完还得藏昊天锤，生怕被人抢，主打一个小心眼子拉满～
            '''
        },
        {
            'url':'https://www.w3schools.com/html/movie.mp4',
            'episode':2,
            'title':'第二集：史莱克入学',
            'funny_desc': '''
史莱克招生处标语：只收怪物！
唐三：我蓝银草+昊天锤，够怪不？
门卫大爷：滚，我们只要有头发的（误）！
最后靠小舞的兔子蹬鹰绝技，才勉强混进校门，
史莱克：面子不重要，有挂就行～
            '''
        },
        {
            'url':'https://media.w3.org/2010/05/sintel/trailer.mp4',
            'episode':3,
            'title':'第三集：首次魂环猎杀',
            'funny_desc': '''
第一次猎杀魂环就遇上百年曼陀罗蛇，唐三直接开「挂王模式」！
左手蓝银草捆人，右手昊天锤藏裤裆，
吓得魂兽连夜扛着火车跑路：这小子不按套路出牌！
最后魂环到手，唐三：就这？（装杯界的天花板）
            '''
        }
    ]

    # 播放当前选中的视频
    st.video(video_arr[st.session_state['video_ind']]['url'], autoplay=True)

    # 简介区域 
    current_video = video_arr[st.session_state['video_ind']]
    st.markdown(f"""
        <div class="intro-card">
            <p class="current-ep">当前播放：第{current_video['episode']}集 · {current_video['title']}</p>
            <p class="ep-desc">{current_video['funny_desc'].strip()}</p>
        </div>
    """, unsafe_allow_html=True)

    # 切换集数的函数
    def play(i):
        st.session_state['video_ind'] = int(i)

    # 集数按钮
    cols = st.columns(3)
    for idx, video in enumerate(video_arr):
        with cols[idx % 3]:
            st.button(
                f'第{video["episode"]}集',
                use_container_width=True,
                on_click=play,
                args=(idx,),
                type="primary" if idx == st.session_state['video_ind'] else "secondary"
            )

# 简历生成器选项卡
with tab2:
    # 自定义CSS样式（移除原有背景色，适配全局白色背景）
    st.markdown("""
        <style>
        /* 全局样式 */
        body {
            font-family: "Microsoft YaHei", sans-serif;
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

    # 分栏：左侧表单、右侧预览
    c1, c2 = st.columns([1, 2])

    with c1:
        # 左侧表单卡片容器
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 class="form-title">个人信息表单</h3>', unsafe_allow_html=True)
        
        # 表单逻辑
        user_name = st.text_input('姓名', placeholder='请输入您的姓名')
        user_zhiwei = st.text_input('职位', placeholder='请输入意向职位')
        user_iphone = st.text_input('电话', placeholder='请输入联系电话')
        user_youxiang = st.text_input('邮箱', placeholder='请输入邮箱地址')
        user_dizhi = st.text_input('地址', placeholder='请输入居住地址')
        user_shengri = st.date_input(
            "出生日期",
            value=datetime.date(2003, 1, 1),
            format="YYYY-MM-DD",
            min_value=datetime.date(1990, 1, 1)
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
        
        # 头像预览
        col_avatar, col_name = st.columns([1, 4])
        with col_avatar:
            if user_touxiang:
                st.image(user_touxiang, width=100, output_format='PNG', caption='头像', clamp=True)
            else:
                st.markdown('<div style="width:100px;height:100px;border:1px solid #ccc;display:flex;align-items:center;justify-content:center;color:#999;">上传头像</div>', unsafe_allow_html=True)
        
        with col_name:
            st.write(f"<h2 style='color:#2c3e50;margin:0;'>{user_name if user_name else '请填写姓名'}</h2>", unsafe_allow_html=True)
            st.write(f"<p style='color:#7f8c8d;margin:5px 0;'>{user_zhiwei if user_zhiwei else '意向职位待填写'}</p>", unsafe_allow_html=True)
        
        # 基本信息模块
        st.markdown('<div class="preview-module">', unsafe_allow_html=True)
        st.write("<h4 style='color:#3498db;margin:0 0 10px 0;'>基本信息</h4>", unsafe_allow_html=True)
        
        # 基本信息排版优化
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
        st.markdown('</div>', unsafe_allow_html=True)

# 相册选项卡
with tab3:
    # 页面标题
    st.title("我的相册")

    # 定义图片列表
    images = [
        {
            'url': "https://img1.baidu.com/it/u=2528176158,3706650567&fm=253&fmt=auto&app=138&f=JPEG?w=1200&h=800",
            'text': '猫'
        },
        {
            'url': "https://pic.616pic.com/photoone/00/02/40/618cf416207693898.jpg!/fw/1120",
            'text': 'dog'
        },
        {
            'url': "https://img95.699pic.com/photo/60017/6146.jpg_wh860.jpg",
            'text': 'lion'
        }
    ]

    # 显示当前图片
    st.image(images[st.session_state['album_ind']]['url'], caption=images[st.session_state['album_ind']]['text'])

    # 定义切换图片的函数
    def nextImg():
        st.session_state['album_ind'] = (st.session_state['album_ind'] + 1) % len(images)

    def prevImg():
        st.session_state['album_ind'] = (st.session_state['album_ind'] - 1) % len(images)

    # 创建按钮布局
    c1, c2 = st.columns(2)
    with c1:
        st.button("上一张", on_click=prevImg, use_container_width=True)
    with c2:
        st.button("下一张", on_click=nextImg, use_container_width=True)

# 音乐播放器选项卡
with tab4:
    # 页面标题和配置
    st.title("音乐播放器")

    # 音乐数据
    music_data = [
        {
            'url': "http://p2.music.126.net/EDhgL1S2DLGVE_5cjU-hfQ==/109951172410328709.jpg?param=130y130",
            'text': "大东北我的家乡",  
            'geshou':"袁娅维TIA RAY",
            'audio_url':'https://music.163.com/song/media/outer/url?id=3327141886'
        },
        {
            'url': "http://p2.music.126.net/sN5dTpmeJO1DhxIj1ogMLg==/109951163416453597.jpg?param=130y130",
            'text': "篝火旁",  
            'geshou':"吕大叶 / 马子林 / 陈觅Lynne",
            'audio_url':'https://music.163.com/song/media/outer/url?id=518725853'
        },
        {
            'url': "http://p1.music.126.net/RFbUrR2x2JEMB0WGYvwVQg==/109951169642392307.jpg?param=130y130",
            'text': "江南雪",  
            'geshou':"礼越",
            'audio_url':'https://music.163.com/song/media/outer/url?id=2161991028'
        }
    ]

    # 切换歌曲的函数
    def next_song():
        st.session_state['music_ind'] = (st.session_state['music_ind'] + 1) % len(music_data)

    def prev_song():
        st.session_state['music_ind'] = (st.session_state['music_ind'] - 1) % len(music_data)

    # 分栏：左侧显示封面，右侧显示歌曲信息
    a1, a2 = st.columns(2)
    with a1:
        st.image(music_data[st.session_state['music_ind']]['url'], use_container_width=True)
    with a2:
        st.subheader(music_data[st.session_state['music_ind']]['text'])
        st.write(f"歌手：{music_data[st.session_state['music_ind']]['geshou']}")
        
        # 按钮布局
        c1, c2 = st.columns(2)
        with c1:
            st.button("上一首", on_click=prev_song, use_container_width=True)
        with c2:
            st.button("下一首", on_click=next_song, use_container_width=True)

    # 音频播放
    audio_file = music_data[st.session_state['music_ind']]['audio_url']
    st.audio(audio_file)

# 美食数据选项卡
with tab5:
    # 自定义CSS样式（移除原有背景色，适配全局白色背景）
    st.markdown("""
        <style>
        /* 全局样式 */
        .stApp {
            padding: 0 2rem;            /* 页面左右边距 */
        }
        /* 标题样式 */
        h1, h2, h3 {
            font-family: "Microsoft YaHei", sans-serif;  /* 中文友好字体 */
        }
        /* 表格美化 */
        [data-testid="stDataFrame"] {
            border-radius: 8px;         /* 圆角 */
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);  /* 轻微阴影 */
            overflow: hidden;           /* 裁剪圆角 */
            margin-bottom: 2rem;        /* 底部间距 */
        }
        /* 图表美化 */
        [data-testid="stChart"] {
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 1rem;              /* 内边距 */
            background-color: white;    /* 白色背景突出图表 */
            margin-bottom: 2rem;
        }
        /* 地图美化 */
        [data-testid="stMap"] {
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            height: 600px;              /* 增大地图高度 */
            margin-bottom: 2rem;
        }
        /* 全局文字样式 */
        body {
            font-family: "Microsoft YaHei", sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)

    # 数据
    data = {
        "重庆小面":[240,250,280,330,360,300,350,340,360,300,390,330],
        "猪肚鸡":[220,250,190,200,210,270,230,290,240,250,194,236],
        "今邕烧烤":[200,150,180,230,260,200,250,240,260,200,190,230],
        "绿茶":[260,150,180,290,230,290,250,270,260,290,210,230],
        "海底捞":[200,180,230,270,210,200,210,240,210,260,230,270],
    }

    ind = pd.Series(['01月','02月','03月','04月','05月','06月'
                    ,'07月','08月','09月','10月','11月','12月'],name = '月份')

    df = pd.DataFrame(data, index=ind)

    # 页面内容
    st.title("🍜 南宁美食销量数据可视化")
    st.divider()

    # 表格
    st.subheader("📊 美食月度销量原始数据")
    st.dataframe(df, use_container_width=True)

    # 折线图
    st.subheader("📈 美食12个月销量走势折线图")
    st.line_chart(df, use_container_width=True)

    # 条形图
    st.subheader("📉 美食月度销量对比条形图")
    st.bar_chart(df, use_container_width=True)

    # 地图
    st.subheader("🗺️ 南宁美食店铺地理位置分布")
    map_data = {
        'latitude':[22.854095,22.854150,22.839873,22.844065,22.814558],
        'longitude':[108.222746,108.222864,108.245630,108.290863,108.322835]
    }

    mp_df = pd.DataFrame(map_data)
    st.map(mp_df, use_container_width=True)

# 陈奕迅档案选项卡（修改为白色背景+黑色字体，保留原有风格的同时适配全局）
with tab6:
    # 自定义CSS样式（替换深色宇宙风格为浅色系，适配全局白色背景）
    st.markdown("""
        <style>
        /* 子标题样式：科技青蓝+半透明下划线 */
        .stHeader {
            color: #06b6d4;  /* 子标题主色：科技青蓝 */
            border-bottom: 1px solid #06b6d480;  /* 半透明下划线分隔 */
            padding-bottom: 5px;
        }
        /* 指标标签样式：深灰色，作为辅助文字 */
        .metric-label {
            color: #666;
        }
        /* 指标数值样式：玫红，突出核心数据 */
        .metric-value {
            color: #ec4899;          /* 数值主色：玫红 */
            font-size: 1.5rem;       /* 字号放大 */
        }
        /* 表格表头样式：浅蓝背景+科技青蓝文字 */
        .dataframe thead tr th {
            background-color: #f0f9ff !important;  /* 表头背景：浅蓝 */
            color: #06b6d4 !important;             /* 表头文字：科技青蓝 */
            font-weight: bold;
            border: 1px solid #06b6d450;
        }
        /* 表格内容样式：白色背景+浅紫边框 */
        .dataframe tbody tr td {
            color: black !important;             /* 内容文字：黑色 */
            background-color: white !important;  /* 白色背景 */
            border: 1px solid #8b5cf630;
        }
        /* 代码块样式优化：浅灰背景+浅紫边框 */
        .stCodeBlock {
            background-color: #f8f9fa !important;  /* 代码块背景：浅灰 */
            border: 1px solid #8b5cf650;
        }
        </style>
    """, unsafe_allow_html=True)

    # 页面标题区域
    st.title("🎤 陈奕迅 档案 🌌", anchor="top")
    st.markdown("---")

    # 基础信息板块
    with st.container():
        st.header("🆔 档案基础信息", anchor="basic", help="艺人核心档案")
        st.markdown("""
        **艺人名**：陈奕迅 🎙️  
        **档案ID**：Singer-2000-001 🪐  
        **活跃状态**：<span style='color:#06b6d4;'>巅峰状态 🚀</span>  
        **音色标签**：<span style='color:#ec4899;'>磁性/沙哑/治愈 ✨</span>
        """, unsafe_allow_html=True)

    # 歌唱能力矩阵（星图）板块
    with st.container():
        st.header("🎚️ 歌唱能力星图", anchor="skills")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<p class='metric-label'>音域跨度 🎶</p>", unsafe_allow_html=True)
            st.markdown("<p class='metric-value'>3.8个八度</p>", unsafe_allow_html=True)
            st.markdown("<span style='color:#06b6d4;'>↑ 12% 📈</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<p class='metric-label'>声音迷人程度 💖</p>", unsafe_allow_html=True)
            st.markdown("<p class='metric-value'>98%</p>", unsafe_allow_html=True)
            st.markdown("<span style='color:#06b6d4;'>↑ 5% 📈</span>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("<p class='metric-label'>情感传达力 🫶</p>", unsafe_allow_html=True)
            st.markdown("<p class='metric-value'>99%</p>", unsafe_allow_html=True)
            st.markdown("<span style='color:#06b6d4;'>↑ 8% 📈</span>", unsafe_allow_html=True)

    # 高热度作品星表板块
    with st.container():
        st.header("📀 高热度作品星表", anchor="works")
        work_data = {
            "发行日期 📅": ["2021-11", "2003-04", "2006-11", "2010-03", "2007-04"],
            "作品名 🎵": ["《孤勇者》", "《十年》", "《爱情转移》", "《浮夸》", "《富士山下》"],
            "热度状态 📊": ["超新星级", "星系级封神", "星系级封神", "星系级封神", "星系级封神"],
            "热度评级 ⭐": ["⭐⭐⭐⭐⭐+", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"]
        }
        st.dataframe(work_data, hide_index=True, use_container_width=True)

    # 演唱核心算法板块
    with st.container():
        st.header("🔧 演唱核心算法", anchor="code")
        sing_code = """
def eason_sing(song, emotion_intensity=10):
    # 音色调制：磁性+沙哑融合 🎛️
    voice = magnetic_filter(original_voice) + hoarse_effect(voice)
    # 情感渲染：强度放大 💓
    voice = emotion_amplifier(voice, emotion_intensity)
    # 音准锁定：误差≤0.01Hz（高精度）🎯
    voice = pitch_lock(voice, precision=0.01)
    return voice

# 执行演唱 🎤
final_performance = eason_sing("孤勇者", emotion_intensity=12)
        """
        st.code(sing_code, language="python")

    # 系统状态信息板块
    st.markdown("---")
    st.markdown("""
    **系统状态**：在线 🟢  
    **当前模式**：功率输出 ⚡  
    **最后更新**：2025-12-11 15:30:00 🕒  
    **档案等级**：S级 🏆
    """)
