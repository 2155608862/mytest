import streamlit as st

# ========== 自定义样式：美化简介卡片+按钮 ==========
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
        line-height: 1.8;  /* 增大行高，读起来不挤 */
        margin: 0;
        white-space: pre-line;  /* 支持换行显示 */
    }
    /* 按钮间距优化 */
    div[data-testid="column"] > button {
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title('兄弟们，开武魂！')

# 视频数据：
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

# 初始化会话状态（记录当前播放集数索引）
if 'ind' not in st.session_state:
    st.session_state['ind'] = 0

# 播放当前选中的视频
st.video(video_arr[st.session_state['ind']]['url'], autoplay=True)

# 简介区域 
current_video = video_arr[st.session_state['ind']]
st.markdown(f"""
    <div class="intro-card">
        <p class="current-ep">当前播放：第{current_video['episode']}集 · {current_video['title']}</p>
        <p class="ep-desc">{current_video['funny_desc'].strip()}</p>
    </div>
""", unsafe_allow_html=True)

# 切换集数的函数
def play(i):
    st.session_state['ind'] = int(i)

# 集数按钮：三集一行 + 当前集高亮 
cols = st.columns(3)  # 创建3列布局
for idx, video in enumerate(video_arr):
    with cols[idx % 3]:  # 自动换行（3个一组）
        # 当前播放的集数按钮高亮（蓝色主按钮）
        st.button(
            f'第{video["episode"]}集',
            use_container_width=True,
            on_click=play,
            args=(idx,),
            type="primary" if idx == st.session_state['ind'] else "secondary"
        )
