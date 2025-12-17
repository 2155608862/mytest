import streamlit as st

# 修改网页的文字和图标
st.set_page_config(page_title="音乐播放器", page_icon="🐱")
st.title("音乐播放器")

# 初始化当前索引的会话状态
if 'ind' not in st.session_state:
    st.session_state['ind'] = 0

# 修复：字典键值对添加缺失的逗号
images = [
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
def next_img():
    st.session_state['ind'] = (st.session_state['ind'] + 1) % len(images)

def last_img():
    st.session_state['ind'] = (st.session_state['ind'] - 1) % len(images)



# 分栏：左侧显示封面，右侧显示歌曲/歌手信息
a1, a2 = st.columns(2)
with a1:
    # 修复：闭合st.image的括号
    st.image(images[st.session_state['ind']]['url'], use_container_width=True)
with a2:
    # 优化：用st.write显示歌曲和歌手（替代错误的caption拆分）
    st.subheader(images[st.session_state['ind']]['text'])
    st.write(f"歌手：{images[st.session_state['ind']]['geshou']}")
    # 分栏放“上一张/下一张”按钮（移除重复的列定义）
    c1, c2 = st.columns(2)
    with c1:
        st.button("上一首", on_click=last_img, width="stretch")
    with c2:
        st.button("下一首", on_click=next_img, width="stretch")

audio_file = images[st.session_state['ind']]['audio_url']
st.audio(audio_file)
