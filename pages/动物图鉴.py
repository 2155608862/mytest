import streamlit as st

# 修改网页的文字和图标
st.set_page_config(page_title="相册", page_icon="🐱")
st.title("我的相册")

# 把当前图片的索引存储到streamlit的内存中，下面的代码将当前索引存储在内存中的ind变量中
# 如果没有找到ind，那就设置为0，否则不要设置ind
if 'ind' not in st.session_state:
    st.session_state['ind'] = 0

images = [
    {
        'url': "https://img95.699pic.com/photo/50276/6121.jpg_wh860.jpg",
        'text': "老虎"
    },
    {
        'url': "https://img95.699pic.com/photo/60027/3798.jpg_wh860.jpg",
        'text': "老鹰"
    },
    {
        'url': "https://img95.699pic.com/photo/60024/5902.jpg_wh300.jpg!/fh/300/quality/90",
        'text': "鹦鹉"
    }
]

# url:图片的地址  caption:图片注释介绍
st.image(images[st.session_state['ind']]['url'], caption=images[st.session_state['ind']]['text'])

def next_img():
    st.session_state['ind'] = (st.session_state['ind'] + 1) % len(images)

def last_img():
    st.session_state['ind'] = (st.session_state['ind'] - 1) % len(images)

# 分列容器 课本110页
c1, c2 = st.columns(2)
c1, c2 = st.columns((1, 2))
with c1:
    st.button("上一张", on_click=last_img, use_container_width=True)
with c2:
    # 按钮 课本73页
    st.button("下一张", on_click=next_img, use_container_width=True)
