import streamlit as st
import pickle
import pandas as pd

# 加载保存的特征名（关键：与模型训练时的特征名保持一致）
try:
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except FileNotFoundError:
    st.error("特征名文件feature_names.pkl未找到，请先运行train_model.py生成！")
    st.stop()

def introduce_page():
    """当选择简介页面时，将呈现该函数的内容"""
    st.write("# 欢迎使用！")
    st.sidebar.success("单击 预测医疗费用")
    st.markdown("""
# 医疗费用预测应用
这个应用利用机器学习模型来预测医疗费用，为保险公司的保险定价提供参考。

## 背景介绍
1. 开发目标：帮助保险公司合理定价保险产品，控制风险。
2. 模型算法：利用随机森林回归算法训练医疗费用预测模型。

## 使用指南
1. 输入准确完整的被保险人信息，可以得到更准确的费用预期。
2. 预测结果可以作为保险定价的重要参考，但需审慎决策。
- 有任何问题欢迎联系我们的技术支持。

技术支持：email：support@example.com
    """)

def predict_page():
    """当选择预测费用页面时，将呈现该函数的内容"""
    st.markdown("""
## 使用说明
这个应用利用机器学习模型来预测医疗费用，为保险公司的保险定价提供参考。

**输入信息**：在下面输入被保险人的个人信息、疾病信息等。
**费用预测**：应用会预测被保险人的未来医疗费用支出。
    """)
    # 运用表单和表单提交按钮
    with st.form('user_inputs'):
        age = st.number_input('年龄', min_value=0, value=25)  # 添加默认值提升体验
        sex = st.radio('性别', options=['男性', '女性'])
        bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=22.5)  # 限制最大值+默认值
        children = st.number_input("子女数量：", step=1, min_value=0, value=0)
        smoke = st.radio("是否吸烟", ("是", "否"))
        region = st.selectbox('区域', ('东南部', '西南部', '东北部', '西北部'))
        submitted = st.form_submit_button('预测费用')

        if submitted:
            # ========== 特征编码（与模型训练时的逻辑完全一致） ==========
            # 1. 初始化所有特征值为0（根据训练好的特征名初始化）
            feature_values = {name: 0 for name in feature_names}

            # 2. 赋值数值特征（age, bmi, children）
            feature_values['age'] = age
            feature_values['bmi'] = bmi
            feature_values['children'] = children

            # 3. 赋值性别独热编码（sex_female, sex_male）
            if sex == '女性':
                feature_values['sex_female'] = 1
            else:  # 男性
                feature_values['sex_male'] = 1

            # 4. 赋值吸烟状态独热编码（smoker_yes, smoker_no）
            if smoke == '是':
                feature_values['smoker_yes'] = 1
            else:  # 否
                feature_values['smoker_no'] = 1

            # 5. 赋值区域独热编码（region_东南部, region_西南部, region_东北部, region_西北部）
            feature_values[f'region_{region}'] = 1  # 直接拼接，与训练时列名一致

            # 6. 按特征名顺序提取值（顺序不能乱）
            format_data = [feature_values[name] for name in feature_names]

            # ========== 加载模型并预测 ==========
            try:
                with open('rfr_model.pkl', 'rb') as f:
                    rfr_model = pickle.load(f)
            except FileNotFoundError:
                st.error("模型文件rfr_model.pkl未找到，请先运行train_model.py生成！")
                return
            except Exception as e:
                st.error(f"模型加载失败：{str(e)}")
                return

            # 转换为DataFrame（列名与训练时一致）
            try:
                format_data_df = pd.DataFrame(
                    data=[format_data],
                    columns=feature_names
                )
            except Exception as e:
                st.error(f"特征数据转换失败：{str(e)}")
                st.write("当前特征名：", feature_names)
                st.write("当前特征值：", format_data)
                return

            # 预测并展示结果
            try:
                predict_result = rfr_model.predict(format_data_df)[0]
                st.success(f'根据您输入的数据，预测该客户的医疗费用是：{round(predict_result, 2)} 元')
                st.write("技术支持：email：support@example.com")
            except Exception as e:
                st.error(f"预测失败：{str(e)}")
                st.write("传入的特征数据：", format_data_df)  # 调试用：显示特征数据

# 设置页面的标题、图标和布局
st.set_page_config(
    page_title="医疗费用预测",
    page_icon="🏥",
    layout="wide"
)

# 在左侧添加侧边栏并设置单选按钮
nav = st.sidebar.radio("导航", ["简介", "预测医疗费用"])

# 根据选择的结果，展示不同的页面
if nav == "简介":
    introduce_page()
else:
    predict_page()
