import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle
import time  # 新增：记录训练时间

# 1. 数据加载与预处理（优化：指定更多列的dtype，减少内存占用）
start_time = time.time()
df = pd.read_csv(
    'student_data_adjusted_rounded.csv',
    encoding='utf-8-sig',
    dtype={
        '学号': str,          # 学号保持字符串
        '性别': 'category',  # 分类列指定category类型，减少内存
        '专业': 'category'
    }
)
df.dropna(inplace=True)  # 若缺失值较多，可替换为填充（如df.fillna(method='ffill')）
print(f"数据集形状：{df.shape}（耗时{time.time()-start_time:.2f}秒）")
print("特征列：", df.columns.tolist())

# 2. 定义特征和目标变量（与CSV列名保持一致）
features = df[['性别', '专业', '每周学习时长（小时）', '上课出勤率', '期中考试分数', '作业完成率']]
target = df['期末考试分数']

# 3. 分类特征编码（无修改，若专业类别>20个，可考虑标签编码+树模型原生支持）
features_encoded = pd.get_dummies(features, drop_first=False)
print("编码后的特征列数：", len(features_encoded.columns.tolist()))

# 4. 划分训练集（5万数据的8:2拆分足够，无需交叉验证）
x_train, x_test, y_train, y_test = train_test_split(
    features_encoded, target, train_size=0.8, random_state=42, shuffle=True
)

# 5. 模型训练（优化：并行训练+适当增加树数量）
rfr = RandomForestRegressor(
    n_estimators=150,    # 5万数据可适度增加树数量，提升精度（原100）
    random_state=42,
    n_jobs=-1,           # 启用所有CPU核心，训练速度提升3-5倍
    max_depth=12,        # 限制树深度，避免过拟合（可选）
    min_samples_leaf=5   # 避免单样本叶节点，增强泛化能力
)
train_start = time.time()
rfr.fit(x_train, y_train)
print(f"模型训练完成（耗时{time.time()-train_start:.2f}秒）")

# 6. 模型评估（无修改）
y_pred = rfr.predict(x_test)
print(f"模型评估结果：")
print(f"决定系数（R²）：{r2_score(y_test, y_pred):.4f}")
print(f"平均绝对误差（MAE）：{mean_absolute_error(y_test, y_pred):.2f}分")

# 7. 保存模型（无修改，5万数据训练的模型文件约50-100MB，pickle兼容）
with open('rfr_model.pkl', 'wb') as f:
    pickle.dump(rfr, f)
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(features_encoded.columns.tolist(), f)
with open('unique_values.pkl', 'wb') as f:
    pickle.dump({
        '性别': df['性别'].unique().tolist(),
        '专业': df['专业'].unique().tolist()
    }, f)

print(f"全部流程完成（总耗时{time.time()-start_time:.2f}秒）")
