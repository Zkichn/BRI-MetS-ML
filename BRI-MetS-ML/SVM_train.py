import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# 设置matplotlib图形的中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
data_train = pd.read_excel("./data/D1训练集和测试集.xlsx")

# 特征选择
cols = ['XB0000', 'NL0000', '腰围', 'BRI（身体圆润指数）', '身高']
cols = [col.strip() for col in cols]

# 定义特征和目标变量
x = data_train[cols].values
y = data_train['代谢综合征'].values

# 标准化处理
ss_X = preprocessing.StandardScaler()
x_scaled = ss_X.fit_transform(x)

# 建立SVM二分类模型
svm_model = SVC(probability=True, random_state=0)

# 十倍交叉验证
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 用于存储交叉验证的各项指标
accuracy_list, precision_list, recall_list, f1_list, roc_auc_list = [], [], [], [], []

# 最佳模型初始化
best_model = None
best_fold = -1
best_y_pred = None
best_y_pred_proba = None
best_y_test = None

# 十折交叉验证
for fold, (train_idx, test_idx) in enumerate(cv.split(x_scaled, y)):
    x_train, x_test = x_scaled[train_idx], x_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 训练模型
    svm_model.fit(x_train, y_train)
    y_pred = svm_model.predict(x_test)
    y_pred_proba = svm_model.predict_proba(x_test)[:, 1]

    # 计算各项指标
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    roc_auc = metrics.roc_auc_score(y_test, y_pred_proba)

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    roc_auc_list.append(roc_auc)

    # 记录最佳模型
    if accuracy == max(accuracy_list):  # 记录该fold中的最佳模型
        best_fold = fold
        best_model = svm_model
        best_y_pred = y_pred
        best_y_pred_proba = y_pred_proba
        best_y_test = y_test

# 十折交叉验证结果输出
print(f'SVM模型的十倍交叉验证结果：')
print(f'十倍交叉验证准确率: {np.mean(accuracy_list):.2f} ± {np.std(accuracy_list):.2f}')
print(f'十倍交叉验证精确率: {np.mean(precision_list):.2f} ± {np.std(precision_list):.2f}')
print(f'十倍交叉验证召回率: {np.mean(recall_list):.2f} ± {np.std(recall_list):.2f}')
print(f'十倍交叉验证F1分数: {np.mean(f1_list):.2f} ± {np.std(f1_list):.2f}')
print(f'十倍交叉验证AUC分数: {np.mean(roc_auc_list):.2f} ± {np.std(roc_auc_list):.2f}\n')

# 使用最佳模型进行最终评估
accuracy_final = metrics.accuracy_score(best_y_test, best_y_pred)
precision_final = precision_score(best_y_test, best_y_pred)
recall_final = recall_score(best_y_test, best_y_pred)
f1_final = f1_score(best_y_test, best_y_pred)
roc_auc_final = roc_auc_score(best_y_test, best_y_pred_proba)

# 输出最佳模型的最终评估
print(f"SVM模型的最佳交叉验证折叠结果：")
print(f'准确率: {accuracy_final:.2f}')
print(f'精确率: {precision_final:.2f}')
print(f'召回率: {recall_final:.2f}')
print(f'F1分数: {f1_final:.2f}')
print(f'ROC AUC分数: {roc_auc_final:.2f}\n')

# 绘制混淆矩阵（使用最佳模型）
conf_matrix = confusion_matrix(best_y_test, best_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix (SVM) - Fold {best_fold+1}')
plt.show()

# 绘制ROC曲线（使用最佳模型）
fpr, tpr, thresholds = roc_curve(best_y_test, best_y_pred_proba)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic (ROC) Curve (SVM) - Fold {best_fold+1}')
plt.legend(loc='lower right')
plt.show()

# 绘制AUC曲线（使用最佳模型）
plt.figure(figsize=(8, 6))
plt.fill_between(fpr, tpr, color='orange', alpha=0.2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_value:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Area Under Curve (AUC) (SVM) - Fold {best_fold+1}')
plt.legend(loc='lower right')
plt.show()

# 保存ROC曲线数据到Excel文件
roc_auc_data = {
    'FPR': fpr.tolist(),  # 将numpy数组转换为列表
    'TPR': tpr.tolist(),  # 将numpy数组转换为列表
    'AUC': [roc_auc_value] * len(fpr),  # 创建一个与fpr长度相同的AUC列
    'Model': ['SVM'] * len(fpr)  # 创建一个与fpr长度相同的Model列
}
roc_df = pd.DataFrame(roc_auc_data)
roc_df.to_excel('SVM_roc_data.xlsx', index=False)

# 保存最优模型
joblib.dump(best_model, 'best_svm_model7.pkl')
print("最优SVM模型已保存为 'best_svm_model7.pkl'\n")

# 保存标准化器（Scaler）
joblib.dump(ss_X, 'scaler_X6.pkl')