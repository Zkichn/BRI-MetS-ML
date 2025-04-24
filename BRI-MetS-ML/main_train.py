import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, roc_curve, auc
import joblib

# 设置matplotlib图形的中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
data_train = pd.read_excel("./data/D1训练集和测试集.xlsx")

# 特征选择
cols = ['XB0000', 'NL0000', '腰围', '身高', 'BRI（身体圆润指数）']
cols = [col.strip() for col in cols]

# 定义特征和目标变量
x = data_train[cols].values
y = data_train['代谢综合征'].values

# 标准化处理
ss_X = preprocessing.StandardScaler()
x_scaled = ss_X.fit_transform(x)

# 定义模型字典
models = {
    'XGB': xgb.XGBClassifier(max_depth=7, learning_rate=0.01, n_estimators=200, objective='binary:logistic',
                             booster='gbtree', scale_pos_weight=10, random_state=0),
    'Logistic Regression': LogisticRegression(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'ANN': MLPClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'GBDT': GradientBoostingClassifier(random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
}

# 十倍交叉验证
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 计算并绘制每个模型的性能
for model_name, model in models.items():
    accuracy_list, precision_list, recall_list, f1_list, roc_auc_list = [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(x_scaled, y)):
        x_train, x_test = x_scaled[train_idx], x_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]

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

    # 十折交叉验证结果输出
    print(f'{model_name}模型的十倍交叉验证结果：')
    print(f'十倍交叉验证准确率: {np.mean(accuracy_list):.2f} ± {np.std(accuracy_list):.2f}')
    print(f'十倍交叉验证精确率: {np.mean(precision_list):.2f} ± {np.std(precision_list):.2f}')
    print(f'十倍交叉验证召回率: {np.mean(recall_list):.2f} ± {np.std(recall_list):.2f}')
    print(f'十倍交叉验证F1分数: {np.mean(f1_list):.2f} ± {np.std(f1_list):.2f}')
    print(f'十倍交叉验证AUC分数: {np.mean(roc_auc_list):.2f} ± {np.std(roc_auc_list):.2f}\n')

    # 使用最佳模型进行最终评估
    best_model = model
    best_y_pred = y_pred
    best_y_test = y_test
    best_y_pred_proba = y_pred_proba
    accuracy_final = metrics.accuracy_score(best_y_test, best_y_pred)
    precision_final = metrics.precision_score(best_y_test, best_y_pred)
    recall_final = metrics.recall_score(best_y_test, best_y_pred)
    f1_final = metrics.f1_score(best_y_test, best_y_pred)
    roc_auc_final = metrics.roc_auc_score(best_y_test, best_y_pred_proba)

    print(f'{model_name}模型的最佳交叉验证折叠结果：')
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
    plt.title(f'{model_name} - Confusion Matrix')
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
    plt.title(f'{model_name} - Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # 绘制AUC曲线（使用最佳模型）
    plt.figure(figsize=(8, 6))
    plt.fill_between(fpr, tpr, color='orange', alpha=0.2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_value:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - Area Under Curve (AUC)')
    plt.legend(loc='lower right')
    plt.show()

    # 保存ROC曲线数据到Excel文件
    fpr_df = pd.DataFrame(fpr, columns=['FPR'])
    tpr_df = pd.DataFrame(tpr, columns=['TPR'])
    roc_auc_value_df = pd.DataFrame({'AUC': [roc_auc_value] * len(fpr)})  # 创建一个与fpr长度相同的AUC列
    model_name_df = pd.DataFrame({'Model': [model_name] * len(fpr)})  # 创建一个与fpr长度相同的Model列

    # 合并所有DataFrame
    roc_df = pd.concat([fpr_df, tpr_df, roc_auc_value_df, model_name_df], axis=1)

    # 保存到Excel文件
    roc_df.to_excel(f'{model_name}_roc_data.xlsx', index=False)

    # 保存最优模型
    joblib.dump(best_model, f'best_{model_name}_model.pkl')
    print(f"最优{model_name}模型已保存为 'best_{model_name}_model.pkl'\n")

# 保存标准化器（Scaler）
joblib.dump(ss_X, 'scaler_X.pkl')