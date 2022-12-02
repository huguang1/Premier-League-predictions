# 导入必须的包
import os
import time
import warnings
import pandas as pd
import numpy as np
from time import time, sleep
import xgboost as xgb
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')

"""
winner of the match (H for home team, A for away team, D for draw)
2000-2022年
"""
"""
选择哪些特征进行训练
1. HTGDA: 每赛季，到第几周时的主队平均进球数
2. ATGDA: 每赛季，到第几周时的客队平均进球数
3. HTPA: 每赛季，到第几周时的主队平均得分
4. ATPA: 每赛季，到第几周时的客队平均得分
7. 上一场主场和客场的比赛情况
8. 上上场主场和客场的比赛情况
9. 上三场主场和客场的比赛情况
"""

"""
数据分成测试集和训练集，同时
"""


def train_classifier(model, X_train, y_train):
    # 记录训练时长
    start = time()
    model.fit(X_train, y_train)
    end = time()
    print("训练时间 {:.4f} 秒".format(end - start))


def predict_labels(model, features, target):
    # 记录预测时长
    start = time()
    y_pred = model.predict(features)
    end = time()
    print("预测时间 in {:.4f} 秒".format(end - start))
    e = [i[0] for i in target.values]
    f1 = f1_score(e, y_pred, pos_label='H')
    accuracy = accuracy_score(e, y_pred)
    precision = precision_score(e, y_pred, pos_label='H')
    recall = recall_score(e, y_pred, pos_label='H')
    print(confusion_matrix(target, y_pred))
    print(classification_report(target, y_pred))
    return f1, accuracy, precision, recall


def train_predict(model, X_train, y_train, X_test, y_test):
    print("训练 {} 模型，样本数量 {}。".format(model.__class__.__name__, len(X_train)))
    # 训练模型
    train_classifier(model, X_train, y_train)
    # 在测试集上评估模型
    f1, accuracy, precision, recall = predict_labels(model, X_train, y_train)
    print("训练集上的 F1 分数和准确率为: {:.4f} , {:.4f}, {:.4f} , {:.4f}。".format(f1, accuracy, precision, recall))
    f1, accuracy, precision, recall = predict_labels(model, X_test, y_test)
    print("测试集上的 F1 分数和准确率为: {:.4f} , {:.4f}, {:.4f} , {:.4f}。".format(f1, accuracy, precision, recall))


def machine_learning():
    x_all = pd.read_csv('x_whole3.csv', error_bad_lines=False).iloc[:, 1:]
    y_all = pd.read_csv('y_whole3.csv', error_bad_lines=False).iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=2,
                                                        stratify=y_all)
    # dummy = DummyClassifier(strategy="most_frequent").fit(x_all, y_all)
    # ydummy = dummy.predict(x_all)
    # print(confusion_matrix(y_all, ydummy))
    # print(classification_report(y_all, ydummy))
    # 分别建立三个模型
    model_A = LogisticRegression(random_state=42)
    model_B = SVC(random_state=42, kernel='rbf', gamma='auto')
    model_C = xgb.XGBClassifier(seed=42)
    train_predict(model_A, X_train, y_train, X_test, y_test)
    print('')
    train_predict(model_B, X_train, y_train, X_test, y_test)
    print('')
    train_predict(model_C, X_train, y_train, X_test, y_test)


def logitstic_parameter():
    x_all = pd.read_csv('x_whole12.csv', error_bad_lines=False).iloc[:, 1:]
    y_all = pd.read_csv('y_whole12.csv', error_bad_lines=False).iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=2, stratify=y_all)
    penaltys = ['l1', 'l2']
    Cs = [0.0000001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    # 调优的参数集合，搜索网格为2x5，在网格上的交叉点进行搜索
    tuned_parameters = dict(penalty=penaltys, C=Cs)

    lr_penalty = LogisticRegression(solver='liblinear')
    grid = GridSearchCV(lr_penalty, tuned_parameters, cv=2, scoring='neg_log_loss', n_jobs=4)
    grid.fit(X_train, y_train)
    y_predict = grid.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict, pos_label='H')
    print("f1_score={}".format(f1))
    print("accuracy={}".format(accuracy))
    print("params={} scores={}".format(grid.best_params_, grid.best_score_))


def logitstic_f1score_plot():
    x_all = pd.read_csv('x_whole3.csv', error_bad_lines=False).iloc[:, 1:]
    y_all = pd.read_csv('y_whole3.csv', error_bad_lines=False).iloc[:, 1]
    y_all_true = []
    for i in y_all.iloc[:]:
        if i == "H":
            y_all_true.append(1)
        else:
            y_all_true.append(-1)
    c_range = [0.01, 0.1, 1, 10, 100, 1000]
    mean_f1, std_f1 = [], []
    mean_acc, std_acc = [], []
    for c in c_range:
        model = LogisticRegression(penalty='l1', C=c, solver='liblinear',)
        acc = cross_val_score(model, x_all, y_all_true, cv=5, scoring='accuracy')
        f1 = cross_val_score(model, x_all, y_all_true, cv=5, scoring='f1')
        mean_f1.append(np.array(f1).mean())
        std_f1.append(np.array(f1).std())
        mean_acc.append(np.array(acc).mean())
        std_acc.append(np.array(acc).std())
    plt.axes(xscale="log")
    print(mean_f1)
    print(mean_acc)
    plt.errorbar(c_range, mean_f1, yerr=std_f1, c='r', label='f1 score')
    plt.errorbar(c_range, mean_acc, yerr=std_acc, c='g', label='accuracy')
    plt.xlabel('C')
    plt.ylabel('f1 score and accuracy')
    plt.title("LogisticRegression with l1")  # title
    plt.xlim((0, 1200))
    plt.legend(loc=0, bbox_to_anchor=(1, 1))
    plt.show()


def SVC_parameter():
    x_all = pd.read_csv('x_whole3.csv', error_bad_lines=False).iloc[:, 1:]
    y_all = pd.read_csv('y_whole3.csv', error_bad_lines=False).iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=2, stratify=y_all)
    # parameters = {'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    #               'C': [0.01, 0.1, 1, 10, 100, 1000, 10000]}
    # parameters = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    parameters = {'gamma': [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    # n_jobs =-1使用全部CPU并行多线程搜索
    gs = GridSearchCV(SVC(), parameters, refit=True, cv=5, verbose=1, n_jobs=-1)
    gs.fit(X_train, y_train)  # Run fit with all sets of parameters.
    y_predict = gs.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict, pos_label='H')
    print('最优参数: ', gs.best_params_)
    print('最佳性能: ', gs.best_score_)
    print("f1_score={}".format(f1))
    print("accuracy={}".format(accuracy))
    print("params={} scores={}".format(gs.best_params_, gs.best_score_))


def SVC_f1score_plot():
    x_all = pd.read_csv('x_whole3.csv', error_bad_lines=False).iloc[:, 1:]
    y_all = pd.read_csv('y_whole3.csv', error_bad_lines=False).iloc[:, 1]
    y_all_true = []
    for i in y_all.iloc[:]:
        if i == "H":
            y_all_true.append(1)
        else:
            y_all_true.append(-1)
    # c_range = [0.01, 0.1, 1, 10, 100]
    gamma = [0.001, 0.01, 0.1, 1, 10]
    mean_f1, std_f1 = [], []
    mean_acc, std_acc = [], []
    for g in gamma:
        model = SVC(C=1, gamma=g)
        f1 = cross_val_score(model, x_all, y_all_true, cv=5, scoring='f1')
        acc = cross_val_score(model, x_all, y_all, cv=5, scoring='accuracy')
        print(f1)
        print(acc)
        mean_f1.append(np.array(f1).mean())
        std_f1.append(np.array(f1).std())
        mean_acc.append(np.array(acc).mean())
        std_acc.append(np.array(acc).std())
    plt.axes(xscale="log")
    print(mean_f1)
    plt.errorbar(gamma, mean_f1, yerr=std_f1, c='r', label='f1 score')
    plt.errorbar(gamma, mean_acc, yerr=std_acc, c='g', label='accuracy')
    plt.xlabel('gamma')
    plt.ylabel('f1 score and accuracy')
    plt.title("SVC with C=1")  # title
    plt.xlim((0, 120))
    plt.legend(loc=0, bbox_to_anchor=(1, 1))
    plt.show()


def XGBClassifier_parameter():
    x_all = pd.read_csv('x_all.csv', error_bad_lines=False).iloc[:, 1:]
    y_all = pd.read_csv('y_all.csv', error_bad_lines=False).iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=2, stratify=y_all)

    parameters = {
        'max_depth': [5, 10, 20, 25],
        'n_estimators': [50, 150, 300, 500],
        'min_child_weight': [0, 5, 10, 20],
        'max_delta_step': [0, 0.6, 1, 2],
        'colsample_bytree': [0.5, 0.7, 0.9],
        'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1]
    }

    clf_C = xgb.XGBClassifier(seed=42)
    gsearch = GridSearchCV(clf_C, param_grid=parameters, scoring='accuracy', cv=3)
    gsearch.fit(X_train, y_train)
    y_predict = gsearch.predict(X_test)
    accuracy = accuracy_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict, pos_label='H')
    print('最优参数: ', gsearch.best_params_)
    print('最佳性能: ', gsearch.best_score_)
    print("f1_score={}".format(f1))
    print("accuracy={}".format(accuracy))
    print("params={} scores={}".format(gsearch.best_params_, gsearch.best_score_))


def XGBClassifier_f1score_plot():
    x_all = pd.read_csv('x_whole3.csv', error_bad_lines=False).iloc[:, 1:]
    y_all = pd.read_csv('y_whole3.csv', error_bad_lines=False).iloc[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=2, stratify=y_all)

    parameters = {
        'max_depth': [1, 2, 3, 4, 5, 10],
        'n_estimators': [50, 150, 300, 500]
    }

    # clf_C = xgb.XGBClassifier(seed=42)
    n_estimators = [0, 10, 20, 50, 200]
    mean_acc, std_acc = [], []
    mean_f1, std_f1 = [], []
    for c in n_estimators:
        model = xgb.XGBClassifier(seed=42, max_depth=2, n_estimators=c)
        f1 = cross_val_score(model, x_all, y_all, cv=5, scoring='f1')
        acc = cross_val_score(model, x_all, y_all, cv=5, scoring='accuracy')
        print(f1)
        print(acc)
        mean_acc.append(np.array(acc).mean())
        std_acc.append(np.array(acc).std())
        mean_f1.append(np.array(f1).mean())
        std_f1.append(np.array(f1).std())
    plt.axes(xscale="log")
    print(mean_acc)
    # plt.errorbar(max_depth, mean_f1, yerr=std_f1, c='r', label='f1 score')
    plt.errorbar(n_estimators, mean_acc, yerr=std_acc, c='g', label='accuracy')
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    plt.title("XGBClassifier with max_depth=2")  # title
    plt.xlim((0, 200))
    plt.legend(loc=0, bbox_to_anchor=(1, 1))
    plt.show()


def ROC_curve():
    x_all = pd.read_csv('x_whole3.csv', error_bad_lines=False).iloc[:, 1:]
    y_all = pd.read_csv('y_whole3.csv', error_bad_lines=False).iloc[:, 1]
    y_all_true = []
    for i in y_all.iloc[:]:
        if i == "H":
            y_all_true.append(1)
        else:
            y_all_true.append(-1)
    X_train, X_test, y_train, y_test = train_test_split(x_all, y_all_true, test_size=0.2, random_state=2, stratify=y_all_true)

    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    logic = LogisticRegression(penalty='l2', C=1).fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, logic.decision_function(X_test))
    plt.plot(fpr, tpr, c='b', label='LogisticRegression', linestyle='--')

    dummy = DummyClassifier(strategy="uniform").fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, dummy.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, c='y', label='baseline most_frequent', linestyle='--')

    model = SVC(C=1).fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, model.decision_function(X_test))
    plt.plot(fpr, tpr, c='r', label='SVC')

    model = xgb.XGBClassifier(seed=42, max_depth=2, n_estimators=50).fit(X_train, y_train)
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, c='g', label='XGBClassifier')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(loc='lower right')
    plt.show()


a = time()
ROC_curve()
print(time() - a)
