# 导入必须的包
import os
import time
import warnings
import pandas as pd
from time import time, sleep
import xgboost as xgb
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


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


class PredictResult:
    def __init__(self, filename):
        self.filename = filename
        self.raw_name = []  # 存放原始数据
        self.file_list = []  # 获取文件名字
        self.time_list = []  # 获取比赛的时间
        self.play_statistics = []  # 创造处理后数据名存放处
        self.X_all = None
        self.y_all = None

    def read_file(self):
        for root, dirs, files in os.walk(self.filename):
            files.sort()
            for i, file in enumerate(files):
                if os.path.splitext(file)[1] == '.csv':
                    self.file_list.append(file)
                    self.raw_name.append('raw_data_' + str(i + 1))

        self.time_list = [self.file_list[i][0:4] for i in range(len(self.file_list))]
        # print(self.time_list)

        for i in range(len(self.raw_name)):
            self.raw_name[i] = pd.read_csv(self.filename + self.file_list[i], error_bad_lines=False)
            # print('第%2s个文件是%s,数据大小为%s' % (i + 1, self.file_list[i], self.raw_name[i].shape))

    def check_data(self):
        self.read_file()
        # 获取比赛城市信息
        # print(self.raw_name[0]['HomeTeam'].unique())
        # 获取列表头最大的列数，然后获取器参数
        shape_list = [self.raw_name[i].shape[1] for i in range(len(self.raw_name))]
        for i in range(len(self.raw_name)):
            if self.raw_name[i].shape[1] == max(shape_list):
                # print('%s年数据是有最大列数:%s,列元素表头：\n %s' % (self.time_list[i], max(shape_list), self.raw_name[i].columns))
                pass

        # 将挑选的信息放在一个新的列表中, 主队，客队，主队进球，客队进球， 比赛结果
        columns_req = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']

        for i in range(len(self.raw_name)):
            self.play_statistics.append('playing_statistics_' + str(i + 1))
            self.play_statistics[i] = self.raw_name[i][columns_req]
            # print(self.time_list[i], 'playing_statistics[' + str(i) + ']', self.play_statistics[i].shape)

    def get_goal_diff(self, playing_stat):
        # 创建一个字典，每个 team 的 name 作为 key
        team = {}
        for i in playing_stat.groupby('HomeTeam').mean().T.columns:
            team[i] = []
        # 对于每一场比赛
        for i in range(len(playing_stat)):
            # 主场队伍的进球数
            HTGS = playing_stat.iloc[i]['FTHG']
            # 客场队伍的进球数
            ATGS = playing_stat.iloc[i]['FTAG']

            # 主场队伍的净胜球数
            team[playing_stat.iloc[i].HomeTeam].append(HTGS - ATGS)
            # 客场队伍的净胜球数
            team[playing_stat.iloc[i].AwayTeam].append(ATGS - HTGS)

        # 行是 team 列是 MW
        goal_difference = pd.DataFrame(data=team, index=[i for i in range(1, 39)]).T
        goal_difference[0] = 0
        # 累加的净胜球数
        for i in range(2, 39):
            goal_difference[i] = goal_difference[i] + goal_difference[i - 1]
        return goal_difference

    def get_gss(self, playing_stat):
        # 统计净胜球数
        GD = self.get_goal_diff(playing_stat)
        j = 0
        #  主客场的净胜球数
        HTGDA = []
        ATGDA = []
        # 全年一共380场比赛
        for i in range(380):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam
            HTGDA.append(GD.loc[ht][j])
            ATGDA.append(GD.loc[at][j])
            if ((i + 1) % 10) == 0:
                j = j + 1
        # 把每个队的 HTGDA ATGDA 信息补充到 dataframe 中
        playing_stat.loc[:, 'HTGDA'] = HTGDA
        playing_stat.loc[:, 'ATGDA'] = ATGDA
        return playing_stat

    def show_gss(self):
        self.check_data()
        for i in range(len(self.play_statistics)):
            self.play_statistics[i] = self.get_gss(self.play_statistics[i])
        # print(self.play_statistics[2].tail())

    """
    统计每个球队的得分
    """

    # 把比赛结果转换为得分，赢得三分，平局得一分，输不得分
    def get_scores(self, result):
        if result == 'W':
            return 3
        elif result == 'D':
            return 1
        else:
            return 0

    def get_cuml_scores(self, match):
        match_scores = match.applymap(self.get_scores)
        for i in range(2, 39):
            match_scores[i] = match_scores[i] + match_scores[i - 1]
        match_scores.insert(column=0, loc=0, value=[0 * i for i in range(20)])
        return match_scores

    def get_match(self, playing_stat):
        # 创建一个字典，每个 team 的 name 作为 key
        teams = {}
        for i in playing_stat.groupby('HomeTeam').mean().T.columns:
            teams[i] = []
        # 把比赛结果分别记录在主场队伍和客场队伍中
        for i in range(len(playing_stat)):
            if playing_stat.iloc[i].FTR == 'H':
                # 主场 赢，则主场记为赢，客场记为输
                teams[playing_stat.iloc[i].HomeTeam].append('W')
                teams[playing_stat.iloc[i].AwayTeam].append('L')
            elif playing_stat.iloc[i].FTR == 'A':
                # 客场 赢，则主场记为输，客场记为赢
                teams[playing_stat.iloc[i].AwayTeam].append('W')
                teams[playing_stat.iloc[i].HomeTeam].append('L')
            else:
                # 平局
                teams[playing_stat.iloc[i].AwayTeam].append('D')
                teams[playing_stat.iloc[i].HomeTeam].append('D')
        return pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T

    def only_hw(self, string):
        if string == 'H':
            return 'H'
        else:
            return 'NH'

    def get_agg_scores(self, playing_stat):
        match = self.get_match(playing_stat)
        cum_scores = self.get_cuml_scores(match)
        HTPA = []
        ATPA = []
        j = 0
        for i in range(380):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam
            HTPA.append(cum_scores.loc[ht][j])
            ATPA.append(cum_scores.loc[at][j])

            if ((i + 1) % 10) == 0:
                j = j + 1
        # 主场累计得分
        playing_stat.loc[:, 'HTPA'] = HTPA
        # 客场累计得分
        playing_stat.loc[:, 'ATPA'] = ATPA
        playing_stat['FTR'] = playing_stat.FTR.apply(self.only_hw)
        return playing_stat

    def get_total_scores(self):
        self.show_gss()
        for i in range(len(self.play_statistics)):
            self.play_statistics[i] = self.get_agg_scores(self.play_statistics[i])
        # print(self.play_statistics[2].tail())
        # print(222222222)

    """
    添加球队近三场的比赛信息
    """

    def get_form(self, playing_stat, num):
        form = self.get_match(playing_stat)
        form_final = form.copy()
        for i in range(num, 39):
            form_final[i] = ''
            j = 0
            while j < num:
                form_final[i] += form[i - j]
                j += 1
        return form_final

    def add_form(self, playing_stat, num):
        form = self.get_form(playing_stat, num)
        # M 代表 unknown， 因为没有那么多历史
        h = ['M' for i in range(num * 10)]
        a = ['M' for i in range(num * 10)]
        j = num
        for i in range((num * 10), 380):
            ht = playing_stat.iloc[i].HomeTeam
            at = playing_stat.iloc[i].AwayTeam

            past = form.loc[ht][j]
            h.append(past[num - 1])

            past = form.loc[at][j]
            a.append(past[num - 1])

            if ((i + 1) % 10) == 0:
                j = j + 1
        playing_stat['HM' + str(num)] = h
        playing_stat['AM' + str(num)] = a
        return playing_stat

    def add_form_df(self, playing_statistics):
        playing_statistics = self.add_form(playing_statistics, 1)
        playing_statistics = self.add_form(playing_statistics, 2)
        playing_statistics = self.add_form(playing_statistics, 3)
        return playing_statistics

    def show_ham(self):
        self.get_total_scores()
        for i in range(len(self.play_statistics)):
            self.play_statistics[i] = self.add_form_df(self.play_statistics[i])
        # 查看构造特征后的05-06年的后5五条数据
        # print(self.play_statistics[2].tail())

    """
    加入比赛周数
    """

    def get_mw(self, playing_stat):
        j = 1
        MW = []
        for i in range(380):
            MW.append(j)
            if ((i + 1) % 10) == 0:
                j = j + 1
        playing_stat['MW'] = MW
        return playing_stat

    def set_mw(self):
        self.show_ham()
        for i in range(len(self.play_statistics)):
            self.play_statistics[i] = self.get_mw(self.play_statistics[i])
        # 查看构造特征后的05-06年的后五条数据
        print(self.play_statistics[2].tail())

    """
    删除前三周的数据
    """

    def delete_wk(self, playing_stat):
        playing_stat = playing_stat[playing_stat.MW > 18]
        return playing_stat

    def delete_some_data(self):
        self.set_mw()
        for i in range(len(self.play_statistics)):
            self.play_statistics[i] = self.delete_wk(self.play_statistics[i])
        # print(self.play_statistics[2].tail())
        # print(33333333333)

    """
    将数据合并在一起
    """

    def preprocess_features(self, X):
        '''把离散的类型特征转为哑编码特征 '''
        output = pd.DataFrame(index=X.index)
        for col, col_data in X.iteritems():
            if col_data.dtype == object:
                col_data = pd.get_dummies(col_data, prefix=col)
            output = output.join(col_data)
        return output

    def conbine_data(self):
        self.delete_some_data()
        # 将各个DataFrame表合并在一张表中
        playing_stat = pd.concat(self.play_statistics, ignore_index=True)

        # HTGDA, ATGDA ,HTPA, ATPA 除以 week 数，得到平均分
        cols = ['HTGDA', 'ATGDA']
        playing_stat.MW = playing_stat.MW.astype(float)
        for col in cols:
            playing_stat[col] = playing_stat[col] / playing_stat.MW

        # 查看构造特征后数据集的后5五条数据
        # print(playing_stat.tail())
        # 抛弃前三周的比赛
        playing_stat = playing_stat[playing_stat.MW > 18]
        playing_stat.drop(['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'MW', 'HTPA', 'ATPA'], 1, inplace=True)

        # 我们查看下此时的数据的特征
        # print(playing_stat.keys())
        # 比赛总数
        n_matches = playing_stat.shape[0]
        # 特征数
        n_features = playing_stat.shape[1] - 1
        # 主场获胜的数目
        n_homewins = len(playing_stat[playing_stat.FTR == 'H'])
        # 主场获胜的比例
        win_rate = (float(n_homewins) / (n_matches)) * 100

        # Print the results
        # print("比赛总数: {}".format(n_matches))
        # print("总特征数: {}".format(n_features))
        # print("主场胜利数: {}".format(n_homewins))
        # print("主场胜率: {:.2f}%".format(win_rate))
        # 把数据分为特征值和标签值
        self.X_all = playing_stat.drop(['FTR'], 1)
        self.y_all = playing_stat['FTR']
        # 特征值的长度
        # print(len(self.X_all))
        cols = [['HTGDA', 'ATGDA']]
        for col in cols:
            self.X_all[col] = scale(self.X_all[col])
        self.X_all = self.preprocess_features(self.X_all)
        self.X_all.to_csv('x_whole18.csv')
        self.y_all.to_csv('y_whole18.csv')
        # print("Processed feature columns ({} total features):\n{}".format(len(self.X_all.columns),
        #                                                                   list(self.X_all.columns)))
        # print("\nFeature values:")
        # print(self.X_all.head())

    def pearson(self):
        self.conbine_data()
        # 防止中文出现错误
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 制成皮尔森热图
        # 把标签映射为0和1
        y_all = self.y_all.map({'NH': 0, 'H': 1})
        # 合并特征集和标签
        train_data = pd.concat([self.X_all, y_all], axis=1)
        colormap = plt.cm.RdBu
        plt.figure(figsize=(18, 10))
        plt.title('Pearson Correlation', y=1.05, size=15)
        b = round(train_data.astype(float).corr(), 1)
        sns.heatmap(b, linewidths=0.1, vmax=1.0,
                    square=True, cmap=colormap, linecolor='white', annot=True)


name = './original_data/'
predictor = PredictResult(name)
a = time()
predictor.conbine_data()
print(time() - a)

















