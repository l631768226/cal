#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
author: sophay
date: 2021/1/6
email: 1427853491@qq.com
"""
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from multiprocessing import Process, Lock, Queue
from pylab import mpl

# matplotlib 解决不能使用中文的方法
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False
plt.style.use(['dark_background'])


class DatasetPreprocessing:
    def __init__(self, rsc_path: str, dist_path: str):
        print("数据处理中...")
        # 读取签到数据中用到的特征，数据集变化可能需要修改
        self.df = pd.read_csv(rsc_path, low_memory=False, encoding='gbk',
                              usecols=['user_id', 'placeID', 'lat', 'lng', 'time_offset', 'time', 'label'])
        self.users = None
        self.dist_path = dist_path
        self.remove_infrequently_data()
        self.numerical_place_id()
        self.transform_utc_time()
        # 输出处理完成的df按照标签顺序
        self.df = self.df[['user_id', 'placeID', 'lat', 'lng', 'localtime', 'label']]
        self.df.to_csv(os.path.join(self.dist_path, 'part1_result.csv'), index=None)
        self.plot_user_checkins()

    def numerical_place_id(self):
        """将placeID数值化, useID重新排序"""
        print("将placeID数值化")
        unique_place_id = self.df['placeID'].unique()
        map_dict = dict(zip(unique_place_id, range(len(unique_place_id))))
        self.df['placeID'] = self.df['placeID'].map(map_dict)
        unique_user_id = self.df['user_id'].unique()
        map_dict = dict(zip(unique_user_id, range(len(unique_user_id))))
        self.df['user_id'] = self.df['user_id'].map(map_dict)

    def transform_utc_time(self):
        """将UTC时间加上时间偏移量得到localtime"""
        print("将UTC时间加上时间偏移量得到localtime")
        self.df['time'] = pd.to_datetime(self.df['time'])  # 这里最好提前解析时间，不然会很耗时
        self.df['time'] = self.df['time'].dt.tz_localize(None)  # 去掉时区
        # 使用numpy进行时间校正，效率更快
        self.df['localtime'] = (self.df.time.values.astype('M8[s]') +
                                self.df.time_offset.values * np.timedelta64(1, 'h')).astype('M8[s]')

    def remove_infrequently_data(self):
        """移除八个半月访问次数少于5次的数据"""
        print("移除八个半月访问次数少于5次的数据")
        features = pd.DataFrame()
        users = self.df['user_id'].unique()
        # 遍历所有用户，提取每个用户的周期，加入群体的features中
        for user in users:
            # 选中特定ID的用户，去除用户签到次数不大于5次的签到点，提取周期
            user_df = self.df[self.df['user_id'] == user]
            satisfied_data_counts = user_df['placeID'].value_counts()
            satisfied_data_index = satisfied_data_counts[satisfied_data_counts > 5].index
            satisfied_data = user_df[user_df['placeID'].isin(satisfied_data_index)]
            # 如果用户的数据不符合条件，则过滤掉该用户
            if satisfied_data is None:
                continue
            features = pd.concat([features, satisfied_data], ignore_index=True)
        self.df = features

    def plot_user_checkins(self):
        """将每个用户的信息绘制出来"""
        users = self.df['user_id'].values
        users_dict = {}
        for item in users:
            users_dict[item] = users_dict.get(item, 0) + 1
        users_dict_temp = dict(sorted(users_dict.items(), key=lambda x: x[1], reverse=True))
        self.users = list(users_dict_temp.keys())[:10]
        user = np.array(list(users_dict.keys()))
        times = np.array(list(users_dict.values()))

        plt.figure(facecolor='#084874', figsize=(10, 8), dpi=100)
        ax = plt.gca()
        ax.set_facecolor('#084874')
        ax.plot(user, times, 'w')
        ax.set_title("用户签到次数折线图")
        ax.set_xlabel("用户ID")
        ax.set_ylabel("签到次数")
        plt.savefig(os.path.join(self.dist_path, 'all_users_checkin.png'))

        def plot_single_user(this, user_id):
            df = this.df[this.df['user_id'] == user_id]
            data = df['placeID'].values
            data_dict = {}
            for each in data:
                data_dict[each] = data_dict.get(each, 0) + 1
            place_id = np.array(list(data_dict.keys()))
            freq = np.array(list(data_dict.values()))
            plt.figure(facecolor='#084874', figsize=(10, 8), dpi=100)
            ax2 = plt.gca()
            ax2.set_facecolor('#084874')
            ax2.bar(place_id, freq, color='white')
            ax2.set_title("用户%s签到地点柱状图" % user_id)
            ax2.set_xlabel("签到地点ID")
            ax2.set_ylabel("签到次数")
            plt.savefig(os.path.join(this.dist_path, '%s.png' % user_id))

        for user in self.users:
            plot_single_user(self, user)


class PeriodMining:
    def __init__(self, rsc_path, dist_path):
        print("周期模式挖掘中...")
        self.rsc_path = rsc_path
        self.dist_path = os.path.join(dist_path, 'part2_result.csv')
        if os.path.exists(self.dist_path):
            print("检测到有历史文件，正在删除...")
            os.remove(self.dist_path)
            print("删除文件%s完成!" % self.dist_path)
        self.periods = {}
        self.multiprocessing_mining()

    def appended_write_csv(self, user_df):
        # 以追加的模式输出user_df
        if not os.path.exists(self.dist_path):
            user_df.to_csv(self.dist_path, index=None)
        else:
            user_df.to_csv(self.dist_path, index=None, mode='a', header=False)

    def period_mining(self, user_df: pd.DataFrame):
        """设置用户初始时间为0将时间转化为时间序列(0,1,2,...)(小时), 得到单个用户全部活动的周期"""

        def get_time_intervals(t, base_t):
            """返回 t减去base_t的小时数"""
            diff = pd.to_datetime(t) - pd.to_datetime(base_t)
            return diff.days * 24 + diff.seconds / 3600

        x = np.array(user_df['localtime'].apply(lambda t: get_time_intervals(t, user_df['localtime'].min())))
        y = user_df['placeID'].to_numpy()
        # 遍历全部的placeID对某个单独的placeID进行周期提取
        for place_id in np.unique(y):
            y_copy = y.copy()
            # 将其他place_id设为 -1
            y_copy[y_copy != place_id] = -1
            # 控制最大频率(频率范围)，因为知道周期不会小于1小时，则频率必定落在(0, 1)中
            ls = LombScargle(x, y_copy)
            frequency, power = ls.autopower(minimum_frequency=0.0001, maximum_frequency=1)
            try:
                period = 1 / frequency[np.where(power > power.max() * 0.5)]
                # period = 1 / frequency[np.where(power == power.max())]
                self.periods[str(place_id)] = round(period.max()[0], 2)
            except (ValueError, IndexError):
                # 没有周期性的时候将周期设置为-1表示没有周期性
                self.periods[str(place_id)] = -1
        return self.periods

    def multiprocessing_mining(self):
        processes = []
        queue = Queue()
        lock = Lock()
        df = pd.read_csv(self.rsc_path, low_memory=False)
        users = df['user_id'].unique()
        for user in users:
            queue.put(user)
        start_t = time.time()
        for _ in range(8):
            p = Process(target=self.multiprocessing_task, args=(df, queue, lock))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print("耗时%.2f分钟" % ((time.time() - start_t) / 60))

    def multiprocessing_task(self, df, queue, lock):
        while not queue.empty():
            cur_user = queue.get()
            user_df = df[df['user_id'] == cur_user].copy()
            user_periods = self.period_mining(user_df)
            user_df['period'] = user_df['placeID'].apply(lambda x: user_periods.get(str(x)))
            user_df = user_df[['user_id', 'placeID', 'lat', 'lng', 'localtime', 'period', 'label']]
            lock.acquire()  # 获取锁
            self.appended_write_csv(user_df)
            lock.release()  # 释放锁
            print("剩余%s" % queue.qsize())


class FeatureExtraction:
    def __init__(self, rsc_path):
        print("特征提取中...")
        self.rsc_path = rsc_path
        self.df = pd.read_csv(self.rsc_path, low_memory=False)
        self.time_extraction()
        self.label_dealing()
        self.save()

    def time_extraction(self):
        # 处理时间，分成日，周，月
        temp_time = pd.to_datetime(self.df['localtime'])
        self.df['day_time'] = temp_time.dt.day
        self.df['week_time'] = temp_time.dt.dayofweek
        self.df['month_time'] = temp_time.dt.month
        self.df['year_time'] = temp_time.dt.year

    def label_dealing(self):
        # 将标签数值化
        self.df['activity'] = self.df['label']
        activities = self.df['label'].unique().tolist()
        activities_dict = dict(zip(activities, range(len(activities))))
        self.df['activity'] = self.df['activity'].apply(lambda x: activities_dict.get(str(x)))

    def save(self):
        # 将特征进行排序整理，标签放在最后一行
        self.df = self.df.reset_index(drop=True)
        self.df = self.df[['user_id', 'lng', 'lat', 'placeID', 'day_time', 'week_time', 'month_time', 'year_time',
                           'period', 'activity']]


class Model:
    def __init__(self, df, path, user_id):
        self.df = df
        self.user_id = user_id
        self.dist_path = os.path.join(path, 'part3_result.csv')
        self.rf_model()

    def rf_model(self):
        """使用 Random Forest 分类器进行分类，选用kernel核，输出混淆矩阵和各个评价值"""
        # 将数据集分成7:3进行训练和测试
        # self.features = self.df[['lng', 'lat', 'day_time', 'week_time', 'month_time', 'year_time', 'period']].values
        # self.labels = self.df['activity'].values
        # x_train, x_test, y_train, y_test = train_test_split(self.features, self.labels, test_size=0.3)
        train_features = self.df
        train_df = train_features[self.df['user_id'] != self.user_id]
        test_df = train_features[self.df['user_id'] == self.user_id]
        x_train = train_df[['lng', 'lat', 'day_time', 'week_time', 'month_time', 'year_time', 'period']].to_numpy()
        y_train = train_df['activity'].to_numpy()
        x_test = test_df[['lng', 'lat', 'day_time', 'week_time', 'month_time', 'year_time', 'period']].to_numpy()
        y_test = test_df['activity'].to_numpy()
        rf = RandomForestClassifier(oob_score=True, random_state=10, n_estimators=84, n_jobs=-1)
        rf.fit(x_train, y_train)

        y_test_pre = rf.predict(x_test)
        df_pred = pd.DataFrame(x_test,
                               columns=['lng', 'lat', 'day_time', 'week_time', 'month_time', 'year_time', 'period'])
        df_pred['activity_real'] = y_test
        df_pred['activity_pred'] = y_test_pre
        activities = ['Sports', 'Shopping', 'Entertainment', 'Restaurant', 'Travel', 'Work', 'Rest',
                      'Service', 'Medical', 'Art', 'Meeting', 'Education']
        activities_dict = dict(zip(map(str, range(len(activities))), activities))
        df_pred['activity_real'] = df_pred['activity_real'].apply(lambda x: activities_dict.get(str(int(x))))
        df_pred['activity_pred'] = df_pred['activity_pred'].apply(lambda x: activities_dict.get(str(int(x))))

        df_pred = df_pred[['year_time', 'month_time', 'day_time', 'week_time', 'lng', 'lat', 'period',
                           'activity_real', 'activity_pred']]
        df_pred.to_csv(self.dist_path, index=None)


def data_analysis(rsc_path1, dist_path1):
    obj = DatasetPreprocessing(rsc_path1, dist_path1)
    return obj.users


def period_mining(dist_path1, dist_path2):
    obj = PeriodMining(dist_path1, dist_path2)
    return obj.periods


def activity_semantic_recognition(dist_path2, dist_path3, user_id):
    obj = FeatureExtraction(dist_path2)
    Model(obj.df, dist_path3, user_id)
    return None


if __name__ == '__main__':
    data_analysis(r'E:\building\NYC.csv', r'E:\building\sf\result')
    period_mining(r'E:\building\sf\result\part1_result.csv', r'E:\building\sf\result')
    activity_semantic_recognition(r'E:\building\sf\result\part2_result.csv', r'E:\building\sf\result', 217)
