import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def pos_to_signal(df):
    """
    处理拐点数据，计算与上日的差值，若差值大于0，则认为上升趋势;若差值小于0，则认为下降趋势
    :param df:
    :return:
    """
    # 计算与上日的差值，若差值大于0，则认为上升趋势;若差值小于0，则认为下降趋势
    df['trend'] = df['close'] - df['close'].shift(1)
    df.loc[df['trend'] > 0, 'pos'] = 1
    df.loc[df['trend'] < 0, 'pos'] = -1

    # 若trend=0则认为价格没有变化，为上一段趋势的延续
    df['pos'] = df['pos'].ffill()

    # 计算拐点数据
    df['turnover'] = df['pos'] - df['pos'].shift(-1)
    trend_spot = df.loc[(df['turnover'] == 2) | (df['turnover'] == -2)]

    return df, trend_spot


def basic_process_oneside(df):
    """
    单边数据处理
    :param df: 原始数据结构
    :return: 返回开盘，收盘，交易量数据，以及交易拐点数据
    """
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
    df['datetime'] = df['date'] + ' ' + df['time']
    df['datetime'] = pd.to_datetime(df['datetime'])

    df = df[['datetime', 'open', 'close', 'tickvol']]

    # 计算总体的成交量
    df['vol_cumsum'] = df['tickvol'].cumsum()

    # 计算拐点数据
    df, trend_spot = pos_to_signal(df)
    return df, trend_spot

class TrendAnalyst:

    def __init__(self, time, k1_list, k2_list, direction):
        # time_unit是计算持续时间的参数,换算为分钟
        # for example: 如果是输入15分钟，则写time_unit = 15
        self.time_unit = time

        self.k1_list = k1_list
        self.k2_list = k2_list
        self.direction = direction

    def cal_break_trend(self,
                        trend_spot: pd.DataFrame,
                        k1,
                        k2):

        trend_spot['trend_1'] = np.nan
        trend_spot['trend_2'] = np.nan

        if self.direction == 1:
            # 上升趋势
            trend_spot.loc[trend_spot['spread'] > k1, 'trend_1'] = 1
            trend_spot['diff_spread'] = trend_spot['spread'].abs() - trend_spot['spread'].abs().shift(-1)
            trend_spot.loc[(trend_spot['diff_spread'] < 0) & (trend_spot['turnover'] == -2), 'trend_2'] = -1

        if self.direction == -1:
            # 下降趋势
            trend_spot.loc[trend_spot['spread'] < k2, 'trend_1'] = 1
            trend_spot['diff_spread'] = trend_spot['spread'].abs() - trend_spot['spread'].abs().shift(-1)
            trend_spot.loc[(trend_spot['diff_spread'] < 0) & (trend_spot['turnover'] == 2), 'trend_2'] = -1

        trend_spot['trend_2'] = trend_spot['trend_2'].fillna(0)
        trend_spot['trend_2'] = trend_spot['trend_2'] - trend_spot['trend_2'].shift(1)
        trend_spot['trend_1'] = trend_spot['trend_1'].fillna(0)
        trend_spot['final_trend'] = trend_spot['trend_1'] + trend_spot['trend_2']

        trend_spot['final_up_trend'] = np.nan
        trend_spot['trend_spread'] = np.nan
        for i in range(0, len(trend_spot) - 1):
            if (trend_spot['final_trend'][i] == 2) | (trend_spot['final_trend'][i] == 1):
                if (trend_spot['final_trend'][i - 1] == -1) & (trend_spot['final_trend'][i - 2] == 2):
                    trend_spot['final_up_trend'][i] = 3
                    trend_spot['final_up_trend'][i - 1] = 2
                    trend_spot['final_up_trend'][i - 2] = 1
                if trend_spot['final_up_trend'][i] == 3:
                    trend_spot['trend_spread'][i] = trend_spot['spread'][i] + trend_spot['spread'][i - 1] + \
                                                    trend_spot['spread'][i - 2]
        trend_record = trend_spot.loc[~trend_spot['final_up_trend'].isna()]
        return trend_record

    def cal_break_reverse_stats(self,
                                trend_spot: pd.DataFrame,
                                k1,
                                k2):
        """
        冲高回落，即价差大于K1后，回落的spread的值
        触底反弹，即价差小于K2后，反弹的spread的值

        Parameters
        ----------
        trend_spot : pd.DataFrame
            DESCRIPTION: 节点数据

        Returns
        -------
        trend_spot : pd.DataFrame
            DESCRIPTION： 经过处理的节点数据
        filter_spot : pd.DataFrame
            DESCRIPTION: 只带有冲高回落以及触底反弹相关数据
        stats_break : pd.DataFrame
            DESCRIPTION： 两者统计数据

        """

        trend_spot['choosen_up'] = np.nan
        trend_spot['choosen_down'] = np.nan
        trend_spot.loc[trend_spot['spread'] > k1, 'choosen_up'] = 1
        trend_spot.loc[trend_spot['spread'] < k2, 'choosen_down'] = -1

        trend_spot['choosen_up'] = trend_spot['choosen_up'].fillna(0)
        trend_spot['choosen_up'] = trend_spot['choosen_up'] - trend_spot['choosen_up'].shift(1)

        trend_spot['choosen_down'] = trend_spot['choosen_down'].fillna(0)
        trend_spot['choosen_down'] = trend_spot['choosen_down'] - trend_spot['choosen_down'].shift(1)

        filter_spot = trend_spot.loc[(trend_spot['choosen_up'] != 0) | (trend_spot['choosen_down'] != 0)]
        filter_spot.dropna(axis=0, inplace=True)
        data_break_up = pd.DataFrame(filter_spot.loc[filter_spot['choosen_up'] == -1]['spread'].abs().describe())
        data_break_down = pd.DataFrame(filter_spot.loc[filter_spot['choosen_down'] == 1]['spread'].abs().describe())

        stats_break = pd.concat([data_break_up, data_break_down], axis=1)
        stats_break.columns = ['冲高%s回落' % k1, '冲低%s反弹' % k2]

        return filter_spot, stats_break

    def cal_stats_all(self,
                      trend_spot: pd.DataFrame,
                      column: list):
        """
        对传入的columns中的名称进行统计分析

        Parameters
        ----------
        trend_spot : pd.DataFrame
            DESCRIPTION. 节点数据
        column : list
            DESCRIPTION. 传入列名list

        Returns
        -------
        stats_all : pd.DataFrame
            DESCRIPTION. 整合所有统计数据

        """

        stats_all = pd.DataFrame()
        # 对上面list中进行循环计算
        for col in column:
            overall_trend = pd.DataFrame(trend_spot[col].abs().describe(percentiles=[.5, .75, .90, .95, .97]))
            down_trend = pd.DataFrame(
                trend_spot.loc[trend_spot['turnover'] == -2][col].abs().describe(percentiles=[.5, .75, .90, .95, .97]))
            up_trend = pd.DataFrame(
                trend_spot.loc[trend_spot['turnover'] == 2][col].describe(percentiles=[.5, .75, .90, .95, .97]))
            data = pd.concat([overall_trend, up_trend, down_trend], axis=1)
            data.columns = ['overall', 'up_trend', 'down_trend']
            data['name'] = col
            data = data.round(decimals=2)
            stats_all = pd.concat([stats_all, data], axis=1)
            print(data)
        return stats_all

    def cal_groupby_timelen(self,
                            trend_spot: pd.DataFrame):
        """
        按持续时间进行groupby 进行统计分析
        Parameters
        ----------
        trend_spot : pd.DataFrame
            DESCRIPTION. 节点数据

        Returns
        -------
        data_concat_all : pd.DataFrame
            DESCRIPTION. 输出所有数据的统计值
        data_concat : pd.DataFrame
            DESCRIPTION. 分别输出 上升趋势与下降趋势的统计值

        """

        # 对time_len进行分类 groupby 查看不同时间段内的各种值的变化
        data_all = pd.DataFrame(
            trend_spot.abs().groupby('time_len')[['vol_sum', 'vol_sum_pertime', 'spread', 'change_rate']].mean())
        data_count_all = pd.DataFrame(trend_spot.groupby('time_len').time_len.count())
        data_concat_all = pd.concat([data_all, data_count_all], axis=1)
        data_concat_all.columns = ['vol_sum', 'vol_sum_per_time', 'spread', 'change_rate', 'count_all']

        # 上升下降趋势分类统计
        # 上升趋势
        data_up = pd.DataFrame(trend_spot.loc[trend_spot['turnover'] == 2].abs().groupby('time_len')[[
            'vol_sum', 'vol_sum_pertime', 'spread', 'change_rate']].mean())
        data_count_up = pd.DataFrame(trend_spot.loc[trend_spot['turnover'] == 2].groupby('time_len').time_len.count())

        # 下降趋势
        data_down = pd.DataFrame(trend_spot.loc[trend_spot['turnover'] == -2].abs().groupby('time_len')[[
            'vol_sum', 'vol_sum_pertime', 'spread', 'change_rate']].mean())
        data_count_down = pd.DataFrame(
            trend_spot.loc[trend_spot['turnover'] == -2].groupby('time_len').time_len.count())

        # 数据合并
        data_concat = pd.concat([data_up, data_count_up, data_down, data_count_down], axis=1)
        data_concat.columns = ['vol_sum', 'vol_sum_per_time', 'spread', 'change_rate', 'count_up',
                               'vol_sum', 'vol_sum_per_time', 'spread', 'change_rate', 'count_down']
        return data_concat_all, data_concat

    def get_stats_data(self,
                       trend_spot: pd.DataFrame,
                       k1,
                       k2):
        """
        处理数据以及调用各种统计函数

        Parameters
        ----------
        trend_spot : TYPE
            DESCRIPTION.拐点数据

        Returns
        -------
        stats_all : pd.DataFrame
            DESCRIPTION. 基本统计数据
        data_concat_all : pd.DataFrame
            DESCRIPTION. groupby_timelen统计分析
        data_concat : pd.DataFrame
            DESCRIPTION. groupby_timelen 上升与下降统计分析
        stats_break : pd.DataFrame
            DESCRIPTION. 突破k值后回落的统计值

        """

        # 获取交易量变化
        trend_spot['vol_sum'] = trend_spot['vol_cumsum'] - trend_spot['vol_cumsum'].shift(1)

        # 获取价差差距
        trend_spot['spread'] = trend_spot['close'] - trend_spot['close'].shift(1)

        # 获取时间长度差距
        # 思路是trend_spot index是从df上截取的
        # 代表的是当前两个数据节点的差值*时间单位 等于交易时间的差距
        trend_spot.reset_index(inplace=True)
        trend_spot.rename(columns={'index': 'date_index'}, inplace=True)
        trend_spot.set_index('datetime', inplace=True, drop=True)
        trend_spot['time_len'] = (trend_spot['date_index'] - trend_spot['date_index'].shift(1)) * self.time_unit

        # 完整一段趋势交易量交易量/时间跨度
        trend_spot['vol_sum_pertime'] = trend_spot['vol_sum'] / trend_spot['time_len']
        # 获取变化率
        trend_spot['change_rate'] = abs(trend_spot['spread']) / trend_spot['time_len']
        trend_spot.dropna(axis=0, inplace=True)

        stats_break_up = pd.DataFrame()
        stats_break_down = pd.DataFrame()
        # 对K值进行遍历
        for i, j in zip(self.k1_list, self.k2_list):
            print(i, j)
            _, stats_break_loop = self.cal_break_reverse_stats(trend_spot, i, j)
            mean_break = pd.DataFrame(stats_break_loop.iloc[5])
            mean_break_up = pd.DataFrame(mean_break.iloc[0])
            mean_break_down = pd.DataFrame(mean_break.iloc[1])
            stats_break_up = pd.concat([stats_break_up, mean_break_up], axis=1)
            stats_break_down = pd.concat([stats_break_down, mean_break_down], axis=1)

        # 画图
        stats_break_down = stats_break_down.fillna(0)
        stats_break_up = stats_break_up.fillna(0)
        draw_fun(stats_break_up, stats_break_down)

        # 筛选大于k值
        filter_spot, stats_break = self.cal_break_reverse_stats(trend_spot, k1, k2)

        # 是趋势

        trend_record = self.cal_break_trend(trend_spot, k1, k2)
        # 统计数据
        # 统计 时间长度,价格差距,持仓量加和
        column = ['time_len', 'spread', 'vol_sum', 'vol_sum_pertime', 'change_rate']
        stats_all = self.cal_stats_all(trend_spot, column)

        # 按时间进行groupby分类
        data_concat_all, data_concat = self.cal_groupby_timelen(trend_spot)

        return stats_all, data_concat_all, data_concat, filter_spot, stats_break, stats_break_up, stats_break_down, trend_record

def draw_fun(stats_break_up, stats_break_down):
    plt.subplots(2, 1, figsize=(20, 15))

    ax1 = plt.subplot(211)
    ax1.bar(x=stats_break_up.columns, height=stats_break_up.iloc[0])
    ax1.set_title('冲高回落')

    ax2 = plt.subplot(212)
    ax2.bar(x=stats_break_down.columns, height=stats_break_down.iloc[0])
    ax2.set_title('触底反弹')

    plt.show()
