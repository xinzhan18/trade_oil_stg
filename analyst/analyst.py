import pandas as pd
import matplotlib.pyplot as plt
from utils.process_utils import *


target = "a"
title = 'CFD_OIL'
k1 = 80
k2 = -10
direction = 1
time_unit = 1

k1_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 190, 220]
k2_list = [-10, -20, -30, -40, -50, -60, -70, -80, -90, -100, -110, -120, -150, -190, -220]

df = pd.read_csv('../data/CL-OIL_M1_202307051835_202310161718.csv', sep='\t')

# 单边处理
spread_data, spread_spot = basic_process_oneside(df)

trend = TrendAnalyst(time_unit, k1_list, k2_list, direction)
stats_data, stats_time_all, stats_time_up_down, filter_spot, stats_break, stats_break_up, stats_break_down, trend_record = trend.get_stats_data(
    spread_spot, k1, k2)

