# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import pandas as pd
from math import sqrt
import numpy as np

theta_frame = pd.DataFrame({
    'theta': [-34.664, 9.101]
})
alpha = .02

df = pd.read_csv('X_y_data_LinReg_XV.csv')

for each in range(5000):
    dummy = pd.DataFrame([np.ones(df['X'].size), df['X']])
    df['h(x)'] = theta_frame.T.dot(dummy).T
    df['partial_cost'] = (df['h(x)']-df['y'])**2
    df['partial_term_cost_wrt_theta0'] = df['h(x)']-df['y']
    df['partial_term_cost_wrt_theta1'] = df['partial_term_cost_wrt_theta0'] * df['X']

    cost_function = sum(df['partial_cost'])/(2*df['partial_cost'].size)
    partial_cost_wrt_theta0 = sum(df['partial_term_cost_wrt_theta0'])/(df['partial_term_cost_wrt_theta0'].size)
    partial_cost_wrt_theta1 = sum(df['partial_term_cost_wrt_theta1'])/(df['partial_term_cost_wrt_theta1'].size)
    # print(partial_cost_wrt_theta0, partial_cost_wrt_theta1, cost_function)
    diff_cost_operatator = pd.DataFrame({
        'dJ_wrt_theta': [partial_cost_wrt_theta0, partial_cost_wrt_theta1]
    })
    theta_frame['theta'] = theta_frame['theta'] - alpha*diff_cost_operatator['dJ_wrt_theta']

    # print(df.head())
    print(theta_frame['theta'].values, cost_function)


