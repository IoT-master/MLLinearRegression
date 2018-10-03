import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('X_y_data_LinReg_XV.csv')
print(df.head())
print(df.describe())

#Plot X-y
plt.figure(figsize=(10,5))
plt.title("X - y Scatter plot")
plt.scatter(df['X'], df['y'], label='X - y values')
plt.legend(loc="upper left")
plt.grid(True)
plt.show()


tuple_xy = [each for each in zip(df['X'].tolist(), df['y'].tolist())]
number_of_datapoints = df.shape[0]
# theta0_old = -34.670618751
theta0_old = -34
# theta1_old = 9.10207086628
theta1_old = 9
alpha = .005
cost_function_history = [10000, 1000]
threshold = 1e-16

for each_run in range(50):
# while cost_function_history[-2] - cost_function_history[-1] > threshold:
    inner_function = 0
    inner_function_x = 0
    inner_function_sq = 0
    for each_x, each_y in tuple_xy:
        shared_function = theta0_old + theta1_old * each_x - each_y
        inner_function = inner_function + shared_function
        inner_function_x = inner_function_x + (shared_function) * each_x
        inner_function_sq = inner_function_sq + (shared_function) ** 2
    d_j_wrt_theta0 = 1/number_of_datapoints*inner_function
    d_j_wrt_theta1 = 1/number_of_datapoints*inner_function_x
    cost_function = 1/(2*number_of_datapoints)*inner_function_sq
    cost_function_history.append(cost_function)
    theta0_new = theta0_old - alpha * d_j_wrt_theta0
    theta1_new = theta1_old - alpha * d_j_wrt_theta1
    print(theta0_new, theta1_new, cost_function)
    theta0_old = theta0_new
    theta1_old = theta1_new


plt.figure(figsize=(10,5))
plt.title("X - y Scatter plot")
plt.scatter(df['X'], df['y'], label='X - y values')
plt.legend(loc="upper left")
plt.grid(True)
plt.plot(df['X'], theta0_new + theta1_new * df['X'], label="Trend")
plt.show()

plt.figure(figsize=(20, 5))
plt.title("J vs iterations")
plt.legend(loc="upper left")
plt.grid(True)
plt.scatter([each for each in range(len(cost_function_history[2:]))], cost_function_history[2:], label="J")
plt.show()


if __name__ == '__main__':
    pass