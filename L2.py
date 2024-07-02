import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import preprocessing
import csv
from sklearn.model_selection import train_test_split

data_url = "Mobile Price Prediction.csv"


def read(data_url):
    with open(data_url, 'r') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    return data


def receive(data):
    feature_names = [key for key in data[0].keys() if key in [
        'Screen Size (inches)', 'RAM (GB)', 'Storage (GB)',
        'Battery Capacity (mAh)', 'Camera Quality (MP)']]
    features = [[float(row[key]) for key in feature_names] for row in data]
    price = [float(row['Price ($)']) for row in data]
    return preprocessing.normalize(np.array(features)), preprocessing.normalize([np.array(price)])[0]


class GoldenSectionSearch:
    def search(self, f, a_init, b_init, epsilon):
        c = (3 - np.sqrt(5)) / 2
        a, b, y, z, f_y, f_z = [], [], [], [], [], []
        y.append(a_init + c * (b_init - a_init))
        z.append(a_init + (1 - c) * (b_init - a_init))
        f_y.append(f(y[-1]))
        f_z.append(f(z[-1]))

        if f_y[-1] <= f_z[-1]:
            b.append(z[-1])
            a.append(a_init)
        else:
            a.append(y[-1])
            b.append(b_init)

        itr = 0
        while b[-1] - a[-1] > epsilon:
            itr += 1
            if f_y[-1] <= f_z[-1]:
                z.append(y[-1])
                f_z.append(f_y[-1])
                y.append(a[-1] + c * (b[-1] - a[-1]))
                f_y.append(f(y[-1]))
            else:
                y.append(z[-1])
                f_y.append(f_z[-1])
                z.append(a[-1] + (1 - c) * (b[-1] - a[-1]))
                f_z.append(f(z[-1]))

            if f_y[-1] <= f_z[-1]:
                a.append(a[-1])
                b.append(z[-1])
            else:
                a.append(y[-1])
                b.append(b[-1])

        if f_y[-1] <= f_z[-1]:
            x = y[-1]
            f_x = f_y[-1]
        else:
            x = z[-1]
            f_x = f_z[-1]

        return x, f_x


# Функція втрат з L2 регуляризацією
def Loss_L2(X, y, W, b, alpha):
    return np.sum((y - (W @ X.T + b)) ** 2) / X.shape[0] + alpha * np.sum(W ** 2)
# Градієнт функції типувтрат з L2 регуляризацією по вагам W
def dldw_L2(X, y, W, b, alpha):
    return -2 * X.T @ (y - (W @ X.T + b)) / X.shape[0] + 2 * alpha * W
def dldb_L2(X, y, W, b):
    return -2 * np.sum(y - (W @ X.T + b)) / X.shape[0]
def gradient_descent_L2(X, y, W, b, alpha, max_iterations=10000, tol=1e-9):
    itr = 0
    losses = []
    while itr < max_iterations:
        loss = Loss_L2(X, y, W, b, alpha)
        beta = get_beta_L2(X, y, W, b, alpha)
        W_new = W - beta * dldw_L2(X, y, W, b, alpha)
        b_new = b - beta * dldb_L2(X, y, W, b)
        new_loss = Loss_L2(X, y, W_new, b_new, alpha)
        if np.abs(loss - new_loss) < tol:
            break
        W, b = W_new, b_new
        print(f"Iteration {itr} | W = {W}| b = {b}")
        losses.append(new_loss)
        itr += 1
    return W, b, losses


# для пошуку бета яке мінімізує функцію на певному кроці
def get_beta_L2(X, y, W, b, alpha):
    gss = GoldenSectionSearch()
    epsilon = 1e-5

    def next_approx(beta):
        return Loss_L2(X, y, W - beta * dldw_L2(X, y, W, b, alpha), b - beta * dldb_L2(X, y, W, b), alpha)

    beta, _ = gss.search(next_approx, 0, 10, epsilon)
    return beta


print("*******Using gradient descent with L2 regularization*******")
X, y = receive(read(data_url))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
alpha = 0.1
W = np.array([0.4] * X.shape[1])
b = 0.4
start_time = time.time()
W, b, losses = gradient_descent_L2(X_train, y_train, W, b, alpha)
print("Loss on test set:", Loss_L2(X_test, y_test, W, b, alpha))
end_time = time.time()
y_pred_test = W @ X_test.T + b

execution_time = end_time - start_time
print("Час виконання: ", execution_time, " секунд")


