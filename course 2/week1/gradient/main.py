import pandas as pd
import numpy as np
from scipy import spatial as sp
adver_data = pd.read_csv('advertising.csv')

X = np.array([adver_data['TV'].values, adver_data['Radio'].values, adver_data['Newspaper'].values])
y = adver_data['Sales'].values
print(y.shape)

means, stds = np.mean(X, axis=1), np.std(X, axis=1)
print(means)
print(stds)
print(means.shape)

X = np.array([(X[i] - means[i])/stds[i] for i in range(3)])

one = np.ones(np.shape(X[0]))
X = np.hstack(X)
X = np.hstack((X, one))
X = X.reshape((4, len(X)//4))
X = X.T

def mserror(y, y_pred):
    return sum((y-y_pred)*(y-y_pred)) / len(y)
a = adver_data['Sales'][:30]
print(a.shape)
b = np.array([np.median(adver_data['Sales'])]*30)
print( mserror(a,b).shape)

answer1 = mserror(adver_data['Sales'], np.array([np.median(adver_data['Sales'])]*200))
print(round(answer1, 3))
print(adver_data['Sales'].shape)
print(np.array([np.median(adver_data['Sales'])]*200).shape)

def normal_equation(X, y):
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))


norm_eq_weights = normal_equation(X, y)
print(norm_eq_weights)

nmeans = np.array(np.mean(X, axis=0))
print(nmeans)
print(norm_eq_weights)
answer2 = np.dot(nmeans, norm_eq_weights)
print(answer2)
print(round(answer2, 3))

def linear_prediction(X, w):
    return np.dot(X, w)
V = linear_prediction(X, norm_eq_weights)
print(X.shape, norm_eq_weights.shape)
print(V.shape)

answer3 = mserror(y, V)
print(answer3)
print(type(answer3))
#print(round(answer3, 3))
print(X.shape)
print(y.shape, V.shape)

def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    grad0 = linear_prediction(X[train_ind], w) - y[train_ind]
    grad1 = X[train_ind][1] * (linear_prediction(X[train_ind], w) - y[train_ind])
    grad2 = X[train_ind][2] * (linear_prediction(X[train_ind], w) - y[train_ind])
    grad3 = X[train_ind][3] * (linear_prediction(X[train_ind], w) - y[train_ind])
    return  w - 2 * eta * np.array([grad1, grad2, grad3, grad0]) / len(y)
print(stochastic_gradient_step(X, y, norm_eq_weights, 2, eta=0.01))
t = stochastic_gradient_step(X, y, norm_eq_weights, 2, eta=0.01)
print(t.shape)
print(linear_prediction(X, t).shape)
print(mserror(y, linear_prediction(X, t)))


def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e5,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    # Инициализируем расстояние между векторами весов на соседних
    # итерациях большим числом.
    weight_dist = np.inf
    # Инициализируем вектор весов
    w = w_init
    # Сюда будем записывать ошибки на каждой итерации
    errors = []
    # Счетчик итераций
    iter_num = 0
    # Будем порождать псевдослучайные числа
    # (номер объекта, который будет менять веса), а для воспроизводимости
    # этой последовательности псевдослучайных чисел используем seed.
    np.random.seed(seed)

    # Основной цикл
    while weight_dist > min_weight_dist and iter_num < max_iter:
        # порождаем псевдослучайный
        # индекс объекта обучающей выборки
        random_ind = np.random.randint(X.shape[0])
        w_new = stochastic_gradient_step(X, y, w, random_ind, eta)
        #print('w', w, w.shape, 'w_new', w_new, w_new.shape)
        weight_dist = sp.distance.euclidean(w, w_new)
        y_predict = linear_prediction(X, w)
        #print('current y_predict shape:', '\n', y_predict.shape)
        err = mserror(y, y_predict)
        w = w_new
        # print('current err:', err, '\n', err.shape)
        errors.append(err)
        iter_num += 1

    return w, errors


print(linear_prediction(X, norm_eq_weights).shape)
print(y.shape)

w_init = np.array([[0, 0, 0, 0]])
w_init = w_init.reshape(4,1)
#Z = X[:10]
#print(Z)
#stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, y, w_init, eta=0.01, max_iter=1e5,
                               # min_weight_dist=1e-8, seed=42, verbose=True)
#print(stoch_grad_desc_weights, stoch_errors_by_iter)



#print(stoch_grad_desc_weights)

#print(stoch_errors_by_iter[-1][-1])