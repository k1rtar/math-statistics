# Импортируем необходимые библиотеки
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma, expon

# 1) Выбор распределения
# Выбираем нормальное распределение, так как у него существуют все четыре момента.
mu, sigma = 0, 1 # параметры нормального распределения
N = 10000 # объем выборки
M = 1000 # количество выборок

# 2) Генерация выборок
# Генерируем большое количество выборок большого объема.
samples = np.random.normal(mu, sigma, (M, N))

# 3) Вычисление статистик
# Вычисляем соответствующие статистики для каждой выборки.
means = samples.mean(axis=1)
variances = samples.var(axis=1)
medians = np.median(samples, axis=1)

# 4) Построение гистограмм
# Строим гистограммы результатов для каждой статистики.
plt.figure(figsize=(18, 6))

plt.subplot(131)
plt.hist(means, bins=30, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma/np.sqrt(N))
plt.plot(x, p, 'k', linewidth=2)
plt.title('Выборочное среднее')

plt.subplot(132)
plt.hist(variances, bins=30, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = gamma.pdf(x, N/2, scale=2/(N-1))
plt.plot(x, p, 'k', linewidth=2)
plt.title('Выборочная дисперсия')

plt.subplot(133)
plt.hist(medians, bins=30, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma*np.sqrt(np.pi/(2*N)))
plt.plot(x, p, 'k', linewidth=2)
plt.title('Выборочная медиана')

plt.show()

# 5) Вывод статистик
# Выводим математическое ожидание, дисперсию и медиану.
print(f'Среднее: {means.mean()}, Дисперсия: {variances.mean()}, Медиана: {np.median(medians)}')

# 6) Вычисление 𝑛𝐹(𝑋(2)) и 𝑛(1−𝐹(𝑋(𝑛)))
# Вычисляем 𝑛𝐹(𝑋(2)) и 𝑛(1−𝐹(𝑋(𝑛))) для каждой выборки.
F_X2 = np.sum(samples <= 2, axis=1) / N
U1 = N * F_X2

F_Xn = np.sum(samples <= N, axis=1) / N
U2 = N * (1 - F_Xn)

# 7) Построение гистограмм для 𝑛𝐹(𝑋(2)) и 𝑛(1−𝐹(𝑋(𝑛)))
# Строим гистограммы результатов для 𝑛𝐹(𝑋(2)) и 𝑛(1−𝐹(𝑋(𝑛))).
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.hist(U1, bins=30, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = gamma.pdf(x, 2, scale=1)
plt.plot(x, p, 'k', linewidth=2)
plt.title('nF(x(2))')

plt.subplot(122)
plt.hist(U2, bins=30, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = expon.pdf(x, scale=1)
plt.plot(x, p, 'k', linewidth=2)
plt.title('n(1-F(X(n)))')

plt.show()

# 8) Вывод статистик
# Выводим среднее значение 𝑛𝐹(𝑋(2)) и 𝑛(1−𝐹(𝑋(𝑛))).
print(f'Среднее nF(x(2)): {U1.mean()}, Среднее nF(x(2)): {U2.mean()}')
