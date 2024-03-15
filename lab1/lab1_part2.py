# Импортируем необходимые библиотеки
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
# Используем функцию read_csv из библиотеки pandas для чтения CSV файла
df = pd.read_csv('sex_bmi_smokers.csv')

# Расчет количества курящих мужчин и некурящих женщин
# Используем логическую индексацию для фильтрации данных
smoking_men = df[(df['sex'] == 'male') & (df['smoker'] == 'yes')].shape[0]
non_smoking_women = df[(df['sex'] == 'female') & (df['smoker'] == 'no')].shape[0]

# Выводим результаты
print(f"Количество курящих мужчин: {smoking_men}")
print(f"Количество некурящих женщин: {non_smoking_women}")

# Расчет статистических показателей ИМТ
# Группируем данные по полу и статусу курения, затем применяем функцию agg для расчета статистических показателей
groups = df.groupby(['sex', 'smoker'])['bmi']
stats = groups.agg(['mean', 'var', 'median', lambda x: x.quantile(0.6)]).rename(columns={'mean': 'Среднее значение', 'var': 'Дисперсия', 'median': 'Медиана', '<lambda>': 'Квантиль порядка 3/5'})

# Выводим результаты
print(stats)

# Построение эмпирической функции распределения, гистограммы и box-plot ИМТ
# Используем библиотеки seaborn и matplotlib для построения графиков
for (sex, smoker), group in df.groupby(['sex', 'smoker']):
    sns.ecdfplot(group['bmi'], label=f"{sex}, {'курит' if smoker == 'yes' else 'не курит'}")
plt.legend()
plt.show()

for (sex, smoker), group in df.groupby(['sex', 'smoker']):
    sns.histplot(group['bmi'], kde=True, label=f"{sex}, {'курит' if smoker == 'yes' else 'не курит'}")
plt.legend()
plt.show()

sns.boxplot(x='sex', y='bmi', hue='smoker', data=df)
plt.show()
