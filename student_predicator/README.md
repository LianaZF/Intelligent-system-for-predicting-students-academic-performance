Установка и настройка
1. Требования к системе
Python 3.8 или выше

pip (менеджер пакетов Python)

2. Установка зависимостей
bash
# Перейдите в папку проекта
cd student_predicator

# Создайте виртуальное окружение (рекомендуется)
python -m venv venv

# Активируйте виртуальное окружение:
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Установите зависимости
pip install -r requirements.txt
Если файла requirements.txt нет, установите пакеты вручную:

bash
pip install pandas scikit-learn catboost xgboost matplotlib openpyxl flask numpy joblib
## Обучение моделей
**Вариант 1: Все модели сразу**
bash
python src/train.py
Что делает:

Обучает 5 различных моделей (Random Forest, KNN, Naive Bayes, Linear Regression, Voting Ensemble)

Проверяет 3 гипотезы

Сохраняет результаты в Excel

Генерирует графики

**Вариант 2: Только CatBoost (рекомендуется)**
bash
python src/catboost_only_final.py
Что делает:

Обучает CatBoost (лучшая модель)

Сохраняет финальные модели

Генерирует отчеты и графики

**Вариант 3: Сравнение XGBoost и CatBoost**
bash
python src/compare_xgboost_catboost.py
Что делает:

Сравнивает две модели градиентного бустинга

Выбирает лучшую

**Вариант 4: KNN регрессия**
bash
python src/train_regression.py
Что делает:

Обучает KNN-регрессор

Оценивает качество предсказаний

## Прогнозирование
**Интерактивный режим (консоль)**
bash
python src/predict.py
Пример использования:

text
Доступные модели:
   1. random_forest_model
   2. catboost_final_model
   3. knn_regression_model

Выберите номер модели (1-3): 2

Введите данные студента:
  → Часы самостоятельной подготовки в неделю: 10
  → Часы активности в LMS за семестр: 45
  → Посещаемость занятий, % (0–100): 85
  → Средний балл за предыдущие сессии: 78

РЕЗУЛЬТАТ:
ПРОГНОЗ: УСПЕШНЫЙ СТУДЕНТ — всё отлично
Уверенность: 87.5%


## Результаты и отчеты
**Где найти результаты:**
**Excel отчеты:**

model/final_report/Сравнение_всех_моделей.xlsx

model/catboost_final/catboost_валидация.xlsx

model/results/knn_regression/knn_регрессия_результаты.xlsx

**Графики:**

model/final_report/График_лучшая_гипотеза.png

model/catboost_final/catboost_финальный_график.png

model/results/knn_regression/knn_п_прогноз_vs_реальность.png

**Модели:**

model/catboost_final/catboost_classifier.pkl

model/catboost_final/catboost_regressor.pkl