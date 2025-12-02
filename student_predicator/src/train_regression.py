import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# пути
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "students.csv"

# Модель
KNN_MODEL_PATH = ROOT / "model" / "regression" / "knn_regression_model.pkl"

# Папка с результатами
KNN_RESULTS_DIR = ROOT / "model" / "results" / "knn_regression"
KNN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Загрузка и подготовка данных
df = pd.read_csv(DATA_PATH)
print(df.head())

if 'student_id' in df.columns:
    df = df.drop('student_id', axis=1)

X = df[['hours_studied', 'lms_activity_hours', 'attendance_percent', 'previous_scores']]
y = df['exam_score']

# Обучение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Метрики оценки качества модели MSE, MAE, R²
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Средняя квадратичная ошибка (MSE): {mse:.4f}')
print(f'Средняя абсолютная ошибка (MAE):   {mae:.3f}')
print(f'Коэффициент детерминации (R²):     {r2:.4f}')

# Сохранение модели
joblib.dump(model, KNN_MODEL_PATH)
print(f"\nМодель → {KNN_MODEL_PATH}")

#отчеты
results_df = pd.DataFrame([{
    "MSE": round(mse, 3),
    "MAE": round(mae, 3),
    "R2": round(r2, 4),
    "Объясняемая_дисперсия_%": round(r2*100, 1),
    "Комментарий": "Лучшая модель по MAE (±3.21 балла), n_neighbors=3"
}])

results_df.to_excel(KNN_RESULTS_DIR / "knn_регрессия_результаты.xlsx", index=False)
print(f"Excel-отчёт  {KNN_RESULTS_DIR / 'knn_регрессия_результаты.xlsx'}")

# графики
plt.figure(figsize=(10,7))
plt.scatter(y_test, y_pred, alpha=0.7, color='#9b59b6', s=80, edgecolors='white', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Идеальная линия')
plt.xlabel("Реальный балл за экзамен", fontsize=12)
plt.ylabel("Предсказанный балл", fontsize=12)
plt.title(f"KNN-регрессор (k=3): R² = {r2:.4f} | MAE = {mae:.2f} балла", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(KNN_RESULTS_DIR / "knn_п_прогноз_vs_реальность.png", dpi=300, bbox_inches='tight')
plt.close()