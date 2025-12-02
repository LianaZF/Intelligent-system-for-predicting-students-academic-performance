from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor

# пути
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "students_15000_CTGAN.csv"
RESULTS_DIR = ROOT / "model" / "xgboost_vs_catboost_only"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Загрузка и подготовка данных
df = pd.read_csv(DATA_PATH)
df['risk_level'] = df['risk_category'].map({
    'В зоне риска': 0,
    'Требует внимания': 1,
    'Успешный студент': 2
})

# Три гипотезы
hypotheses = [
    ("hyp1", "Все признаки", ['hours_studied', 'lms_activity_hours', 'attendance_percent', 'previous_scores']),
    ("hyp2", "LMS + Посещаемость", ['lms_activity_hours', 'attendance_percent']),
    ("hyp3", "LMS + Посещ. + Прошлые", ['lms_activity_hours', 'attendance_percent', 'previous_scores'])
]

# разбиение 70/15/15
X_all = df[hypotheses[0][2]]
y_cls = df['risk_level']
y_reg = df['exam_score']

X_train, X_temp, y_cls_train, y_cls_temp, y_reg_train, y_reg_temp = train_test_split(
    X_all, y_cls, y_reg, test_size=0.3, random_state=42, stratify=y_cls
)

X_val, X_test, y_cls_val, y_cls_test, y_reg_val, y_reg_test = train_test_split(
    X_temp, y_cls_temp, y_reg_temp, test_size=0.5, random_state=42, stratify=y_cls_temp
)

results_val = []
models_to_test = ["XGBoost", "CatBoost"]

print("Сравнение XGBoost и CatBoost (по отдельности) на валидации...\n")

#отдельные модели
def train_clf(model_name: str, X_tr, y_tr):
    if model_name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=500,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1
        ).fit(X_tr, y_tr)
    else:  # CatBoost
        return CatBoostClassifier(
            iterations=500,
            random_state=42,
            verbose=0,
            class_weights=[2, 1, 1]  
        ).fit(X_tr, y_tr)


def train_reg(model_name: str, X_tr, y_tr):
    if model_name == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        ).fit(X_tr, y_tr)
    else:  # CatBoost
        return CatBoostRegressor(
            iterations=500,
            random_state=42,
            verbose=0
        ).fit(X_tr, y_tr)

# валидация

for hyp_name, hyp_title, feats in hypotheses:
    X_train_h = X_train[feats]
    X_val_h = X_val[feats]

    for model_name in models_to_test:
        # Классификация
        clf = train_clf(model_name, X_train_h, y_cls_train)
        pred_cls = clf.predict(X_val_h)
        acc = accuracy_score(y_cls_val, pred_cls)
        recall_risk = recall_score(y_cls_val, pred_cls, average=None)[0]

        # Регрессия
        reg = train_reg(model_name, X_train_h, y_reg_train)
        pred_reg = reg.predict(X_val_h)
        r2 = r2_score(y_reg_val, pred_reg)

        results_val.append({
            "Гипотеза": hyp_title,
            "Модель": model_name,
            "Accuracy_val": round(acc, 4),
            "Recall_риск_val": round(recall_risk, 4),
            "R2_val": round(r2, 4)
        })

# Таблица результатов на валидации
val_df = pd.DataFrame(results_val)
print(val_df.to_string(index=False))

# Выбор лучшей по R²
best = max(results_val, key=lambda x: x["R2_val"])
print(f"\nЛучшая модель на валидации: {best['Модель']} + {best['Гипотеза']} → R² = {best['R2_val']}")

best_feats = next(f for _, t, f in hypotheses if t == best['Гипотеза'])

# Объединяем train + val
X_train_full = pd.concat([X_train[best_feats], X_val[best_feats]])
y_cls_full = pd.concat([y_cls_train, y_cls_val])
y_reg_full = pd.concat([y_reg_train, y_reg_val])
X_test_best = X_test[best_feats]

# Финальные модели
final_clf = train_clf(best['Модель'], X_train_full, y_cls_full)
final_reg = train_reg(best['Модель'], X_train_full, y_reg_full)

# Предсказания
pred_cls_test = final_clf.predict(X_test_best)
pred_reg_test = final_reg.predict(X_test_best)

# Метрики
acc_test = accuracy_score(y_cls_test, pred_cls_test)
recall_risk_test = recall_score(y_cls_test, pred_cls_test, average=None)[0]
mae_test = mean_absolute_error(y_reg_test, pred_reg_test)
mse_test = mean_squared_error(y_reg_test, pred_reg_test)
r2_test = r2_score(y_reg_test, pred_reg_test)


print(f"Модель:         {best['Модель']}")
print(f"Гипотеза:       {best['Гипотеза']}")
print(f"Accuracy:       {acc_test:.4f}")
print(f"Recall (риск):  {recall_risk_test:.4f}")
print(f"R²:             {r2_test:.4f}")
print(f"MAE:            ±{mae_test:.2f} баллов")
print(f"MSE:            {mse_test:.2f}")

# График
plt.figure(figsize=(10, 7))
plt.scatter(y_reg_test, pred_reg_test, alpha=0.7, color='#9b59b6', s=70, edgecolor='k', linewidth=0.5)
min_v, max_v = y_reg_test.min() - 2, y_reg_test.max() + 2
plt.plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label='Идеальная линия')
plt.title(f"Лучшая моедль: {best['Модель']} | {best['Гипотеза']}\nR² = {r2_test:.4f} | MAE = ±{mae_test:.2f}", fontsize=14)
plt.xlabel("Реальный балл")
plt.ylabel("Предсказанный балл")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "финальный_график.png", dpi=300, bbox_inches='tight')
plt.close()

# Сохранение
val_df.to_excel(RESULTS_DIR / "сравнение_валидация.xlsx", index=False)

with open(RESULTS_DIR / "ФИНАЛЬНЫЙ_РЕЗУЛЬТАТ.txt", "w", encoding="utf-8") as f:
    f.write(f"Модель: {best['Модель']}\n"
            f"Гипотеза: {best['Гипотеза']}\n"
            f"Accuracy: {acc_test:.4f}\n"
            f"Recall (риск): {recall_risk_test:.4f}\n"
            f"R²: {r2_test:.4f}\n"
            f"MAE: ±{mae_test:.2f}\n")

joblib.dump(final_clf, RESULTS_DIR / "final_classifier.pkl")
joblib.dump(final_reg, RESULTS_DIR / "final_regressor.pkl")

print(f"\nсохранено в папку:\n{RESULTS_DIR}")