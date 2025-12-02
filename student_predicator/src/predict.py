import joblib
from pathlib import Path
import pandas as pd

# пути
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "model" / "models"
SCALERS_DIR = ROOT / "model" / "scalers"

# Список всех доступных моделей
model_files = sorted(MODELS_DIR.glob("*_model.pkl"))
if not model_files:
    print("ОШИБКА: Не найдено ни одной модели в папке model/models/")
    print("Сначала запусти: python src/train.py")
    exit()

print("Доступные модели:")
for i, path in enumerate(model_files, 1):
    print(f"   {i}. {path.stem}")

print("\n" + "="*70)
choice = int(input("Выберите номер модели (1–9): ")) - 1
if choice < 0 or choice >= len(model_files):
    print("Неверный номер!")
    exit()

selected_model_path = model_files[choice]
model_name = selected_model_path.stem

model = joblib.load(selected_model_path)
scaler_path = SCALERS_DIR / f"{model_name.replace('_model', '')}_scaler.pkl"
scaler = joblib.load(scaler_path) if scaler_path.exists() else None

# Классы риска
classes = {
    0: "ВЫСОКИЙ РИСК — срочно нужна поддержка!",
    1: "ЗОНА ВНИМАНИЯ — рекомендуется помощь",
    2: "УСПЕШНЫЙ СТУДЕНТ — всё отлично"
}

print("\n" + "=" * 70)
print("Ввод данных студента. Для выхода нажмите Ctrl+C")
print("-" * 70)

while True:
    try:
        print("\nВведите данные:")
        hours = float(input("  → Часы самостоятельной подготовки в неделю: "))
        lms   = float(input("  → Часы активности в LMS за семестр: "))
        att   = float(input("  → Посещаемость занятий, % (0–100): "))
        prev  = float(input("  → Средний балл за предыдущие сессии: "))

        # Формируем данные
        student = pd.DataFrame([{
            'hours_studied': hours,
            'lms_activity_hours': lms,
            'attendance_percent': att,
            'previous_scores': prev
        }])

        if scaler:
            student = scaler.transform(student)

        # Предсказание
        pred = model.predict(student)[0]
        proba = model.predict_proba(student)[0]
        confidence = max(proba) * 100

        print("\n" + "="*70)
        print(f"МОДЕЛЬ: {model_name}")
        print(f"ПРОГНОЗ: {classes[pred]}")
        print(f"Уверенность: {confidence:.1f}%")
        if confidence < 60:
            print("Внимание: Низкая уверенность")
        print("="*70)

        again = input("\nПроверить ещё одного студента? (да/любая клавиша = да, нет/n = выход): ")
        if again.lower() in ['нет', 'n', 'no']:
            print("До свидания!")
            break

    except ValueError:
        print("Ошибка: введите число!")
    except KeyboardInterrupt:
        print("\n\nРабота завершена.")
        break