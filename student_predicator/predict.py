# predict.py
# Проверка работы модели на новых данных
# Запуск: python predict.py

import joblib
from pathlib import Path
import pandas as pd

# Пути
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "classifier.pkl"

# Загружаем обученную модель
model = joblib.load(MODEL_PATH)
print("Модель успешно загружена.\n")

# Описание классов
classes = {
    0: "Незачёт (группа риска)",
    1: "Удовлетворительно",
    2: "Хорошо",
    3: "Отлично"
}

print("Введите данные нового студента (можно вводить несколько раз):")
print("-" * 60)

while True:
    try:
        # Ввод данных
        hours = float(input("Часы самостоятельной подготовки в неделю: "))
        lms   = float(input("Часы активности в LMS за семестр: "))
        att   = float(input("Посещаемость занятий, % (0–100): "))
        prev  = float(input("Средний балл за предыдущие сессии (0–100): "))

        # Формируем DataFrame точно так же, как при обучении
        new_student = pd.DataFrame([{
            'hours_studied': hours,
            'lms_activity_hours': lms,
            'attendance_percent': att,
            'previous_scores': prev
        }])

        # Предсказание
        prediction = model.predict(new_student)[0]
        probability = model.predict_proba(new_student)[0]
        prob_max = max(probability)

        print("\n" + "="*60)
        print(f"ПРОГНОЗ УСПЕВАЕМОСТИ ПО ИТОГАМ ЭКЗАМЕНА:")
        print(f"→ {classes[prediction]}")
        print(f"Уверенность модели: {prob_max*100:.1f} %")
        print("="*60 + "\n")

        # Предлагаем проверить ещё одного
        again = input("Проверить ещё одного студента? (да/нет): ").strip().lower()
        if again not in ['да', 'yes', 'д', 'y', 'lf']:
            print("Работа завершена.")
            break
        print("-" * 60)

    except ValueError:
        print("Ошибка: введите числовые значения!\n")
    except KeyboardInterrupt:
        print("\n\nРабота прервана пользователем.")
        break