import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import os
import warnings
warnings.filterwarnings("ignore")

# Фиксируем путь
BASE_PATH = r"D:\Project_PK\GAN_dataset"
os.chdir(BASE_PATH)
print(f"Работаем тут: {BASE_PATH}")

# Загружаем оригинальные 200 строк
df = pd.read_csv("students.csv")
print(f"Оригинальных строк загружено: {len(df)}")

data = df.drop("student_id", axis=1)

# Метаданные
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

synthesizer = CTGANSynthesizer(
    metadata,
    enforce_rounding=True,
    epochs=600,              # чуть больше эпох для большего объёма
    batch_size=500,          # большой батч = быстрее на больших данных
    generator_dim=(256, 256, 256),
    discriminator_dim=(256, 256, 256),
    verbose=True,
    cuda=True                # если есть NVIDIA — будет в 5–10 раз быстрее!
)

synthesizer.fit(data)

# Генерируем сразу 14 800 синтетических + 200 оригинальных = 15 000
synthetic = synthesizer.sample(num_rows=14800)

# Склеиваем
full = pd.concat([data, synthetic], ignore_index=True)
full = full.sample(frac=1, random_state=42).reset_index(drop=True)

# Добавляем risk_category
def get_risk(score):
    if score >= 75:
        return "Успешный студент"
    elif score >= 50:
        return "Требует внимания"
    else:
        return "В зоне риска"

full["risk_category"] = full["exam_score"].apply(get_risk)

# ID от S00001 до S15000
full.insert(0, "student_id", [f"S{i:05d}" for i in range(1, 15001)])

# Сохраняем
output_file = "students_15000_CTGAN.csv"
full.to_csv(output_file, index=False)

print("\nГОТОВО! 15 000 строк сгенерировано!")
print(f"Файл сохранён: {output_file}")
print(f"Размер: {len(full):,} строк")
print("\nРаспределение классов:")
print(full["risk_category"].value_counts().sort_index())
print("\nКорреляции с exam_score:")
print(full.corr(numeric_only=True)["exam_score"].sort_values(ascending=False))