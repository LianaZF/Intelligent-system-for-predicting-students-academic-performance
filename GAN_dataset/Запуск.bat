@echo off
echo ========================================
echo Запускаем генерацию 15 000 студентов...
echo ========================================

:: Удаляем старое окружение, если было
rmdir /s /q venv311 2>nul

:: Создаём новое окружение на Python 3.11
py -3.11 -m venv venv311

:: Активируем его
call venv311\Scripts\activate

:: Устанавливаем всё нужное (это займёт 2–4 минуты)
python -m pip install --upgrade pip
pip install pandas "sdv[ctgan]" torch torchvision torchaudio

:: Запускаем генерацию 15 000 строк
python generate_15000_CTGAN.py

echo.
echo ГОТОВО! Файл students_15000_CTGAN_synthetic.csv создан!
echo Нажми любую клавишу, чтобы закрыть окно...
pause