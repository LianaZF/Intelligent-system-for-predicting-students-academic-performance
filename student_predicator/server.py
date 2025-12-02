# server.py — 100% работает в 2025 году
import http.server
import socketserver
import json
import joblib
import pandas as pd

# Загружаем модели один раз при старте
clf = joblib.load("model/catboost_final/catboost_classifier.pkl")
reg = joblib.load("model/catboost_final/catboost_regressor.pkl")

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        # Отвечаем на CORS preflight
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/predict':
            try:
                length = int(self.headers['Content-Length'])
                body = self.rfile.read(length)
                data = json.loads(body)

                # Формируем признаки (поддерживаем обе версии колонок)
                features = []
                if 'lms_activity_hours' in data:
                    features = [data['lms_activity_hours'], data['attendance_percent'], data['previous_scores']]
                elif 'hours_studied' in data:
                    features = [data['hours_studied'], data['attendance_percent'], data['previous_scores']]
                else:
                    features = [data.get('lms_activity_hours', 0), data.get('attendance_percent', 0), data.get('previous_scores', 0)]

                df = pd.DataFrame([features])

                risk = int(clf.predict(df)[0])        # 0, 1 или 2
                score = int(round(reg.predict(df)[0]))

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = json.dumps({"risk": risk, "score": score}).encode()
                self.wfile.write(response)

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

# Запуск сервера
PORT = 8000
print("Сервер запущен → http://127.0.0.1:8000")
print("Открой monitoring_v3.html — теперь точно заработает!")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен")
        