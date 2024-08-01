# Використовуємо базовий образ з Python
FROM python:3.12.1

# Встановлюємо робочий каталог в контейнері
WORKDIR /app

# Копіюємо файл вимог і встановлюємо залежності
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копіюємо весь код програми в контейнер
COPY . .

COPY models /app/models

# Вказуємо команду для запуску програми
CMD ["streamlit", "run", "web_part_streamlit.py"]
