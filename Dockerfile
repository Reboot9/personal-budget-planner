FROM python:3.11-slim

WORKDIR /personal_budget_planner

RUN apt-get update && apt-get install -y \
    libpq-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

COPY wait-for-db.sh /wait-for-db.sh
RUN chmod +x /wait-for-db.sh

CMD sh -c "/wait-for-db.sh && \
    python manage.py migrate --noinput && \
    python manage.py collectstatic --noinput && \
    gunicorn --bind 0.0.0.0:8000 personal_budget_planner.wsgi:application --workers 4 --threads 4 --capture-output --log-level info"