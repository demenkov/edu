# Фиксированный Python — неважно, что стоит на Mac
FROM python:3.11-slim

WORKDIR /app

# Зависимости: torch ставим CPU-сборкой (легче), остальное из PyPI
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Скачиваем данные NASA GISTEMP на этапе сборки и кладём в образ.
# ADD умеет тянуть URL без curl/wget. После этого контейнеру интернет не нужен.
ADD https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv /app/GLB.Ts+dSST.csv

COPY climate_forecast_test.py .

# Данные подставляются всегда; флаги из `docker run` ДОБАВЛЯЮТСЯ к этим.
#   docker run --rm img            -> на данных GISTEMP
#   docker run --rm img --diff     -> GISTEMP + режим дифференцирования
#   docker run --rm img --window 36 --epochs 100
# Переопределить файл при желании:  docker run --rm img --csv other.csv
ENTRYPOINT ["python", "climate_forecast_test.py", "--csv", "GLB.Ts+dSST.csv", "--gistemp"]