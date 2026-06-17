docker build -t climate-test .
docker run --rm climate-test            # как было — на уровнях (сеть проигрывает)
docker run --rm climate-test --diff     # дифференцирование (сеть выигрывает)
docker run --rm climate-test --diff --horizon 12    # год вперёд

Можно переопределять параметры прямо при запуске:

docker run --rm climate-test --window 36 --epochs 100   # окно 36 мес.
docker run --rm climate-test --download                 # перекачать свежие данные