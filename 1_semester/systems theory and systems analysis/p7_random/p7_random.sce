// p7_random.sce
n = 1000;
X = grand(1, n, "nor", 0, 1); // нормальное(0,1)
m = mean(X); s = stdev(X);
clf(); histplot(30, X);
xtitle("Нормальное распределение (0,1)","X","частота");
disp("Выборочное среднее:"); disp(m);
disp("Выборочное СКО:"); disp(s);