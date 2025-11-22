// p6_fit.sce
// Синтетические данные
x = linspace(0, 10, 50)';
y = 2 + 0.8*x;
noise = grand(50, 1, "nor", 0, 0.5);
y + noise;
// Линейная регрессия (reglin): y ≈ a0 + a1*x
a = reglin(x, y); // a(1)=a0, a(2)=a1
y_hat = a(1) + a(2)*x;
// Качество (RMSE)
rmse = sqrt(mean((y - y_hat).^2));
disp("Оценки:"); disp(a);
disp("RMSE:"); disp(rmse);
// Интерполяция
xi = 0:0.1:10;
yi = interp1(x', y', xi, "linear"); // линейная интерполяция
clf(); plot(x, y, "o"); plot(x, y, "-"); plot(xi, yi, "--");
legend("данные","истинная модель","интерполяция"); xgrid();
// Полиномиальная аппроксимация (2-й степени) через МНК
V = [ones(size(x,1),1) x x.^2];
coef = V \ y;
y2 = V*coef;
disp("Коэф. полинома степени 2:"); disp(coef);
