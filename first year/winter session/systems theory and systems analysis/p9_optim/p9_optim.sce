// p9_optim.sce
// Функция Розенброка
function y = rosenbrock(x)
    y = (1 - x(1))^2 + 100 * (x(2) - x(1)^2)^2;
endfunction

x0 = [-1.2; 1];

// Так как сигнатура вызова optim изменилась
// Используем код примера из официальной документации 
// https://help.scilab.org/docs/2026.0.0/ru_RU/optim.html
function [f, g, ind]=rosenbrockCost(x, ind)
    f = rosen(x);
    g = numderivative(rosen, x);
endfunction

[fopt, xopt] = optim(rosenbrockCost, x0)

disp("Минимум: fopt = " + string(fopt));
disp("В точке: xopt = " + string(xopt'));


// Проверим с fminsearch совместимой со старой сигнатурой вызова
[xopt, fopt] = fminsearch(rosen, x0);

disp("Минимум: fopt = " + string(fopt));
disp("В точке: xopt = " + string(xopt'));