// p9_optim.sce
// Функция Розенброка
function y = rosen(x)
    y = (1 - x(1))^2 + 100 * (x(2) - x(1)^2)^2;
endfunction

x0 = [-1.2; 1];

// Используем fminsearch вместо optim
[xopt, fopt] = fminsearch(rosen, x0);

disp("Минимум: fopt = " + string(fopt));
disp("В точке: xopt = " + string(xopt'));
