// p4_equations.sce
// Корни полинома x^2 - 5x + 6
p = [1 -5 6];
r = roots(p);
disp("Корни:"); disp(r);
// Система нелинейных уравнений:
// x^2 + y^2 = 1, x - y = 0
function F = sys(u)
x = u(1); y = u(2);
F(1) = x^2 + y^2 - 1;
F(2) = x - y;
endfunction
[sol, fval] = fsolve([0.5; 0.5], sys);
disp("Решение:"); disp(sol);
disp("Проверка:"); disp(fval); // должно быть близко к [0;0]