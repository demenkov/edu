// p5_diff_int.sce
// Численная производная sin(x)
x = 0:0.01:%pi;
y = sin(x);
dy = diff(y) ./ diff(x);
xm = (x(1:$-1) + x(2:$)) / 2;
clf(); plot(xm, dy);
xtitle("Численная производная sin(x)", "x", "dy/dx"); xgrid();
// Численный интеграл ∫0^π sin(x) dx
function val = f(x)
val = sin(x);
endfunction
I = intg(0, %pi, f);
disp("Интеграл sin(x) от 0 до π:"); disp(I); // ожидаемо ~2