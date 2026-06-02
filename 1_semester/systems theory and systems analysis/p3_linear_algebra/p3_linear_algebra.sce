// p3_linear_algebra.sce
A = [3 2 -1; 2 -2 4; -1 0.5 -1];
b = [1; -2; 0];
x = A \ b; // решение СЛАУ
disp(x);
// Проверка невязки
res = A*x - b;
disp("Невязка:"); disp(res);
// Собственные значения/векторы (дополнительно)
[vecs, vals] = spec(A); // eigenvectors/eigenvalues
disp(vals);