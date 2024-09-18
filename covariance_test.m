clear

n = 3;
m = 1;
p = 1;

A = rand(n);
B = rand(n,m);
C = rand(p,n);
K = dlqr(A,B,eye(n),eye(m));
A = A-B*K;

Q = diag(rand(n,1));
R = diag(rand(p,1));
[L,P] = dlqr(A',C',Q,R);
L = L';
S = C*P*C' + R;

x = zeros(n,1);
xhat = x;
y = C*x;
z = y - C*xhat;
S_store = [];
Se_store = [];

r = 10000;
t = 0;
dr = r/5;
Q2 = 10*diag(rand(n,1));
for i = 1:r
    if i >= dr
        Q = Q2;
    end
    w = mvnrnd(zeros(n,1),Q)';
    v = mvnrnd(zeros(p,1),R)';
    x = A*x + w;
    y = C*x + v;
    z = [z ; (y - C*xhat)'];
    xhat = A*xhat + L*(y - C*A*xhat);
    if i > t
        Se = cov(z);
        S_store = [S_store ; S];
        Se_store = [Se_store ; Se];
    end
end

figure(1); plot(1:r-t,[S_store, Se_store]);
