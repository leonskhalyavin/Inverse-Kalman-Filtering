% CURRENTLY: KEEP 0 INPUT
clear;

n = 3;
m = 1; 
p = 1;

A = rand(n);
B = rand(n,m);
K = dlqr(A,B,eye(n),eye(m));
A = A-B*K;
C = rand(p,n);
Q = diag(rand(n,1));
R = diag(rand(p,1));
P = eye(n);
Qnominal = Q;
Rnominal = R;
[L,Pnominal] = dlqr(A',C',Qnominal,Rnominal);
L = L';
S = C*Pnominal*C' + R;

P0 = Pnominal;
% P0 = eye(n);
sys = linear_forward_kalman_filter(A,B,C,Q,R,P0,Qnominal,Rnominal);

x0 = rand(n,1);
r = 1000;
% K = dlqr(A,B,eye(n),eye(m));
K = zeros(m,n);
sys = sys.auto_generate_forward_problem(x0,r,K);
t = 0;

test.R = sys.R;
test.C = C;
test.P = P0;

[Qe,out] = l_ls_ikf(sys.system_trajectory,sys.L_trajectory,sys.xhat_trajectory,sys.K_trajectory,sys.y_trajectory,sys.P0,t,test);