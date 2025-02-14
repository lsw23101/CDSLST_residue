clear;

%% Plant discrete model and Sampling time

M = 0.5;
m = 0.2;
b = 0.1;
I = 0.006;
g = 9.8;
l = 0.3;

p = I*(M+m)+M*m*l^2; %denominator for the A and B matrices

A = [0      1              0           0;
     0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
     0      0              0           1;
     0 -(m*l*b)/p       m*g*l*(M+m)/p  0];
B = [     0;
     (I+m*l^2)/p;
          0;
        m*l/p];
C = [1 0 0 0];
D = [0];

states = {'x' 'x_dot' 'phi' 'phi_dot'};
inputs = {'u'};
outputs = {'x'};

sys_ss = ss(A,B,C,D,'statename',states,'inputname',inputs,'outputname',outputs);

Ts = 0.01;

sys_d = c2d(sys_ss,Ts,'zoh')



%% observer based controller design

% dimensions
[n,m] = size(B);
[l,~] = size(C);

% controller design
% Q = eye(n);
% controller design
Q = [1 0 0 0;
     0 1 0 0;
     0 0 1 0;
     0 0 0 1];

R1 = 1 * eye(m);
R2 = eye(l);
[~, K, ~] = idare(A,B,Q,R1,[],[]);
K = -K
[~, L, ~] = idare(A.', C.', Q, R2, [], []);
L = L.'


clear;
close all;

%% Plant discrete model and Sampling time

M = 0.5;
m = 0.2;
b = 0.1;
I = 0.006;
g = 9.8;
l = 0.3;

p = I*(M+m)+M*m*l^2; %denominator for the A and B matrices

A0 = [0      1              0           0;
     0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
     0      0              0           1;
     0 -(m*l*b)/p       m*g*l*(M+m)/p  0];
B0 = [     0;
     (I+m*l^2)/p;
          0;
        m*l/p];

C = [1 0 0 0];

D = [0];



% sampling time
Ts = 0.01;

% discretize
sysC = ss(A0,B0,C,[]);
sysD = c2d(sysC, Ts);
A = sysD.A;
B = sysD.B;

% dimensions
[n,m] = size(B);
[l,~] = size(C);

% controller design
Q = [1 0 0 0;
     0 1 0 0;
     0 0 1 0;
     0 0 0 1];

R1 = eye(m);
R2 = eye(l);
[~, K, ~] = idare(A,B,Q,R1,[],[]);
K = -K
[~, L, ~] = idare(A.', C.', Q, R2, [], []);
L = L.'