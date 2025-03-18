clear;

M = 0.5;
m = 0.2;
b = 0.1;
I = 0.006;
g = 9.8;
l = 0.3;
p = I*(M+m)+M*m*l^2; %denominator for the A and B matrices

% (A0,B0,C): continuous-time system
A0 = [0      1              0           0;
     0 -(I+m*l^2)*b/p  (m^2*g*l^2)/p   0;
     0      0              0           1;
     0 -(m*l*b)/p       m*g*l*(M+m)/p  0];

B0 =[     0;
     (I+m*l^2)/p;
          0;
        m*l/p];

C = [1 0 0 0;
     0 0 1 0];


% sampling time
Ts = 0.05;


% discretize
sysC = ss(A0,B0,C,[]);
sysD = c2d(sysC, Ts);
A = sysD.A;
B = sysD.B;

% dimensions
[nx,nu] = size(B);
[ny,~] = size(C);


% controller design
Q = C'*C;
Q(1,1) = 1000;
Q(3,3) = 10000;

R1 = eye(nu);
R2 = eye(ny);
[~, K, ~] = idare(A,B,1*Q,10*R1,[],[]);

[~, L, ~] = idare(A.', C.', 4*Q, R2, [], []);
L = L.';

% (F,G,H): resulting controller
F = A - B*K - L*C;
G = L;
H = K;
J = [0 0];



%% Simulation
iter = 2000;
xp0 = [0; 0; 0.5; 0];
xc0 = [0; 0; 0; 0];

% % variables for simulation with original controller
% xp = xp0;
% xc = xc0;
% u = [];
% y = [];
% x = [];


% 
% for i = 1:iter
%     % plant + original controller
%     y = [y, C*xp(:,i)];
%     u = [u, H*xc(:,i)];
%     xp = [xp, A*xp(:,i) + B*u(:,i)];
%     x = [x, xp(:,i)];
%     xc = [xc, F*xc(:,i) + G*y(:,i)];
% 
% end
% 
% figure(1)
% plot(Ts*(0:iter-1), u)
% 
% figure(2)
% plot(Ts*(0:iter-1), y)
% 




set_param('cart_pendulum_06/Observer based Discrete Controller', 'A', mat2str(F), 'B', mat2str(G), 'C', mat2str(H), 'D', mat2str(J));


A_d = sysD.A;
B_d = sysD.B;
C_d = sysD.C;
D_d = sysD.D;
set_param('cart_pendulum_06/Discrete plant', 'A', mat2str(A_d),'B', mat2str(B_d), 'C', mat2str(C_d),'D', mat2str(D_d));