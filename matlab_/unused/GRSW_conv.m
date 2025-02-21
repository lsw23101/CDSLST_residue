clc;
clear all;
clc;
close all;
%% Observer-based controller design
% Example: four-tank system
% parameters
A1 = 28; A2 = 32; A3 = 28; A4 = 32;
a1 = 0.071; a2 = 0.057; a3 = 0.071; a4 = 0.057;
kc = 0.5;
g = 981;
h10 = 12.4; h20 = 12.7; h30 = 1.8; h40 = 1.4;
v10 = 3; v20 = 3;
k1 = 3.33; k2 = 3.35;
g1 = 0.7; g2 = 0.6;

% (A0,B0,C): continuous-time system
A0 = [-a1/A1 * 2*g / (2*sqrt(2*g*h10)), 0 , a3/A1 * 2*g / (2*sqrt(2*g*h30)), 0;...
      0, -a2/A2 * 2*g / (2*sqrt(2*g*h20)), 0 , -a4/A2 * 2*g / (2*sqrt(2*g*h40));...
      0, 0, -a3/A3 * 2*g / (2*sqrt(2*g*h30)), 0 ;...
      0, 0, 0, -a4/A4 * 2*g / (2*sqrt(2*g*h40))];
B0 = [g1*k1/A1, 0;...
      0, g2*k2/A2;...
      0, (1-g2)*k2/A3;...
      (1-g1)*k1/A4,0];
C = [kc, 0, 0, 0;...
     0, kc, 0, 0];

% sampling time
Ts = 0.1;

% discretize
sysC = ss(A0,B0,C,[]);
sysD = c2d(sysC, Ts);
A = sysD.A;
B = sysD.B;

% dimensions
[n,m] = size(B);
[l,~] = size(C);

% controller design
Q = eye(n);
R1 = eye(m);
R2 = eye(l);
[~, K, ~] = idare(A,B,Q,R1,[],[]);
K = -K;
[~, L, ~] = idare(A.', C.', Q, R2, [], []);
L = L.';

% (F,G,H): resulting controller
F = A + B*K - L*C;
G = L;
H = K;



%% Converting the state matrix into integers
% One may freely change F, G, and H to different systems as they choose
% Finds R such that (F-RH) is an integer matrix through pole-placement


% Assign integer poles to (F-RH)
poles = [0,1,2,-1]; % Must consist of n-integers!
R = place(F.',H.',poles);
R = R.';

% Convert to modal canonical form
sys = ss(F-R*H, G, H, [], Ts); % Ts 추가
[csys,T] = canon(sys, 'modal');
F_ = T*(F-R*H)/T
R_ = T*R
G_ = T*G
H_ = H/T

%% plot

iter = 100;
xp0 = [1; 0; 0; 0];
xc0 = [0; 0; 0; 0];

% variables for simulation with original controller
xp = xp0;
xc = xc0;
u = [];
y = [];

% variables for simulation with converted controller
x_p = xp0;
x_c = xc0;
u_ = [];
y_ = [];
r_ = [];

% 변환 전 시스템 
% 플랜트: A B C
% 컨트롤러: F G P >>> 변환 후 >>> F_x +G_y + R_r // H_ J_ // P_




for i = 1:iter
    % plant + original controller
    y = [y, C*xp(:,i)];
    u = [u, P*xc(:,i)];

    xp = [xp, A*xp(:,i) + B*u(:,i)];
    xc = [xc, F*xc(:,i) + G*y(:,i)];

    % converted controller
    y_ = [y_, C*x_p(:,i)];
    u_ = [u_, P_*x_c(:,i)];
    r_ = [r_, H_*x_c(:,i) + J_*y_(:,i)];

    x_p = [x_p, A*x_p(:,i) + B*u_(:,i)];
    x_c = [x_c, F_*x_c(:,i) + G_*y_(:,i) + R_*r_(:,i)];
end


figure(1)
plot(Ts*(0:iter-1), u)
hold on

plot(Ts*(0:iter-1), u_)
hold on

plot(Ts*(0:iter-1), r_)


title('Control input and residue ')

legend('original 1','converted 1','residue')

figure(2)
plot(Ts*(0:iter-1), y)
hold on

plot(Ts*(0:iter-1), y_)

title('Plant output y')
legend('original 1','converted 1')
