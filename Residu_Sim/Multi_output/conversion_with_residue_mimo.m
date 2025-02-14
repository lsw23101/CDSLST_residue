clear;

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

C = [1 0 0 0
     0 0 1 0];

D = [0];



% sampling time
Ts = 0.01;

% discretize
sysC = ss(A0,B0,C,[]);
sysD = c2d(sysC, Ts);
A = sysD.A;
B = sysD.B;

%% observer based controller design

% dimensions
[n,m] = size(B);
[l,~] = size(C);

% controller design
% Q = eye(n);
Q = [10 0 0 0; 
     0 1 0 0;
     0 0 10 0;
     0 0 0 1];

R1 = 1 * eye(m);
R2 = eye(l);
[~, K, ~] = idare(A,B,Q,R1,[],[]);
K = -K;
[~, L, ~] = idare(A.', C.', Q, R2, [], []);
L = L.';


F = A+B*K-L*C; % 여기 부호를 어떻게 해야되는거지
G = L;
H = -C;
J = [1 0; 0 1];
P = K;



%% F 행렬 정수화 

% Assign integer poles to (F-RH)
poles = [0, 1, -1, 2]; % Must consist of n-integers!
R = place(F.',H.',poles);
R = R.';

% Convert to modal canonical form
sys = ss(F-R*H, G-R*J, H, [], Ts); % Ts 추가
[csys,T] = canon(sys, 'modal');
F_ = T*(F-R*H)/T
R_ = T*R
G_ = T*(G-R*J)
H_ = H/T

J_ = J
P_ = P/T



%% 양자화하기 

% H = [0 0 0 1] 이고 J는 1 이라서 s를 한번만 곱해도 되는 상황...
% 이제는 H_가 정수가 아니다..

% quantization parameters
r = 0.0001;
s = 0.0001;

% quantization of control parameters
qG = round(G_/s);
qH = round(H_/s);
qP = round(P_/s);
qJ = round(J_/(s*s));
qR = round(R_/s);


%% Simulation
iter = 10000;
xp0 = [0; 0; 0.005; 0];
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

% variables for simulation with converted & quantized controller
Xp = xp0;
qXc = round(T*xc0/(r*s));% 일단 컨트롤러 초기값이 0이라서 영향은 없음...
Xc = xc0;
Y = [];
U = [];
resi = [];
qY = [];
qU = [];
qresi = [];
residue = [];


% 변환 전 시스템 
% 플랜트: A B C
% 컨트롤러: F G P >>> 변환 후 >>> F_x +G_y + R_r // H_ J_ // P_

diff_u = []; % Control input difference
diff_Xc = []; % Controller state difference

for i = 1:iter
    % 외란 추가: i == 10일 때 u에 1만큼 추가
    if i == 10
        disturbance = 0;
    else
        disturbance = 0;
    end

    % plant + original controller
    y = [y, C*xp(:,i)];
    u = [u, P*xc(:,i) + disturbance];


    xp = [xp, A*xp(:,i) + B*u(:,i)];
    xc = [xc, F*xc(:,i) + G*y(:,i)];

    % converted controller
    y_ = [y_, C*x_p(:,i)];
    u_ = [u_, P_*x_c(:,i) + disturbance];
    r_ = [r_, H_*x_c(:,i) + J_*y_(:,i)];

    x_p = [x_p, A*x_p(:,i) + B*u_(:,i)];
    x_c = [x_c, F_*x_c(:,i) + G_*y_(:,i) + R_*r_(:,i)];


    % % plant + quantized controller
    Y = [Y, C*Xp(:,i)];
    qY = [qY, round(Y(:,i)/r)];
    % 여기서 qY = round(Y/r)
    qU = [qU, qP*qXc(:,i)];
    % 여기서 qU는 /r*s*s
    U = [U, qU(:,i)*r*s*s];

    qresi = [qresi, qH*qXc(:,i) + qJ*qY(:,i)]; % 여기 들어가는건 qY
    resi = [resi, qresi(:,i)*s*s];
    residue = [residue, r*resi(:,i)];
    % 실제 residue 는 resi * r 

    % state update
    Xp = [Xp,A*Xp(:,i) + B*U(:,i)];
    
    new_qXc = F_*qXc(:,i) + qG*qY(:,i) + qR*resi(:,i); % 다음 스텝의 qXc 계산
    qXc = [qXc, new_qXc]; % 업데이트된 qXc를 저장
    Xc = [Xc, r*s*new_qXc]; % 바로 저장된 qXc를 사용하여 Xc 계산

    % 차이 계산
    diff_u = [diff_u, u_(:,i) - U(:,i)];
    diff_Xc = [diff_Xc, x_c(:,i) - Xc(:,i)];

end

%% 여기까지 정수화로 변환한 친구랑 원래 시스템 플랏 비교하기
% Control Input Plot
figure(2)
plot(Ts*(0:iter-1), u) % Original control input
hold on
plot(Ts*(0:iter-1), U) % Quantized control input
title('Control Input (u)', 'FontSize', 14)
legend('Original Controller', 'Quantized Controller', 'FontSize', 12)
grid on

% Plant Output Plot
figure(1)
plot(Ts*(0:iter-1), y(1,:)) % x position from original
hold on
plot(Ts*(0:iter-1), Y(1,:)) % x position from quantized
plot(Ts*(0:iter-1), y(2,:)) % phi angle from original
plot(Ts*(0:iter-1), Y(2,:)) % phi angle from quantized
title('Plant Output (y)', 'FontSize', 14)
legend('x position (Original)', 'x position (Quantized)', ...
       'phi angle (Original)', 'phi angle (Quantized)', 'FontSize', 12)
% grid on
% 
% figure(3)
% plot(Ts*(0:iter-1), diff_u)
% 

grid on

