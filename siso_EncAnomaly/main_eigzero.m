%% Section 1: System description
clear all;
close all;
clc;

% Plant: Discretized two-cart system
k = 2; b = 0; m1 = 2; m2 = 0.05;
A0       = [0 1 0 0; -k/m1 -b/m1 k/m1 0; 0 0 0 1; k/m2 0 -k/m2 -b/m2];
B0       = [0; 1/m1; 0; 0];
% Sampling time for CT to DT
plant.Ts = 0.1;   
% CT to DT 
[A, B]   = c2d(A0,B0,plant.Ts);
% cart position is the output
C        = [0 0 1 0];  
D        = 0;
A        = round(A,4);
B        = round(B,4);A
n        = 4;

% Observer based controller design
Q         = eye(4);
R1        = 0.1;
[~, K, ~] = idare(A,B,Q,R1,[],[]);
K         = -K;
[~, L, ~] = idare(A.', C.', Q, R1, [], []);
L         = L.';
K        = round(K,4);
L        = round(L,4);

% [Test]
% Check that the eigenvalues are inside the unit circle
abs(eig(A+B*K))
abs(eig(A-L*C))

% Controller 
% x+ = (A+BK-LC)x + Ly
%  u = Kx 
%  r = y - Cx

% Convert state matrix into Nilpotent matrix
%
% Design R and coordinate transformation matrix T so that (z=T^{-1}x)
% z+ = T^{-1}(A+BK-LC+RC)Tz + T^{-1}(L-R)y + T^{-1}Rr
%  u = KTz 
%  r = y - CTz
% has integer state matrix (A+BK-LC+RC)
%
% 'place' does not assign redundant poles 
% Use observable canonical form since obsv(F,C) = 4 = full rank

S = A+B*K-L*C;
O = obsv(S, C);
invO = inv(O);
p = invO(:,end);
T = [p,S*p,S^2*p,S^3*p];
SS = T\S*T;
R_ = -T*SS(:,end); % cancelling out the last column => all poles = 0

F = round(T\(S+R_*C)*T)
G = T\(L-R_);
H = -C*T;
J = 1;
R = T\R_;
P = K*T;

iter = 150;
%% Section 2: Unencrypted controller
xp_0 = 3*(rand([n,1])-0.5);
xc_0 = 3*(rand([n,1])-0.5);
 
xp = xp_0;
xc = xc_0;

rUnenc = [];
xpUnenc = [xp];
xcUnenc = [xc];
uUnenc = [];
yUnenc = [];


for i = 1:iter
    % plant output
    y = C * xp;
    % controller output
    u = P * xc;
    r = H * xc + J * y;
    % plant update
    xp = A * xp + B * u;
    %controller update
    xc = F * xc + G * y + R * r;

    rUnenc  = [rUnenc, r];
    xpUnenc = [xpUnenc, xp];
    xcUnenc = [xcUnenc, xc];
    uUnenc  = [uUnenc, u];
    yUnenc  = [yUnenc, y];
end


%% Section 3: Quantization and encryption setting

% scale factors
r_ = 10^3;
L_ = 10^3;
s_ = 10^4;

% encryption parameters
% * 'sym' allows exact computation on bigInt
env.N     = 16; % Not secure, just for simulation
env.r     = sym(r_);
env.L     = sym(L_); 
env.s     = sym(s_); 
env.quint = uint64(576460752303415297);
env.q     = sym(env.quint);  
env.sk    = sym(randi(3,[env.N,1])-2); % secret key from ternary distribution
env.sigma = 3.2; % error standard deviation

rmF = sym ( mod(int64(F), env.q) );
rmG = sym ( mod(int64(round(G*s_)) ,env.q) );
rmH = sym ( mod(int64(round(H*s_)) , env.q) );
rmJ = sym ( mod(int64(round(J*s_^2)), env.q) );
rmR = sym ( mod(int64(round(R*s_)) , env.q) );
rmP = sym ( mod(int64(round(P*s_)) , env.q) );

% Finding modular multiplicative inverse -> use 'gcd'
% https://www.mathworks.com/matlabcentral/answers/81859-how-to-write-matlab-code-for-modular-multiplicative-inverse-in-cryptography
[~,rmJinv] = gcd(rmJ,env.q);

env.F1 = mod(rmF - rmG*rmJinv*rmH, env.q);
env.g = rmJ;
env.ginv = rmJinv;
env.psi = rmH;
env.T1 = sym(eye(n));


%% Section 4: Quantized controller
% same initial state
xp = xp_0;
% scale and convert initial state
rmx = quant(xc_0, r_, s_, env);

rInt = [];
xpInt = [xp];
xcInt = [xc_0];
uInt = [];
yInt = [];

for i = 1:iter
    % plant output
    y = C * xp;
    rmy = quant(y, r_, 1, env);

    % controller output
    rmu = mod(rmP*rmx,env.q);
    rmr = mod(rmH*rmx + rmJ*rmy, env.q);

    u = dequant(rmu,r_*s_^2,env);
    r = dequant(rmr, r_*s_^2,env);

    % plant update
    xp = A*xp + B*u;

    %controller update
    roundRmr = quant(r,r_,1,env);
    rmx = mod(rmF*rmx+rmG*rmy+rmR*roundRmr,env.q);
    
    % dequantize
    xc = dequant(rmx,r_*s_,env);
  
    rInt  = [rInt, r];
    xpInt = [xpInt, xp];
    xcInt = [xcInt, xc];
    uInt  = [uInt, u];
    yInt  = [yInt, y];
end

%% Section 5: Encrypted controller
clc;

xp = xp_0;
rmx = quant(xc_0, r_, s_, env);
[xx,Bx] = Enc_0(rmx,env);

% zero dynamics
z = mod(env.T1*Bx,env.q);

rEnc = [];
xpEnc = [xp];
xcEnc = [xc_0];
uEnc = [];
yEnc = [];
zEnc = [z];
rDiff = [];


for i = 1:iter
    % plant output
    y = sym(C * xp);
    rmy = quant(y, r_, 1, env);
    yy = Enc_t(rmy,z,env);
    
    % controller output
    uu = mod(rmP*xx, env.q);
    rr = mod(rmH*xx + rmJ*yy, env.q);
    
    u  = dequant(Dec(uu,env),r_*s_^2*L_,env);
    r1 = dequant(rr(:,1),r_*s_^2*L_,env);
    
    % !!!!
    % 논문의 38번 식 위에처럼 S^2을 곱해서 scale을 바꾸면 안 됨.. 
    % Z_q = [-q/2,q/2)이면 가능한 방법이지만 Z_q={0,1,...,q-1}일 때는 
    % 1. [-q/2,q/2)로 translate
    % 2. scale 조정
    % 3. 다시 {0,1,...,q-1}로 translate
    
    % fed back input   
    rmr1 = mod([quant(r1,r_,L_,env), zeros(1,env.N+1)],env.q);

    % plant update
    xp = A*xp + B*u;

    %controller update
    xx = mod(rmF*xx + rmG*yy+rmR*rmr1,env.q);
    valx = dequant(Dec(xx,env),r_*s_*L_,env);

    % zero dynamics update
    z = mod(env.F1*z,env.q);

    rEnc  = [rEnc, r1];
    uEnc  = [uEnc, u];
    yEnc  = [yEnc, y];
    xcEnc = [xcEnc,valx];
    zEnc  = [zEnc,z];
end


%% plot

close all;

color1 = [0 0.4470 0.7410];
color2 = [0.9290 0.6940 0.1250];
color3 = [0.1290 0.2940 0.5250];
Linewidth = 3;
Linewidth2 = 2;
markerSize   = 8;
markerIndice = 40;
fontSize = 24;


dt = 1:iter;
figure(1);
subplot(2,2,[1,2]);
plot(1:size(rUnenc,2),rUnenc,'r--','LineWidth',Linewidth);
hold on;
plot(1:size(rInt,2),rInt,'-', 'color',color1,'LineWidth',Linewidth2);
hold on;
plot(1:size(rEnc,2),rEnc,'-.', 'color',color2,'LineWidth',Linewidth2);
legend('Controller over real', 'Quantized controller','Encrypted Controller', 'FontSize', fontSize );
title('$r$','interpreter','latex');
fontsize(fontSize,"points")

subplot(2,2,3);
plot(1:size(uUnenc,2),uUnenc,'r--','LineWidth',Linewidth);
hold on;
plot(1:size(uInt,2),uInt,'-', 'color',color1,'LineWidth',Linewidth2);
hold on;
plot(1:size(uEnc,2),uEnc,'-.', 'color',color2,'LineWidth',Linewidth2);
title('$u$','interpreter','latex');
fontsize(fontSize,"points")
subplot(2,2,4);
plot(1:size(yUnenc,2),yUnenc,'r--','LineWidth',Linewidth);
hold on;
plot(1:size(yInt,2),yInt,'-', 'color',color1,'LineWidth',Linewidth2);
hold on;
plot(1:size(yEnc,2),yEnc,'-.', 'color',color2,'LineWidth',Linewidth2);
title('$y$','interpreter','latex');
fontsize(fontSize,"points")
figure(2);
subplot(221);
plot(1:size(xcUnenc,2),xcUnenc(1,:),'r--','LineWidth',Linewidth);
hold on;
plot(1:size(xcInt,2),xcInt(1,:),'-', 'color',color1,'LineWidth',Linewidth2);
hold on;
plot(1:size(xcEnc,2),xcEnc(1,:),'-.', 'color',color2,'LineWidth',Linewidth2);
title('$r$','interpreter','latex');

subplot(222);
plot(1:size(xcUnenc,2),xcUnenc(2,:),'r--','LineWidth',Linewidth);
hold on;
plot(1:size(xcInt,2),xcInt(2,:),'-', 'color',color1,'LineWidth',Linewidth2);
hold on;
plot(1:size(xcEnc,2),xcEnc(2,:),'-.', 'color',color2,'LineWidth',Linewidth2);
title('$r$','interpreter','latex');

subplot(223);
plot(1:size(xcUnenc,2),xcUnenc(3,:),'r--','LineWidth',Linewidth);
hold on;
plot(1:size(xcInt,2),xcInt(3,:),'-', 'color',color1,'LineWidth',Linewidth2);
hold on;
plot(1:size(xcEnc,2),xcEnc(3,:),'-.', 'color',color2,'LineWidth',Linewidth2);
title('$r$','interpreter','latex');

subplot(224);
plot(1:size(xcUnenc,2),xcUnenc(4,:),'r--','LineWidth',Linewidth);
hold on;
plot(1:size(xcInt,2),xcInt(4,:),'-', 'color',color1,'LineWidth',Linewidth2);
hold on;
plot(1:size(xcEnc,2),xcEnc(4,:),'-.', 'color',color2,'LineWidth',Linewidth2);
legend('Controller over real', 'Quantized controller','Encrypted Controller', 'FontSize', fontSize );
title('$r$','interpreter','latex');

