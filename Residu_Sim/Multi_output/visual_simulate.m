% 카트-폴 시스템 애니메이션
clc; clear;

% 시뮬레이션 데이터 로드
% 'conversion.m' 스크립트를 실행하여 'Xp'와 'Ts' 변수를 워크스페이스에 로드합니다.
run('conversion.m');

% 카트-폴 시스템 파라미터
cart_width = 0.4; % 카트의 너비 (단위: m)
cart_height = 0.2; % 카트의 높이 (단위: m)
pole_length = 1; % 폴의 길이 (단위: m)

% 시뮬레이션 데이터에서 카트 위치와 폴 각도 추출
x = Xp(1, :); % 카트의 위치
phi = Xp(3, :); % 폴의 각도

% 애니메이션 설정
figure;
hold on;
grid on;
axis equal;
xlim([-2 2]);
ylim([-0.5 1.5]);
xlabel('위치 (m)', 'FontSize', 12);
ylabel('높이 (m)', 'FontSize', 12);
title('카트-폴 시스템 애니메이션', 'FontSize', 14);

% 카트와 폴 그리기
cart = rectangle('Position', [x(1)-cart_width/2, 0, cart_width, cart_height], 'FaceColor', [0 0.5 1]);
pole = plot([x(1), x(1) + pole_length*sin(phi(1))], ...
            [cart_height/2, cart_height/2 + pole_length*cos(phi(1))], ...
            'LineWidth', 2, 'Color', [1 0 0]);

% 애니메이션 루프
for k = 1:length(x)
    % 카트 위치 업데이트
    set(cart, 'Position', [x(k)-cart_width/2, 0, cart_width, cart_height]);
    
    % 폴 위치 업데이트
    set(pole, 'XData', [x(k), x(k) + pole_length*sin(phi(k))]);
    set(pole, 'YData', [cart_height/2, cart_height/2 + pole_length*cos(phi(k))]);
    
    % 애니메이션 속도 조절
    pause(50*Ts);
end
