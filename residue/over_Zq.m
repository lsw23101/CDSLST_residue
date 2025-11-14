clear; clc; close all;

load('FGH_data.mat','F_bar','G_','H');   % F_bar, G_, H workspace에 로드


s = 10000;
q = ;

F_bar; % 이미 정수
H_bar = round(s * H); % H를 양자화
G_bar = round(s * G_);% G_를 양자화

isprime(q)

H1 = H_bar(1,:);
T2 = H1;

T = []

