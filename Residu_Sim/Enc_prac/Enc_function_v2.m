clear

% p나 L을 키우면 왜 오류가 생기지 >> randi 함수에서 오류가 난것으로 보임
% p*L=q 사이즈가 10^16
% k=20, m=30

% L은 K * e 보다 큰 값
% p는 K * m 보다 큰 값

env.p = 1e6; % 여기서 env라는 구조체에 p L r N 이라는 필드, 변수를 그룹화
env.L = 1e3; 
env.r = 10;
env.N = 4;

% 모듈러값의 범위를 0~p-1 >> -p/2 ~ p/2 로 조절, biased moduler
function y = Mod(x,p)
y = mod(x,p);
y = y - (y>=p/2)*p;
end


sk = Mod(randi(env.p*env.L, [env.N,1]), env.p*env.L); 

%  Functions Enc, Dec

function ciphertext = Enc(m,sk,env)
n=length(m); % 메세지 길이
q= env.L*env.p % 암호문 크기
A = randi(q, [n,env.N]) % 공개키 % randi 는 1~q 값 안에 [n x N] 크기의 정수 행렬 생성
% e = Mod(randi(env.r, [n,1]), env.r) % 에러 주입
e = 2
b = -A*sk + env.L*m + e % 마스크된 메시지
mask = b*env.L*m
ciphertext = Mod([b,A],q); % Enc 함수의 리턴값
end

function plaintext = Dec(c,sk,env)
s = [1; sk]; % 복호화 연산
plaintext = round(Mod(c*s, env.L*env.p)/env.L);
end

% example of encryption and decryption
sk
c = Enc(30,sk,env)
m = Dec(c,sk,env)

mult_c = 20 *c % 여기서 스칼라 곱이 문제가 생기는경우는...

mult_m = Dec(mult_c,sk,env)