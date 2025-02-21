function [c, Bx] = Enc_0(m,env)
    % Input
    % m : message (initial condition)
    % Output
    % c : ciphertext
    % Bx: masking part for running zero-dynamics
    % Eq. (27)
    
    n = size(m,1);

    % Generate random matrix in Z_q
    % Dim : n x N
    % * randi can generate at most 2^53-1... used it instead of env.q
    % Ax = mod(sym(randi(2^(53)-1, [n, env.N])), env.q); 
    Ax = mod(sym(randi(2^(3)-1, [n, env.N])), env.q); 
    
    % Generate random error in Z_q
    % Normal distribution ~ N(0,sigma)
    % Dim : n x 1
    % TODO: Change to bounded discrete Gaussian. Currently, rounding continuous Gaussian 
    ex = mod(sym(round(normrnd(0,env.sigma,[n,1]))),env.q);
    
    % Compute masking part
    Bx = mod(Ax*env.sk + ex, env.q);

    % Compute ciphertext
    % * This is relative degree = 0 case, that is 
    % T_1 = I_n
    % T_2 = 0
    % => V_2T_2B_x = 0
    c = mod([m*env.L + Bx, Ax, zeros(n,1)],env.q);
end