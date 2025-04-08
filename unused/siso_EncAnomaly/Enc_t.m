function c = Enc_t(m, z, env)
    % Input
    % m : message (plant output)
    % z : state of zero-dynamics
    % Output
    % c : ciphertext
    % Eq. (27)
    
    n = size(m,1);

    % Generate random matrix in Z_q
    % Dim : n x N
    % * randi can generate at most 2^53-1... used it instead of env.q
    % Ay = mod(sym(randi(2^(53)-1, [n, env.N])), env.q); 
    Ay = mod(sym(randi(2^(3)-1, [n, env.N])), env.q);

    % Generate random error in Z_q
    % Normal distribution ~ N(0,sigma)
    % Dim : n x 1
    % TODO: Change to bounded discrete Gaussian. Currently, rounding continuous Gaussian 
    ey = mod(sym(round(normrnd(0,env.sigma,[n,1]))),env.q);
  
    % Compute masking part
    By = mod(Ay*env.sk + ey, env.q);
    By_ = mod(By + env.ginv * env.psi * z, env.q);

    % Compute ciphertext
    c = mod([m*env.L + By - By_, Ay, By_],env.q);
end

