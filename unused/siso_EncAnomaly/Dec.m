function m = Dec(c, env)
    % Modified decryption 
    % Eq. (28)
    m = mod(c*[1;-env.sk;1],env.q);
end
