function mbar = quant(m,s1,s2,env)
    % 1. Scale up by s1
    % 2. Round 
    % 3. Scale up by s2
    % 4. Modulo operation to {0,1,...,q-1}
    mbar = mod( s2*round(sym(m)*s1), env.q ) ;
end