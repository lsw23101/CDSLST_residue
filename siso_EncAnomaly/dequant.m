function m = dequant(mbar,scale,env)
    % 1. 'invmod' translates mbar in {0,1,...,q-1} to [-q/2, q/2)
    % 2. Then, scale down
    m = invmod(mbar,env)/ scale;
end