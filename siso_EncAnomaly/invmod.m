function modm = invmod(m,env)
    % Translates m in {0,1,...,q-1} to [-q/2. q/2]
    modm = m - floor( (m + env.q/2)/env.q) * env.q;
end