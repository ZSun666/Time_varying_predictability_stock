function[state_beta,btt,Vtt] = SMC_RB_draw_beta(Y,Z,state_beta_last,Vt,Q,Ht)
    y = Y';
    bp = state_beta_last; % predict bp based on bt
    Vp = Vt + Q; % predict V based on Vt
    cfe = y - Z*bp;   % conditional forecast error
    f = Z*Vp*Z' + Ht;    % variance of the conditional forecast error
    
    inv_f = Z'/f;

  
    btt = bp + Vp*inv_f*cfe; % update b based on bp
    
    Vtt = Vp - Vp*inv_f*Z*Vp; % update V based on Vp
    Vtt = (Vtt' + Vtt) * 0.5;
        [L,D] = ldl(Vtt);
    d = diag(D);
    if min(d) > 0
    root_D = diag(sqrt(d));
    s_temp = L * root_D; 
    else
    index_temp = find(d <= 0);
    d(index_temp) = 0 ;
    root_D = diag(sqrt(d));
    s_temp = L * root_D;
    end

    if isreal(s_temp) == 0
    disp("t");
    end
    state_beta = btt + s_temp * randn(length(btt),1);
    
end