function[state_beta,btt,Vtt] = SMC_draw_beta(Y,Z,state_beta_last,Vt,Q,Ht)
    y = Y';
    bp = state_beta_last; % predict bp based on bt
    Vp = Vt + Q; % predict V based on Vt
    cfe = y - Z*bp;   % conditional forecast error
    f = Z*Vp*Z' + Ht;    % variance of the conditional forecast error
    
    inv_f = Z'/f;

  
    btt = bp + Vp*inv_f*cfe; % update b based on bp
    
    Vtt = Vp - Vp*inv_f*Z*Vp; % update V based on Vp
    Vtt = (Vtt' + Vtt) * 0.5;
    
    state_beta = mvnrnd(btt,Vtt);
    
end