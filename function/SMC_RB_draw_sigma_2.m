function[state_sigma,lik] = SMC_RB_draw_sigma_2(Y,Z,state_last,Vt,W)
    y = Y';
    mp = state_last; % predict bp based on bt
    Vp = Vt+W; % predict V based on Vt
    cfe = y - Z*mp;   % conditional forecast error
    
    Kg = Vp/(Vp+4.94);

  
    mtt = mp +Kg*cfe; % update b based on bp
    
    Vtt = (1-Kg)*Vp; % update V based on Vp
    Vtt = (Vtt' + Vtt) * 0.5;
    
    sigma_mean = mtt;
    if min(eig(Vtt)) >0
        state_sigma = mvnrnd(mtt,Vtt);
        lik = mvnpdf(state_sigma,mtt,Vtt);
    else
        state_sigma = mtt;
        lik = 1;
    end


end