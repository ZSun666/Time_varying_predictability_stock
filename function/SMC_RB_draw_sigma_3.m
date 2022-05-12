function[state_sigma,lik] = SMC_RB_draw_sigma_3(Y,Z,state_last,Vt,W)

q_s     = [  0.00730;  0.10556;  0.00002; 0.04395; 0.34001; 0.24566;  0.25750]; % probabilities
m_s     = [-10.12999; -3.97281; -8.56686; 2.77786; 0.61942; 1.79518; -1.08819]; % means
u2_s    = [  5.79596;  2.61369;  5.17950; 0.16735; 0.64009; 0.34023;  1.26261]; % variances




%% Draw Indicator
    
    for j = 1:numel(m_s)
        temp1= (1/sqrt(2*pi*u2_s(j)))*exp(-.5*(((Y - state_last - m_s(j) + 1.2704)^2)/u2_s(j)));
        prw(j,1) = q_s(j,1)*temp1;
    end
    prw = prw./sum(prw);
    cprw = cumsum(prw);
    trand = rand(1,1);
    if trand < cprw(1,1); imix=1;
    elseif trand < cprw(2,1), imix=2;
    elseif trand < cprw(3,1), imix=3;
    elseif trand < cprw(4,1), imix=4;
    elseif trand < cprw(5,1), imix=5;
    elseif trand < cprw(6,1), imix=6;
    else imix=7;
    end
    statedraw=imix;  % this is a draw of the mixture component index
    


    vart = u2_s(imix);
    yss = Y - m_s(imix) + 1.2704;
   
    
    


    y = yss';
    mp = state_last; % predict bp based on bt
    Vp = Vt+W; % predict V based on Vt
    cfe = y - Z*mp;   % conditional forecast error
    
    Kg = Vp/(Vp+vart);

  
    mtt = mp +Kg*cfe; % update b based on bp
    
    Vtt = (1-Kg)*Vp; % update V based on Vp
    Vtt = (Vtt' + Vtt) * 0.5;
    
    
    if min(eig(Vtt)) >0
        state_sigma = mvnrnd(mtt,Vtt);
        lik = mvnpdf(state_sigma,mtt,Vtt);
    else
        state_sigma = mtt;
        lik = 1;
    end


end