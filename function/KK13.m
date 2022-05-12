function [draw,m,P,sigma] = KK13(x,y,m_last,P_last,sigma_last,lambda,kappa)

    K = size(x,2);
 
    bp = m_last; % predict bp based on bt
    Vp = (1/lambda)*P_last; % predict V based on Vt
    cfe = y - x*bp;   % conditional forecast error
 
    f = x*Vp*x' + sigma_last;    % variance of the conditional forecast error
    
    inv_f = x'/f;

  
    m = bp + Vp*inv_f*cfe; % update b based on bp
    
    P = Vp - Vp*inv_f*x*Vp; % update V based on Vp
    sigma = kappa*sigma_last + (1-kappa)*(y-x*m)^2;
    P = (P' + P) * 0.5;
    
    draw = m+sqrt(P)*randn;


end