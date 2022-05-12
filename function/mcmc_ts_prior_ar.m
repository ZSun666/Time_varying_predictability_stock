function [aols,vbar,ssigmean,sigvar] = mcmc_ts_prior_ar(rawdat,tau,p,plag,const)

yt = rawdat(plag+1:tau+plag,:)';

%m is the number of elements in the state vector
if const==1 % including constants
    m = p + plag*(p^2);
    Zt=[];
    for i = plag+1:tau+plag
        ztemp = eye(p);
        for j = 1:plag;
            xlag = rawdat(i-j,1:p);
            xtemp = zeros(p,p*p);
            for jj = 1:p;
                xtemp(jj,(jj-1)*p+1:jj*p) = xlag;
            end
            ztemp = [ztemp   xtemp];
        end
        Zt = [Zt ; ztemp];
    end
else
    m = plag*(p^2);
    Zt=[];
    for i = plag+1:tau+plag
        ztemp=[];
        for j = 1:plag;
            xlag = rawdat(i-j,1:p);
            xtemp = zeros(p,p*p);
            for jj = 1:p;
                xtemp(jj,(jj-1)*p+1:jj*p) = xlag;
            end
            ztemp = [ztemp   xtemp];
        end
        Zt = [Zt ; ztemp];
    end
end;

vbar = zeros(m,m);
xhy = zeros(m,1);
for i = 1:tau
    zhat1 = Zt((i-1)*p+1:i*p,:);
    vbar = vbar + zhat1'*zhat1;
    xhy = xhy + zhat1'*yt(:,i);
end

vbar = inv(vbar);
aols = vbar*xhy;

sse2 = zeros(p,p);
for i = 1:tau
    zhat1 = Zt((i-1)*p+1:i*p,:);
    sse2 = sse2 + (yt(:,i) - zhat1*aols)*(yt(:,i) - zhat1*aols)';
end
hbar = sse2./tau;

    

    vbar = zeros(m,m);
    for i = 1:tau
        zhat1 = Zt((i-1)*p+1:i*p,:);
        vbar = vbar + zhat1'*inv(hbar)*zhat1;
    end
    vbar = inv(vbar);
    vbar = 0.5*(vbar'+vbar);

    ssig1 = log(hbar);


    hbar1 = inv(tau*hbar);
    hdraw = zeros(p,p);
  
    sigvar = zeros(p,p);
    ssigmean = zeros(p,1);
    
    for irep = 1:4000
        hdraw = wish(hbar1,tau);
        hdraw = inv(hdraw);
        achol = chol(hdraw)';
        ssig = zeros(p,p);
        for i = 1:p
            ssig(i,i) = achol(i,i); 
        end
        sigdraw = zeros(p,1);
        for i=1:p
            sigdraw(i,1) = log(ssig(i,i)^2);
        end

        sigvar = sigvar +sigdraw*sigdraw';
        ssigmean = ssigmean+sigdraw;
    end
  
    ssigmean = ssigmean./4000;
    sigvar = sigvar./4000;
    sigvar = sigvar - ssigmean*ssigmean';
   
end