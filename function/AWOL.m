function B_tilde = AWOL(y,x,sigma,P_0)
T = length(y);
d = size(x,2);
Omega = zeros(d*(T+1),d*(T+1));
c = zeros(d*(T+1),1);
Omega(1:d,1:d) = inv(P_0) + eye(d);
Omega(1:d,d+1:2*d) = eye(d);
B_tilde = zeros(T+1,d);
for t = 1:T-1

    Omega(t*d+1:(t+1)*d,(t-1)*d+1:t*d) = eye(d);    
    Omega(t*d+1:(t+1)*d,(t)*d+1:(t+1)*d) = x(t)'*x(t)/sigma(t)+2*eye(d);
    Omega(t*d+1:(t+1)*d,(t+1)*d+1:(t+2)*d) = eye(d);
    
    c(t*d+1:(t+1)*d,1) = (x(t)'/sigma(t))*y(t);
end

Omega(end-d+1:end,end-2*d+1:end-d) = eye(d);
Omega(end-d+1:end,end-d+1:end) = x(T)'*x(T)/sigma(T)+eye(d);

b_var = inv(Omega);
b_mean = b_var*c;

B_tilde_temp = mvnrnd(b_mean,b_var);
B_tilde(1,:) = B_tilde_temp(1:d);
for t = 1:T
    B_tilde(t+1,:) = B_tilde_temp(t*d+1:(t+1)*d);
end
end