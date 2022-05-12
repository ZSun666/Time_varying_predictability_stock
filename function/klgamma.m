function [kl] = klgamma(pa,pb,qa,qb)

n = max([size(pb,2) size(pa,2)]);

if size(pa,2) == 1, pa = pa*ones(1,n); end
if size(pb,2) == 1, pb = pb*ones(1,n); end
qa = qa*ones(1,n); qb = qb*ones(1,n);

kl = sum( pa.*log(pb)-gammaln(pa) ...
         -qa.*log(qb)+gammaln(qa) ...
	 +(pa-qa).*(psi(pa)-log(pb)) ...
	 -(pb-qb).*pa./pb ,2);
end