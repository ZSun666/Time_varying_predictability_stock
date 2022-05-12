function[output] = weighted_prctile(X,percentile,weight)
ndim = size(X,2);
K = size(X,1);
t = size(X,3);
x_reshape = permute(X,[2,1,3]);
x_reshape = reshape(x_reshape,ndim,[]);
weight = kron(weight,ones(1,K));

ncol = size(x_reshape,2);
    for n = 1:ncol
        [sorted_x,index] = sort(x_reshape(:,n));
        sorted_weight = weight(index,n);
        pos = 0;
        index_count = 1;
 
        while pos <= percentile*0.01
            pos = pos+ sorted_weight(index_count);
            index_count = index_count +1;
        end

        output(n) = sorted_x(index_count-1);
    end
    output =reshape(output,1,K,t);
    output =permute(output,[2,3,1]);
    

end