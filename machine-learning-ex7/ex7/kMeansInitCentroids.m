function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

[y,idx] = datasample(1:size(X,1),K,'Replace',false);
y=y';
for k=1:K
for i=1:size(X,2)
   centroids(k,i)=X(y(k,1),i);
    
end
end




% =============================================================

end

