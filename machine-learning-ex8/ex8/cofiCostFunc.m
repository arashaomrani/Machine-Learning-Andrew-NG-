function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));
[m n]=size(Y);
for i=1:m
    for j=1:n
       J1=(1/2)*(Theta(j,:)*X(i,:)'-Y(i,j))^2; 
       if R(i,j) == 1
          J=J1+J; 
       end   
    end
end
Reg=0;
for i=1:m
 K=length(X(i,:));
 for k=1:K
    Reg1=(lambda/2)*X(i,k)^2;
    Reg=Reg+Reg1;   
 end
end
for j=1:n
 K=length(Theta(j,:));
 for k=1:K
    Reg2=(lambda/2)*Theta(j,k)^2;
    Reg=Reg+Reg2;   
 end
end
J=J+Reg;

S=0;

for i=1:m
 K=length(X(i,:));
 for k=1:K
       for j=1:n
         S1=(Theta(j,:)*X(i,:)'-Y(i,j))*Theta(j,k);
         if R(i,j) == 1
         S=S+S1;
         end
       end
    X_grad(i,k)=S+(lambda)*X(i,k);
    S=0;  
 end
end
S=0;

for j=1:n
 K=length(Theta(j,:));
 for k=1:K
     for i=1:m
         S1=(Theta(j,:)*X(i,:)'-Y(i,j))*X(i,k);
        if R(i,j) == 1
         S=S+S1;
        end
     end
    Theta_grad(j,k)=S+(lambda)*Theta(j,k);
    S=0; 
 end
end


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
















% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
