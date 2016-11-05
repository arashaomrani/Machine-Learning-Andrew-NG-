function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
% You need to return the following variables correctly 
J = 0;
h1=X*theta;
h=1./(1+exp(-1*h1));
grad = zeros(size(theta));


    for j=1:n
       delta=0;
      
        for i=1:m 
          delta1=(1/m)*(h(i)-y(i))*X(i,j);
          delta=delta1+delta;
        end
        grad(j)=grad(j)+delta;
    end
     alpha=0;
    for i= 1:m
        alpha1=(-1/m)*(y(i)*log(h(i))+(1-y(i))*log(1-h(i)));
        alpha=alpha+alpha1;  
    end
    J=alpha;
    

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
