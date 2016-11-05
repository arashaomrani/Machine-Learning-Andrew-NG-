function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n=size(X);
%X=[ones(length(X),1) X];

for iter = 1:num_iters

     h=X*theta;
    J1=(h-y);
   
    for j=1:(n(1,2))
            delta=0;
        for i=1:m 
          delta1=(1/m)*(h(i)-y(i))*X(i,j);
          delta=delta1+delta;
        end 
        theta(j)=theta(j)-alpha*delta;
    end
    
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %











    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end