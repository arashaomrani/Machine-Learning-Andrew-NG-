function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);
J_history = zeros(num_iters, 1);

%X=[ones(length(X),1) X];

for iter = 1:num_iters
    h=X*theta;
    J1=(h-y);

    for j=1:n
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
    %       of the cost function (computeCost) and gradient here.
    %
    % ============================================================

    % Save the cost J in every iteration 
    J_history(iter) = computeCost(X, y, theta);

end

end
