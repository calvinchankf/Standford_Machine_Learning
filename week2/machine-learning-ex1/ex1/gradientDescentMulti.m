function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    theta = theta - (alpha/m) * (X'*(X*theta-y));

    % or
    % lastTheta = theta;
    % for jj = 1:length(lastTheta)
    %   % hypo: row
    %   hypo = lastTheta'*X';
    %   % diff: column
    %   dif = (hypo' - y);
    %   % x0,x1...xn column
    %   xj = X(:,jj);
    %   %1*n matrix * n*1 matrix => 1*1 matrix
    %   difx = dif'*xj;
    %
    %   theta(jj) = lastTheta(jj) - (alpha/m) * difx;
    % end

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
