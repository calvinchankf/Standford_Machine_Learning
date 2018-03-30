function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

% data = [1,2;2,4]
% m = length(data(:,1))
% x = data(:,1)
% X = [ones(m, 1), data(:,1)]
% y = data(:,2)
% theta = [0;2]

%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1); % previous cost junction [x1;x2...;xn]

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % vectorized form
    theta = theta - (alpha/m) * (X'*(X*theta-y));

    % or
    % lastTheta is the old one with previous computed value
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
    J_history(iter) = computeCost(X, y, theta);

end % end for

end
