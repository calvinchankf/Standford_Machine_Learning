function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% compute J
dif = (X*theta-y);
J = 1/(2*m) * sum(dif.^2) + lambda/(2*m) * sum(theta(2:end,:).^2);

% compute gradient
for idx = 1:size(theta)

  % intuitive
  % g = 1/m* sum(X(:,idx)'*dif);
  % if (idx > 1)
  %   g += lambda/m*theta(idx);
  % end
  % grad(idx) = g;

  % one line
  grad(idx) = 1/m* sum(X(:,idx)'*dif) + (idx > 1) * lambda/m*theta(idx);

end



% =========================================================================

grad = grad(:);

end
