function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% 1 line
J = 1/(2*m)*sum((X*theta-y).^2);

% rows to columns
% inversedTheta = theta';
%
% % to make it multipliable with theta
% inversedX = X';
%
% % hypothesis = matrix: (t0x0 t1x1 t2x2 ...tnxn)
% hypo = inversedTheta*inversedX;
%
% % vector - vector
% dif = (hypo' - y);
%
% % (h(x)-y)^2
% dif2 = dif.^2;
%
% J = 1/(2*m)*sum(dif2(:));

% =========================================================================

end
