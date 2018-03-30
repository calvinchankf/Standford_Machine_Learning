function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% steps
% 1.
%   z2 = Theta1 * a1
%   a2 = g(z2)
% 2.
%   z3 = Theta2 * a2
%   a3 = g(z3)

a1 = [ones(m, 1), X]; % 5000 * 401
z2 = a1*Theta1'; % 5000 * 25
a2 = sigmoid(z2); % 5000 * 25

a2 = [ones(m, 1), a2];
z3 = a2*Theta2'; % 5000 * 10
a3 = sigmoid(z3); % 5000 * 10. also known as hypothesis, h(Theta)

% y now is 5000 * 1
% [10;10;9;9.......0]
% y needs to be 5000 * 10
% e.g.
% [1 0 0 0 0 0 0 0 0 0]
% [1 0 0 0 0 0 0 0 0 0]
% ...until row = 5000
Y = zeros(m, num_labels);
for i = 1:m;
  temp = y(i);
  Y(i, y(i)) = 1;
end

% compute J. i.e. cost function
for k = 1:num_labels;
  logH = log(a3(:,k));
  log1minusH = log(1-a3(:,k));
  JJ = 1/m * sum(-Y(:,k).*logH-(1-Y(:,k)).*log1minusH);
  J = J + JJ;
end

% with regularization. No first term(bias)!!!
Theta1NoFirst = Theta1(:,2:end);
Theta2NoFirst = Theta2(:,2:end);
regularization = lambda/(2*m)*(sum(Theta1NoFirst(:).^2)+sum(Theta2NoFirst(:).^2));
J = J + regularization;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

triangle1 = zeros(size(Theta1));
triangle2 = zeros(size(Theta2));

% ===
% for-loop implementation
% ===
%
% for i = 1:m;
%
%   % make each input to be a column vector
%   a1 = [1; X(i,:)'];
%   z2 = Theta1*a1;
%   a2 = [1; sigmoid(z2)];
%
%   z3 = Theta2 * a2;
% 	a3 = sigmoid(z3);
%
%   % comput sigma
%   sigma3 = a3 - Y(i, :)';
%   sigma2 = (Theta2(:,2:end)' * sigma3).* sigmoidGradient(z2);
%
%   % comput triangles
%   triangle1 = triangle1 + (sigma2 * a1');
%   triangle2 = triangle2 + (sigma3 * a2');
%
% end

% ===
% vectorization implementation
%
% actually i think vectorization is easier to understand and implement
%
% ===

% comput sigma
sigma3 = a3 - Y;
sigma2 = (sigma3*Theta2).*sigmoidGradient([ones(m, 1), z2]);
sigma2 = sigma2(:,2:end);

% comput triangle
triangle2 = sigma3'*a2;
triangle1 = sigma2'*a1;

% pure theta gradient(without regularization)
Theta1_grad = (1/m) * triangle1; % 25  401
Theta2_grad = (1/m) * triangle2; % 10 * 26

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% for each computed Theta_grad(l,i,j) other than bias terms, add regularization
Theta1_grad(:, 2:end) += (lambda/m) * Theta1(:,2:end);
Theta2_grad(:, 2:end) += (lambda/m) * Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
