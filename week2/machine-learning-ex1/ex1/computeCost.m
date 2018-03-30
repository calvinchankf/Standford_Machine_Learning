function J = computeCost(X, y, theta)
% function J = computeCost()

% test y = 2x + 0, trainong = [1,2:2,4], Ans: J = 0
% data = [1,2;2,4;3,6]
% m = length(data(:,1))
% x = data(:,1)
% X = [ones(m, 1), data(:,1)]
% y = data(:,2)
% theta = [0;2]

m = length(y); % number of training examples

% vectorized form LOL andrew ng should have told me earlier
J = 1/(2*m)*sum((X*theta-y).^2);

% or
% inversedTheta = theta';
% inversedX = X';
%
% hypo = inversedTheta*inversedX
% dif = (hypo' - y);
% dif2 = dif.^2;
%
% J = 1/(2*m)*sum(dif2(:));

end
