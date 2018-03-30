function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

c_arr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_arr = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

g_err = 1;
for i = 1:size(c_arr, 2)
  for j = 1:size(sigma_arr, 2)
    model = svmTrain(X, y, c_arr(i), @(X, Xval) gaussianKernel(X, Xval, sigma_arr(j)));
    predictions = svmPredict(model, Xval);
    err = mean(double(predictions ~= yval));
    if (err < g_err)
      g_err = err;
      C = c_arr(i);
      sigma = sigma_arr(j);
    end
  end
end



% =========================================================================

end
