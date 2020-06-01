function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_list = [.01 .03 .1 .3 1 3 10 30];
sigma_list = [.01 .03 .1 .3 1 3 10 30];
results = zeros(numel(C_list)*numel(sigma_list),3);

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
row=1;
for C_val = C_list
    for sigma_val = sigma_list
        model = svmTrain(X, y, C_val, @(x1, x2)gaussianKernel(x1, x2, sigma_val));
        predictions = svmPredict(model, Xval);
        error=mean(double(predictions ~= yval));
        results(row,:) = [C_val sigma_val error];
        row=row+1;
    end
end

[~,i] = min(results(:,3));
C=results(i,1);
sigma=results(i,2);

% =========================================================================

end