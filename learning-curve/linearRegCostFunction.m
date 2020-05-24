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
H=X*theta;
STDDEV = H-y;
VARI = STDDEV.^2;
thetarest=theta(2:length(theta),:);
regularization = ((lambda./(2*m)).*(sum(thetarest.^2)));
J=sum(VARI)/(2*m) + regularization;
grad = ((1./m).*(H-y)'*X) + ((lambda/m).*(theta'));
grad(1)=(1./m).*(H-y)'*X(:,1);

% =========================================================================

grad = grad(:);

end