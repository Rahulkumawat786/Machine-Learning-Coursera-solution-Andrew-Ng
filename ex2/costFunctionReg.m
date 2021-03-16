function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));
temptheta = theta;
temptheta(1,1)=0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J =  -(1/m)*(sum(y.*log(sigmoid(X*theta))+(1-y).*log(1-sigmoid(X*theta))))+(lambda/(2*m))*sum(temptheta.^2);  %%cost fiunction

grad = (1/m)*(sum((sigmoid(X*theta)-y).*X))' + (lambda/m).*temptheta;

% =============================================================

end
