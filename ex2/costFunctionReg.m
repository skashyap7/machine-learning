function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================





hypothesis = sigmoid(X*theta);
%fprintf(" Hypothesis is %f",hypothesis);
a = (-y')*log(hypothesis);
%fprintf(" A is %f",a);
b = (1-y)'*log(1-hypothesis);
%fprintf(" B is %f",b);
newtheta = [0 ; theta(2:size(theta))];
J =  (a - b)/m + (lambda*(sum(newtheta.^2))/(2*m));

%fprintf(" Cost = %f", J);
%fprintf(" Cost is %f", J);

reg_grad = lambda*newtheta/m;
grad = (X'*(hypothesis-y))/m  + (lambda/m).*newtheta;

end
