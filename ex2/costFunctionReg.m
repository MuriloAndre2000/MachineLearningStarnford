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
h = sigmoid(X*theta);
%to grad, we need make a matrice with (1,1) = 0 
matricereduced = theta(2:size(theta))
%When you take off the (1,1), you eliminate the theta0, that we should don't use in the regularization. That's why i used the matricereduced in the cost(J)
matrice0 = [0;matricereduced]



J = (1/m)*(-y'* log(h) - (1 - y)'* log(1-h)) + (1/2)*(1/m)*lambda*(sum(matricereduced.^2))



grad = (1/m)*(X'*(h - y) + lambda*matrice0);




% =============================================================

end
