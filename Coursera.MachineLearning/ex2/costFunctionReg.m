function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% J = 0;
% grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

modified_theta = theta;
modified_theta(1) = 0;

h_x = 1 ./ (exp(X * theta .* -1) +1);
h_x_T = transpose(h_x);

log_h_x_T = log(h_x_T);
log_1_h_x_T = log(1 - h_x_T);
J = log_h_x_T * y + log_1_h_x_T * (1 - y);
sum_theta_sq = lambda * (modified_theta'*modified_theta);
J = (0-J+sum_theta_sq/2) / m;

grad = transpose(h_x - y) * X;
grad = (transpose(grad) + lambda * modified_theta)./m;



% =============================================================

end
