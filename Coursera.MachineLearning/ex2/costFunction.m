function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% grad = zeros(size(theta));

h_x = 1 ./ (exp(-1 .* X * theta) +1);
h_x_T = transpose(h_x);

log_h_x_T = log(h_x_T);
log_1_h_x_T = log(1 - h_x_T);
J = log_h_x_T * y + log_1_h_x_T * (1 - y);
J = J / (-m);

% for j = 1:n
%     x_j = X(:,j);
%     grad(j) = transpose(h_x - y)*x_j;
% end

grad = transpose(h_x - y) * X;
grad = transpose(grad)./m;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
