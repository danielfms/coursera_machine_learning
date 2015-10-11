function [J, grad] = lrCostFunction(theta, X, y, lambda)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
z= X*theta;
h= sigmoid(z);
regTheta= theta(2:size(theta));
regJ= (lambda/(2*m)).*(regTheta'*regTheta);
Jtmp= (1/m).*(-y'*log(h) - (1-y)'*log(1-h)) + regJ;
% Costo
%sumJ=0;
%for i=1: m
%    sumJ= sumJ+ (-y(i)*log(h(i)) -(1-y(i))*log(1-h(i)));
%end;
J= sum(Jtmp);
regGrad= (lambda/m).*theta;
regGrad(1)=0.0;
grad= (1/m).*(X'*(h-y)) + regGrad;

% Gradiente
%for j=1:length(theta)
%    sumDJ=0;
%    for i=1: m
%        sumDJ= sumDJ + (h(i)-y(i))*X(i,j);
%    end;
%    grad(j)= (sumDJ/m);
%end;




% =============================================================

grad = grad(:);

end
