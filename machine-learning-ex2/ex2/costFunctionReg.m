function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
[m,n] = size(X); % rows and cols

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z= X*theta;
h= sigmoid(z);

% Costo
sumJ=0;
for i=1: m
    sumJ= sumJ+ (-y(i)*log(h(i)) -(1-y(i))*log(1-h(i)));
end;
sumJR=0;
for i=1: n
    if i>1
        sumJR= sumJR + theta(i)^2;
    else
        sumJR= sumJR;
end;

J= sumJ/m +  (lambda/(2*m))*sumJR;

% Gradiente
for j=1:length(theta)
    sumDJ=0;
    for i=1: m
        sumDJ= sumDJ + (h(i)-y(i))*X(i,j);
    end;
    if j>1
        grad(j)= (sumDJ/m) + (lambda/m)*theta(j);
    else
        grad(j)= (sumDJ/m);
    end;
end;





% =============================================================

end
