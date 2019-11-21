function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

print("lambda:")
factor = lambda/(2*m);

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


# do not use the first theta θ0
J = ((1/m) * (-y'*log(sigmoid(X*theta)) - (1-y)'*log(1-(sigmoid(X*theta))))) + (lambda/(2*m))*sum(theta(2:length(theta)).^2);

theta_for_gradient = theta;
theta_for_gradient(1)  = 0; #do not regularize the first theta θ0.
grad = (1/m) * X' * (sigmoid(X*theta)-y) + (lambda/m)*theta_for_gradient; #'

#もうひとつのimplementation ＝＞これは動かない。。。負の値になってしまう。講義ビデオで紹介されたのはなんで？？
#＝＞講義ビデオでは theta =  theta(1-(lambda/m)) - J(θ)となっているが、このプログラムコードでは
# gradientのみをreturnするという設計なので、ここでは機能しないのではないかと思われる。
#(theta_for_gradient*(1-(lambda/m)))
#grad = (theta*(1-(lambda/m))) - ((1/m) * X' * (sigmoid(X*theta)-y));


%Cost at initial theta (zeros): 0.693147
%Expected cost (approx): 0.693
%Gradient at initial theta (zeros) - first five values only:
% -0.008475
% -0.018788
 %-0.000078
% -0.050345
% -0.011501
%Expected gradients (approx) - first five values only:
% 0.0085
% 0.0188
% 0.0001
% 0.0503
% 0.0115
%
%Program paused. Press enter to continue.
%
%Cost at test theta (with lambda = 10): 3.164509
%Expected cost (approx): 3.16
%Gradient at test theta - first five values only:
% -0.346045
% 0.838648
% 0.805204
% 0.773137
% 0.907814
%Expected gradients (approx) - first five values only:
% 0.3460
% 0.1614
% 0.1948
% 0.2269
% 0.0922

%Program paused. Press enter to continue.
%Train Accuracy: 49.152542
%Expected accuracy (with lambda = 1): 83.1 (approx)

% =============================================================



end
