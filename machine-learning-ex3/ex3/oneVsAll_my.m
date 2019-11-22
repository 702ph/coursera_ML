function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds
%   to the classifier for label i

# num_labesはクラスの数が入っている。ここでは 10。

% Some useful variables
m = size(X, 1);  # 5000 training examples トレーニングデータ
n = size(X, 2);  # 400 = 20x20 のピクセルデータ。400のfeaturesである。

% You need to return the following variables correctly
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda.
# 和訳：コードを完成せて num_labels個のLogistic Regression 分類機を完成させよ。その際、regularizationのパラメータにはlambdaを用いよ。

%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
%
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

iterations = 5000;
for label = 1:num_labels
#for label = 1:1 #debug用。ループ１回
  theta = ones(n+1,1);
  y_binarized = (y==label);

  for iteration = 1:iterations
  theta_for_grad = [0; theta(2:end,:)];
  grad =  (lambda/m)*theta_for_grad + (1/m) * X' * (sigmoid(X*theta)-y_binarized);
  theta = theta - grad;
  end

  all_theta(label,:) = theta'; #'
end #'


disp("debug: all_theta")
all_theta

% =========================================================================


end
