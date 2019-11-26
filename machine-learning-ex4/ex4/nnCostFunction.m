function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% NNCOSTFUNCTION Implements the neural network cost function for a two layer
% neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

#size(Theta1)
#size(Theta2)

% Setup some useful variables
m = size(X, 1); # 5000 exampls

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


% X          5000 x 400
% y          5000 x 1
% Theta1    25 * 401 => 401 input, 25 output
% Theta2    10 * 26 => 26 input, 10 output

% まずはyをYにマッピングする -> って、それは一体何なん？なんで必要なん？
% =>あー、なるほど！！！　one-hotエンコーディングの形にする！！
y_one_hot = zeros(m, num_labels); %m:5000 x num_labels:10
for i=1:m %
  y_one_hot(i, y(i)) = 1;
end

a1 = [ones(m,1) X]; %5000 examples x 401features(pixels)(davon 1 bias) =>X transpose-> 401 features x 5000 exmaple
z2 = a1 * Theta1';   % a1: 5000 x 401, Theta1': 25 x 401　＝＞5000 x 25
a2 = [ones(m,1) sigmoid(z2)]; % 500 x 401 (davon 1 bias)

z3 = a2 * Theta2'; #' % a2: 5000 x 26, Theta2: 26 x 10 => z2: 5000 x 10
h = sigmoid(z3); % 5000 x 10

% sum(a,1); # 列ごとの合計
% sum(a,2); # 行ごとの合計
% https://www.coursera.org/learn/machine-learning/supplement/afqGa/cost-function
% うまく行った！感動！スライドの式はわかりにくい！
% まず、行単位でcostを算出し、全行を最後に合計するだけ
J = -(1/m) * sum(sum( (y_one_hot.*log(h) + (1-y_one_hot).*log(1-h)) ,2),1);

% preparation for regulalization
Theta1_reg = Theta1(:,2:end);
Theta2_reg = Theta2(:,2:end);

% regularization term
regularization = lambda/(2*m) * ( sum(sum(Theta1_reg.^2)) + sum(sum(Theta2_reg.^2)) );

% add regularization
J = J + regularization


% ==============================================================================
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function. (このことをPart1で説明しろよ！)
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%

%% forward propagation
a1 = [ones(m,1) X];             % 5000 examples x 401features(pixels)(davon 1 bias))
z2 = a1 * Theta1';              % a1: 5000 x 401, Theta1': 25 x 401　＝＞5000 x 25
a2 = [ones(m,1) sigmoid(z2)];   % 5000 x 26 (davon 1 bias)

z3 = a2 * Theta2'; #' % a2: 5000 x 26, Theta2: 26 x 10 => z2: 5000 x 10
h = sigmoid(z3);     % 5000 x 10

#size(a1)                  % 5000 x 401
#size(Theta2')             % 26 x 10
#size(Theta2(:,2:end)')    % 25 x 10
#size(sigmoidGradient(z2)) % 5000 x 25

% compute delta
delta3 = h - y_one_hot; % 5000 x 10
delta2 = (Theta2(:,2:end)' * delta3')' .* sigmoidGradient(z2); % 5000 x 25 #'

%% accumulate the gradients
## => うーん、ｐｄｆでは、aの方をtransposeしてはいるが。
DELTA2 = delta3' * a2;  % delta': 10 x 5000, a2: 5000 x 26 => 10 x 26
DELTA1 = delta2' * a1;  % delta2': 25 x 5000, a1: 5000 x 401 => 25 x 401

Theta2_grad = (1/m)*DELTA2; % 10 x 26
Theta1_grad = (1/m)*DELTA1; % 25 x 401


% =================================================================================
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------

## since I don't use the first columns in theta, replace their elements with 0. #'
Theta2(:,1) = 0;
Theta1(:,1) = 0;

## update gradients with regularization terms
Theta2_grad = Theta2_grad + (lambda/m) * Theta2;
Theta1_grad = Theta1_grad + (lambda/m) * Theta1;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
