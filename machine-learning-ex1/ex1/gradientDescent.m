function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

#debug
#theta
#num_iters = 3

rate = alpha * (1/length(X));

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    #最初の試み。これだとtheta1とtheta2に対し全く同じ値で更新されてしまう。。。
    #theta = theta - (alpha*(1/m) .* sum((X*theta)-y));

    #これがベクトル化されたもの
    #theta = theta - (alpha/m) .* (X' * (X * theta - y)); #'
    #Theta found by gradient descent:
    #-3.630291
    #1.166362
    #Expected theta values (approx)
    # -3.6303
    #  1.1664

    # ＝＞上記をを分解してみた
    #hypothese = X * theta;
    #diff = hypothese - y;
    #in_sigma = X' * diff; #'
    #delta = (alpha/m) .* in_sigma;
    #theta = theta - delta;
    #＝＞これも分解前の式と同じ結果を出力する！！
    #Theta found by gradient descent:
    #-3.630291
    #1.166362

    #これは自分でやり遂げた！
    #theta(1,1) = theta(1,1) - rate * sum( ((X*theta)-y).*X(:,1) );
    #theta(2,1) = theta(2,1) - rate * sum( ((X*theta)-y).*X(:,2) );

    #theta(1,1) = theta(1,1) - (alpha * (1/m)) * sum( ((X*theta)-y).*X(:,1) );
    #theta(2,1) = theta(2,1) - (alpha * (1/m)) * sum( ((X*theta)-y).*X(:,2) );
    #Theta found by gradient descent:
    #-3.636063
    #1.166989

    theta1 = theta(1) - alpha * (1 / m) * sum(((X * theta) - y) .* X(:, 1));
    theta2 = theta(2) - alpha * (1 / m) * sum(((X * theta) - y) .* X(:, 2));
    #theta(1) = theta1
    #theta(2) = theta2
    theta = [theta1; theta2];
    # ＝＞まじかよ？　望んでいた値になった。なんで？
    #＝＞あ、theta2の式で、更新されたtheta1の値が使われてしまってるからか！！
    # ＝＞頭では理解していたつもりだったが、thetaが同時に更新されていなかった。
    #Theta found by gradient descent:
    #-3.630291
    #1.166362

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

J_history
plot(J_history)
#xlabel('iterations'); ylabel('J');

end
