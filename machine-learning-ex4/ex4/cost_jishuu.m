# わかりやすいかも。
#https://vanhuyz.com/coursera-machinelearning-assignment-4/

% ====================== YOUR CODE HERE ======================


% X          5000 x 400
% y          5000 x 1
% nn_ params 10285 x 1
% Theta1       25 x 401
% Theta2       10 x 26

% まずはyをYにマッピングする
Y = zeros(m,num_ labels);  % 5000 x 10
for i = 1:m
  Y(i,y(i)) = 1;
end

% h_ thetaの計算。行列のサイズを確認しながら実装しましょう
A1 = [ones(m,1) X];             % 5000 x 401
Z2 = A1 * Theta1';              % 5000 x 25
A2 = [ones(m,1) sigmoid(Z2)];   % 5000 x 26
Z3 = A2 * Theta2';              % 5000 x 10
h = sigmoid(Z3);                % 5000 x 10

% 全部揃ったのでコスト関数をloopで計算
for i = 1:m  #m==5000
  for k = 1:num_ labels  #num_labes==10
    J += -Y(i,k) * log(h(i,k)) - (1-Y(i,k)) * log(1 - h(i,k));
  end
end

J = J/m;

% ==========================================
end
loopを使わないバージョン：

J = sum((-Y .* log(h) - (1-Y) .* log(1-h))(:));

Y,hはどれでも5000x10行列です。

それらをでかいベクトルに変換するとベクトル化もできます：

J = -Y(:)' * log(h(:)) - (1-Y(:))' * log(1-h(:));


% ======================
コスト関数を正規化
ロジスティック回帰と同様、各Thetaの一番目のコラムを排除しないといけません。Octaveで書くと、

J += (sum(Theta1(:,2:end)(:).^2) + sum(Theta2(:,2:end)(:).^2)) * lambda / (2*m) ;
ここで(:)は行列をでかいベクトルに変換します(sumを使うためです）。




ーーーーーーーーーーーーーーーーーーー

function [J grad] = nnCostFunction(nn_ params, ...
                                   input_ layer_ size, ...
                                   hidden_ layer_ size, ...
                                   num_ labels, ...
                                   X, y, lambda)

% feedforward
% 1.3のコードなので略

% backpropagation

delta3 = h - Y;                                            % 5000 x 10
delta2 = (delta3*Theta2(:,2:end)) .* sigmoidGradient(Z2);  % 5000 x 25

Delta1 = delta2' * A1; % 25 x 401
Delta2 = delta3' * A2; % 10 x 26

Theta1_ grad = Delta1/m;
Theta2_ grad = Delta2/m;

% Unroll gradients
grad = [Theta1_ grad(:) ; Theta2_ grad(:)];

end



==============================
