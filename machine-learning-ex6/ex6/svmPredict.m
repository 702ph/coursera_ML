function pred = svmPredict(model, X)
%SVMPREDICT returns a vector of predictions using a trained SVM model
%(svmTrain).
%   pred = SVMPREDICT(model, X) returns a vector of predictions using a
%   trained SVM model (svmTrain). X is a mxn matrix where there each
%   example is a row. model is a svm model returned from svmTrain.
%   predictions pred is a m x 1 column of predictions of {0, 1} values.
%


#disp("hallo1 svmPredict.m")

% Check if we are getting a column vector, if so, then assume that we only
% need to do prediction for a single example
if (size(X, 2) == 1)
    % Examples should be in rows
    X = X';
end

#disp("hallo2 svmPredict.m")
% Dataset
m = size(X, 1);
p = zeros(m, 1);
pred = zeros(m, 1);


if strcmp(func2str(model.kernelFunction), 'linearKernel') #'
    % We can use the weights and bias directly if working with the
    % linear kernel
    p = X * model.w + model.b;
    #disp("hallo3 svmPredict.m")
elseif strfind(func2str(model.kernelFunction), 'gaussianKernel')
    % Vectorized RBF Kernel
    % This is equivalent to computing the kernel on every pair of examples
    X1 = sum(X.^2, 2);
    X2 = sum(model.X.^2, 2)';
    disp("hallo4 svmPredict.m")
    K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));

    disp("hallo4-1 svmPredict.m")
    K = model.kernelFunction(1, 0) .^ K;

    disp("hallo4-2 svmPredict.m")
    K = bsxfun(@times, model.y', K);

    disp("hallo4-3 svmPredict.m")
    K = bsxfun(@times, model.alphas', K);
    p = sum(K, 2);
else
    % Other Non-linear kernel
    #disp("hallo5 svmPredict.m")
    for i = 1:m
        prediction = 0;
        for j = 1:size(model.X, 1)
            prediction = prediction + ...
                model.alphas(j) * model.y(j) * ...
                model.kernelFunction(X(i,:)', model.X(j,:)');
        end
        p(i) = prediction + model.b;
    end
end

#disp("hallo6 svmPredict.m")
% Convert predictions into 0 / 1
pred(p >= 0) =  1;
pred(p <  0) =  0;


disp("size of pred:::::::::")
size(pred)


end
