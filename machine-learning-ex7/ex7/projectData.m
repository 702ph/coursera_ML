function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K
%               eigenvectors in U (first K columns).
%               For the i-th example X(i,:), the projection on to the k-th
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

K % 1
size(Z) % 50 x 1
size(X) % 50 x 2
X;

U_reduce = U(:, 1:K);  % 2 x 1

Z = X * U_reduce;   % (50 x 2) * (2*1) ⇒スライドではU_reduct'*Xだったが、それだとdimension不一致で動かない。　#'


% =============================================================

end
