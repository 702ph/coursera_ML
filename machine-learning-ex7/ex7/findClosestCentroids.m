function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

disp("size of X:")
size(X) % 300 x 2
#plot(X,"xr")

% Set K
K = size(centroids, 1); % K == 3
centroids

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


for x_index = 1:length(X)

  distance = Inf;  % set distance a large value
  for k_index = 1:K

    % ユーグリッド距離を参照し、これをそのまま実装した
    % https://ja.wikipedia.org/wiki/%E3%83%A6%E3%83%BC%E3%82%AF%E3%83%AA%E3%83%83%E3%83%89%E8%B7%9D%E9%9B%A2
    new_distance = sqrt( sum((X(x_index,:)-centroids(k_index,:)).^2) ); % calculate distance with centroids
    if  (new_distance < distance)
      distance = new_distance;  % update distance for the next iteration
      idx(x_index) = k_index;   % save the index of the closest centroid
      #break;
    endif

  endfor
endfor




% =============================================================

end
