function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% Hypothesis for current X and theta
h = X * Theta';

% Hypothesis only for given ratings
h(R == 0) = 0;


%% ----------- Cost Function ------------------

% Error in our hypothetical model
J = (1/2) * sum(sum((h - Y) .^ 2));

% Adding regularization
regTheta = (lambda / 2) * sum(sum(Theta .^ 2));
regX = (lambda / 2) * sum(sum(X .^ 2));

J = J + regTheta + regX;


%% ---------------- Gradient ------------------

% Gradient of movie features
X_grad = (h - Y) * Theta;

% Gradient of user parameters or preferences
Theta_grad = (h - Y)' * X;

% Adding regularization
X_grad = X_grad + (lambda .* X);
Theta_grad = Theta_grad + (lambda .* Theta);


% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
