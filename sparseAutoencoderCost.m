function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));   
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- Compute the cost/optimization objective J_sparse(W,b) --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
%m=10000 training samples
m = size(data,2);
%alpha is the learning rate
%alpha = 0.1;
J_sparse = 0;
%nl = 3;
%Delta W1, W2, b1, b2
DeltaW1 = zeros(size(W1)); 
DeltaW2 = zeros(size(W2));
Deltab1 = zeros(size(b1)); 
Deltab2 = zeros(size(b2));

% calculate the rho
rho = zeros(size(b1));
for i = 1:m
    %Perform a feedforward pass, computing the activations for layers L2,
    %L3, up to the output layer Lnl.
    a1 = data(:,i);
    z2 = W1*a1 + b1;
    a2 = sigmoid(z2);
    rho = rho +a2;
end
rho = rho./m;

%m iteration of batch gradient descent
for i = 1:m
    %Perform a feedforward pass, computing the activations for layers L2,
    %L3, up to the output layer Lnl.
    a1 = data(:,i);
    z2 = W1*a1 + b1;
    a2 = sigmoid(z2);
    z3 = W2*a2 + b2;
    a3 = sigmoid(z3);
    %calculate delta2, delta3
    dersigmoidz3 = a3.*(ones(size(a3))-a3);
    dersigmoidz2 = a2.*(ones(size(a2))-a2);
    delta3 = -1.*(a1-a3).*dersigmoidz3;
    delta2 = ((W2'*delta3) + beta.*(-1*sparsityParam./rho + (1-sparsityParam)./(ones(size(rho))-rho))).*dersigmoidz2;
    %Compute the desired partial derivatives
    graW2 = delta3*a2';
    grab2 = delta3;
    graW1 = delta2*a1';
    grab1 = delta2; 
    DeltaW1 = DeltaW1 + graW1; 
    DeltaW2 = DeltaW2 + graW2;
    Deltab1 = Deltab1 + grab1; 
    Deltab2 = Deltab2 + grab2;
    %The cost function of the single sample 
    JW = (a3 - a1)'*(a3 - a1);
    J_sparse = J_sparse + sum(JW(:))/2;

end

%The penalty term
KLdiv = KL(sparsityParam , rho);

%The cost function
W11 = W1.*W1;
W22 = W2.*W2; 
W = sum(W11(:)) + sum(W22(:));

J_sparse = J_sparse/m + (lambda/2)*W;
cost = J_sparse + beta*KLdiv;



%W1grad should be equal to the term [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] 
W1grad = (DeltaW1./m) + lambda*W1;
b1grad = Deltab1./m;
W2grad = (DeltaW2./m) + lambda*W2;
b2grad = Deltab2./m;
    
%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% our gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which we may find useful
% in our computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

%-------------------------------------------------------------------
% sum of all KL divergence

function KLdiv = KL(x,y)
    s2 = length(y);
    KLdiv = 0;
    for i = 1:s2
        KLdiv = KLdiv + x*log(x/y(i)) + (1-x)*log((1-x)/(1-y(i)));
    end
end

