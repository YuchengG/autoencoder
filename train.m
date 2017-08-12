%% Sparse autoencoder by guoyucheng
% Time : 2013/12/05
%%======================================================================
%% STEP 0: Give the relevant parameters values
clc;
clear;
patchsize = 8;  % we'll use 8x8 patches 
numpatches = 10000; % K training set 
visibleSize = patchsize*patchsize;   % number of input units 
hiddenSize = 25;     % number of hidden units 
sparsityParam = 0.01;   % desired average activation of the hidden units.
lambda = 0.0001;     % weight decay parameter  lambda = 0.0001;     
beta = 3;            % weight of sparsity penalty term  beta = 3;      
%%======================================================================
%% STEP 1: Implement sampleIMAGES
%
%Implementing sampleIMAGES
patches = sampleIMAGES(patchsize, numpatches);

%display a random sample of 200 patches from the dataset
display_network(patches(:,randi(size(patches,2),200,1)),8);

%  Obtain random parameters theta=(W(1),W(2),b(1),b(2))
theta = initializeParameters(hiddenSize, visibleSize);

%%======================================================================
%% STEP 2: Implement sparseAutoencoderCost
%
%  (a) Implement forward propagation in your neural network, and implement the 
%      squared error term of the cost function.  Implement backpropagation to 
%      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking 
%      to verify that the calculations corresponding to the squared error cost 
%      term are correct.
%
%  (b) Add in the weight decay term (in both the cost function and the derivative
%      calculations), then re-run Gradient Checking to verify correctness. 
%
%  (c) Add in the sparsity penalty term, then re-run Gradient Checking to 
%      verify correctness.
%

[cost, grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, ...
                                     sparsityParam, beta , patches);

%%======================================================================
%% STEP 3: Gradient Checking
%
% First, lets make sure our numerical gradient computation is correct for a
% simple function.  After we have implemented computeNumericalGradient.m,
% run the following: 
checkNumericalGradient();

% Now we can use it to check the cost function and derivative calculations
% for the sparse autoencoder.  
numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, visibleSize, ...
                                                  hiddenSize, lambda, ...
                                                  sparsityParam, beta, ...
                                                  patches), theta);

% Use this to visually compare the gradients side by side
disp([numgrad grad]); 

% Compare numerically computed gradients with the ones obtained from backpropagation
diff = norm(numgrad-grad)/norm(numgrad+grad);
disp(diff); % Should be small. These values are usually less than 1e-9.

%%======================================================================
%% STEP 4: After verifying that our implementation 
%  of sparseAutoencoderCost is correct, We can start training our sparse
%  autoencoder with minFunc (L-BFGS).

%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, we
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);

%%======================================================================
%% STEP 5: Visualization 

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file 


