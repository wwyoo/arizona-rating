%Author: William Weimin Yoo
%Predict user ratings of businesses by matrix completion with nuclear
%norm regularization

%add these two folders in search path
%this is for POSIX compliant systems, for windows, use "\" instead of "/"
addpath('Matlab_files/')
addpath('PROPACK_utils/')

%Description:
%The files in Matlab_files contain functions to do matrix completion
%by nuclear norm regularization.
%The algorithm used is the soft-impute algorithm proposed by Mazumder et.al
%(2010).
%This algorithm uses Lanczos bidiagonalization with partial 
%reorthogonalization to compute largest singular values during 
%soft-thresholding
%The folder PROPACK_utils contains the PROPACK Fortran source codes from
%http://soi.stanford.edu/~rmunk/PROPACK/
%Run the following matlab file to run the mex wrapper files and create
%mex binaries for calling within MATLAB

install_mex;

%Use gcc in Linux, will give warning saying for version discrepancy
%Please ignore them, everything will be compiled... 

%Both Matlab_files and PROPACK_utils obtained and modified 
%from Mazumder's website 
%http://www-stat.stanford.edu/~rahulm/software.html

%load data preprocessed from R
load('reviewdat.mat');

n=max(reviewdat(:,1));  %users (row)
m=max(reviewdat(:,2));  %business (column)
r=20;  %number of iterations
K=20;  %path length
Q=10000; %validation and test size

MSE=zeros(1,r);  %test error
timing=zeros(1,r);  %time code
pred=zeros(n,m,K);  %predicted rating matrix
rank_est=zeros(1,r);  %estimated rank
trainerror=zeros(1,r);  %training error

%set options for soft_impute
OPTS=[];
OPTS.TOLERANCE=10^-4; OPTS.MAXITER=100; OPTS.SMALL_SCALE=0;

%run r iterations
for j=1:r
   
    %set random seed
    rng(200+200*j);
    
    %create training, validation and test sets
    idx=randsample(length(reviewdat(:,1)),Q);
    tidx=sort(idx(1:floor(Q/2)));
    vidx=sort(idx((floor(Q/2)+1):Q));
    test=reviewdat(tidx,:);
    val=reviewdat(vidx,:);
    train=reviewdat;
    train(idx,:)=[];
    
    %construct sparse rating (training) matrix
    Xobs=sparse(train(:,1),train(:,2),train(:,3),n,m);
    trainidx=sparse(train(:,1),train(:,2),1,n,m);
    
    %sparse validation matrix
    valid=sparse(val(:,1),val(:,2),val(:,3),n,m);
    validx=sparse(val(:,1),val(:,2),1,n,m);
    
    %sparse test matrix
    testset=sparse(test(:,1),test(:,2),test(:,3),n,m);
    tidx=sparse(test(:,1),test(:,2),1,n,m);
    
    %choose big enough lambda(tuning) such that solution is zero
    %if lambda is max singular values
    %then all singular values will shrink to zero after soft thresholding
    %find spectral radius by Lanczos
    lambda_max=lansvd(Xobs,1,'L');
    
    %create lambda vector
    CVerror=zeros(1,K);
    lambda=linspace(lambda_max*.9,lambda_max/100,K);
    
    tic;  %time the algorithm
    INIT=[];  %warm start
    for i = 1:K
        %compute solution path
        [U,D,V,~]=soft_impute(Xobs, lambda(i), OPTS, INIT);
        pred(:,:,i)=U*D*V';
        
        %choose optimal lambda using CV on validation set
        CVerror(i)=norm(validx.*pred(:,:,i) - valid, 'fro')/norm(valid,'fro');
        
        %warm starts for next (smaller) lambda
        INIT=struct('U',U,'D',D,'V',V);  
    end
    
    %find lambda with min CV error
    minCV=find(CVerror == min(CVerror));
    
    %if there are ties
    if length(minCV)>1
        minCV=minCV(1);  %get the sparsest solution
    end
    
    %solution at optimal lambda
    opt_pred=pred(:,:,minCV);
    
    timing(j)=toc;
    %calculate test error
    MSE(j)=norm(tidx.*opt_pred - testset, 'fro')/norm(testset,'fro');
    trainerror(j)=norm(trainidx.*opt_pred - Xobs, 'fro')/norm(Xobs,'fro');
    rank_est(j)=rank(opt_pred);
    
    %save relative mse, estimated rank,  
    %computational time, and training error
    save('mse.mat','MSE');
    save('estrank.mat','rank_est');
    save('time.mat','timing');
    save('trainerror.mat','trainerror');
end





