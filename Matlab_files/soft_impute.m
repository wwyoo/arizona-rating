function [Glr_mat_u,Glr_mat_d,Glr_mat_v,obj_vals]=soft_impute(GXobs,lambda,OPTS,INIT)
% This function performs SoftImpute, matrix completion with noisy entries:
% min_X  0.5*|| GXobs - P(X) ||_F^2 + lambda ||X||_* --------------------- (A)
% P(X) with size=[nrow, ncol] is a sparse matrix with zeros in the unobserved locations.
% GXobs is the observed data matrix with exactly the same sparsity pattern as P(X). 
% lambda: tuning parameter
% ||X||_* = nuclear norm of X ie sum(singular values of X)
% soft_impute(..) solves the problem (A) at lambda allowing for warm-start INIT (see below).
% If the warm-start is arbitrary the succesive iterates may have varying ranks, in that case use 
% soft_impute_path(..) instead.
% INPUTS:
%1) GXobs_{nrow \times ncol}    : sparse matrix 0's correspond to the missing values (REQUIRED)
%2) lambda                      : scalar value of the tuning parameter (REQUIRED)
%3) Structure OPTS (Optional) with fields
%     TOLERANCE  (optional)  : convergence criterion --lack of progress of succ iterates), default=10^-4
%     MAXITER    (optional)  : max no. of iterations reqd. for convergence,   default =100
%     MAX_RANK   (optional)  : max no of sing-vectors that can be computed,   
%                            default=min(nrow,ncol);if min(nrow,ncol) > 2000; MAX_RANK=min(500,MAX_RANK); 
%                                 if number_observed > 10^7, MAX_RANK=min(50,MAX_RANK)
%     SMALL_SCALE(optional)  :=1 means small-scale, direct factorization based svd will be used; 
%                              default  (program decides to go small scale if (min(nrow,ncol) < 2000) 
%     INCREMENT  (optional) : increase the number of sing-vectors to be computed as a part of PROPACK by this amount, default=10;
%4) Structure INIT (Optional) with fields 
%   U_{nrow\times k}: left matrix of singular vectors 
%   D_{k\times k}: matrix of singular values (diagonal) 
%   V_{ncol \times k}: right matrix of singular vectors 
%         if provided ALL of (U,D,V) are required
%     Default : All U,D, V are set to zero.    
% OUTPUTS:
%   Glr_mat_u :  left singular matrix,
%   Glr_mat_d : singular values (vector)
%   Glr_mat_v : Right singular matrix.
%   obj_vals  : sequence of objective values across iterations 
% For any questions/ suggestions/ comments/ bugs please report to rahulm@stanford.edu

% Matlab code written by Rahul Mazumder <rahulm@stanford.edu>
% Reference: "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
% by Rahul Mazumder, Trevor Hastie, Rob Tibshirani (JMLR vol 11, 2010)


global INCREMENT MAX_RANK lambda_value

if (nargin<2) || isempty(GXobs) || isempty(lambda)
disp('Error: require at least two inputs and also GXobs and lambda \n');    
Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];  obj_vals=[];
return
end

if  (nargin==3) && ~isempty(OPTS) 
     if ~isstruct(OPTS);
disp('Error: OPTS must be a structure \n');    
Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];  obj_vals=[];
return
     end
end

if  (nargin<3) || isempty(OPTS)
      OPTS=[];
end

if  (nargin<4) || isempty(INIT)
      INIT=[];
end



if  isempty(lambda) || (min(lambda) <0) || (length(lambda)>1) 
disp('Error: lambda needs to be non-negative and a scalar \n');
Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];obj_vals=[];
return;
end


if ~issparse(GXobs)
Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];obj_vals=[];
disp('Error: requires Input observed matrix to be a sparse matrix \n');
return
end

dim_check=size(GXobs);
if length(dim_check)~=2
Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];obj_vals=[];
disp('Error: Incorrect dimensions for Observed matrix \n');
return
end

nrow=dim_check(1); ncol=dim_check(2); clear dim_check; 

%% Declare defaults
MAX_RANK=min(nrow,ncol);MAXITER=100; TOLERANCE=10^-4; INCREMENT=10; SMALL_SCALE=0;

%% check if OPTS are supplied, by the user
%% Parse OPTS structure

if isstruct(OPTS)
    c = fieldnames(OPTS);
    for i=1:length(c)  
        if any(strcmpi(c(i),'TOLERANCE')); TOLERANCE = double(getfield(OPTS,'TOLERANCE'));  end
        if any(strcmpi(c(i),'MAX_RANK'));  MAX_RANK= double(getfield(OPTS,'MAX_RANK')); end
        if any(strcmpi(c(i),'MAXITER')); MAXITER = double(getfield(OPTS,'MAXITER')); end
        if any(strcmpi(c(i),'INCREMENT')); INCREMENT = double(getfield(OPTS,'INCREMENT')); end
        if any(strcmpi(c(i),'SMALL_SCALE')); SMALL_SCALE = double(getfield(OPTS,'SMALL_SCALE')); end
    end
end

lambda_value=lambda;

if (min(nrow,ncol) < 2000) || (SMALL_SCALE==1)
%% disp('going for small-scale direct svd factorization code')
[Glr_mat_u,Glr_mat_d,Glr_mat_v,obj_vals]=soft_impute_call2small(GXobs,lambda,OPTS,INIT);
return
end

clear OPTS c

obj_vals=zeros(MAXITER,1);
GXobs=sparse(GXobs);
[i_row, j_col, data]=find(GXobs); number_observed=length(data);

%% update MAX_RANK to safe-guard against large ranks
%% If you really need a large-rank solution, then change the default/ comment this part out.
if min(nrow,ncol) > 2000
MAX_RANK=min(500,MAX_RANK)
end
if number_observed > 10^7
MAX_RANK=min(50,MAX_RANK)
end


if isstruct(INIT)

   if ( (~isempty(INIT.U))&(~isempty(INIT.D)) & (~isempty(INIT.V)) )
           Glr_mat_u =INIT.U; Glr_mat_v=INIT.V; Glr_mat_d=INIT.D;
         dim_check1=size(Glr_mat_u); dim_check2=size(Glr_mat_v); dim_check3= size(Glr_mat_d); 
         clear INIT
         if ( (dim_check1(1)~=nrow) || (dim_check2(1)~=ncol) || (dim_check2(2)~= dim_check2(2) ) )  || ( dim_check3(1) ~= dim_check3(2) )
         disp('Error: wrong dimensions in Input starting point \n'); 
         Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];obj_vals=[];
         return;
         end         
         if (dim_check2(2) > MAX_RANK)
         disp('Error: Input starting point has rank larger than MAX_RANK \n'); 
         Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];obj_vals=[];
         return;
         end
         
   else
   disp('Error: Not proper starting point \n');   
   end
 
else %% intialize guess to zero
Glr_mat_u=zeros(nrow,1); Glr_mat_v=zeros(ncol,1);
Glr_mat_d=spalloc(1,1,1);
GPmZ_old=sparse([],[],[],nrow,ncol,number_observed);  % projected imputation of matrix (Initialized to zero) 
end

Glr_mat_u=Glr_mat_u*Glr_mat_d; 
    if (number_observed <10^6)
          tttemp=sum(Glr_mat_u(i_row,:).*Glr_mat_v(j_col,:),2); 
    else
          tttemp=project_obs_UV(Glr_mat_u,Glr_mat_v,i_row,j_col,number_observed);
    end
GPmZ_old=sparse(i_row,j_col,tttemp,nrow,ncol,number_observed); 

clear dim_check1 dim_check2 dim_check3

tttemp=data-tttemp; train_err= tttemp'*tttemp;  
soft_singvals=max(diag(Glr_mat_d)-lambda,0); 
objval_old=train_err/2 + lambda*sum(soft_singvals);

tol_curr=10^5;

i=0; 

svd_rank=5;

while ((tol_curr>TOLERANCE)&(i<MAXITER)) 

i=i+1 ;

Front_multi =@(x)A_multiply_fun_handle(x,GXobs,Glr_mat_u,Glr_mat_v,GPmZ_old);
Front_Transpose_multi =@(x)At_multiply_fun_handle(x,GXobs,Glr_mat_u,Glr_mat_v,GPmZ_old);

[a,b,c] = lansvd_lambda(Front_multi,Front_Transpose_multi,nrow,ncol,svd_rank,'L');  
 
sing_vals=diag(b); clear b;

if (max(sing_vals) <= lambda)
disp('lambda-too-large, solution is zero');
Glr_mat_u=sparse(nrow,1); Glr_mat_d=0; Glr_mat_v=sparse(ncol,1); obj_vals=[];
return
end


%% soft-threshold, singular values

soft_singvals=max(sing_vals-lambda,0); 
soft_singvals=soft_singvals(soft_singvals>0);
no_singvals=length(soft_singvals);
svd_rank=no_singvals;

Glr_mat_d=sparse([1:no_singvals],[1:no_singvals],soft_singvals,no_singvals,no_singvals); 

Glr_mat_u=a(:,1:no_singvals);  clear a
Glr_mat_u=Glr_mat_u*Glr_mat_d; 
Glr_mat_v=c(:,1:no_singvals); clear c; 
 
if (number_observed <10^5)
tttemp=sum(Glr_mat_u(i_row,:).*Glr_mat_v(j_col,:),2); 
else
tttemp=project_obs_UV(Glr_mat_u,Glr_mat_v,i_row,j_col,number_observed);
end

GPmZ_old=sparse(i_row,j_col,tttemp); 

tttemp=data-tttemp;
train_err= tttemp'*tttemp;  

objval_new=train_err/2 + lambda*sum(soft_singvals);
obj_vals(i)=objval_new;
tol_curr= abs(objval_old - objval_new)/(objval_old+10^-5);

objval_old=objval_new;

end

obj_vals=obj_vals(1:i);



Glr_mat_u=Glr_mat_u*sparse(1:no_singvals,1:no_singvals,1./soft_singvals);

















