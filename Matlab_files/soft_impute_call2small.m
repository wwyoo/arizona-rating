function [Glr_mat_u,Glr_mat_d,Glr_mat_v,obj_vals]=soft_impute_call2small(GXobs,lambda,OPTS,INIT)
% This function performs SoftImpute, matrix completion with noisy entries:
% min_X  0.5*|| GXobs - P(X) ||_F^2 + lambda ||X||_* --------------------- (A)
% P(X) with size=[nrow, ncol] is a sparse matrix with zeros in the unobserved locations.
% GXobs is the observed data matrix with exactly the same sparsity pattern as P(X). 
% lambda: tuning parameter
% ||X||_* = nuclear norm of X ie sum(singular values of X)
%=====================================================
% SoftImpute for small scale problems with direct-factorization based SVD (not iterative SVD).
% Use for small problems with  matrix dimensions most 2K. 
%=====================================================
% Function "soft_impute.m" calls this function if the problem is small-scale. It is recommended to use the
% generic purpose function "soft_impute.m".  
% INPUTS:
%1) GXobs_{nrow \times ncol} : sparse matrix 0's correspond to the missing values (REQUIRED)
%2) lambda   : the (scalar) value of the tuning parameter (REQUIRED)
%3) Structure OPTS (Optional) with fields
%     TOLERANCE  (optional)  : convergence criterion --lack of progress of succ iterates), default=10^-4
%     MAXITER    (optional)  : max no. of iterations reqd. for convergence,   default =500
%4) Structure INIT (Optional) with fields 
%   U_{nrow\times k}: left matrix of singular vectors 
%   D_{k\times k}: matrix of singular values (diagonal) 
%   V_{ncol \times k}: right matrix of singular vectors 
%         if provided ALL of (U,D,V) are required
%     Default: All U,D, V are set to zero.    
% OUTPUTS:
%   Glr_mat_u :  left singular matrix,
%   Glr_mat_d : singular values (vector)
%   Glr_mat_v : Right singular matrix.
%   obj_vals  : sequence of objective values across iterations 
% For any questions/ suggestions/ comments/ bugs please report to rahulm@stanford.edu

% Matlab code written by Rahul Mazumder <rahulm@stanford.edu>
% Reference: "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
% by Rahul Mazumder, Trevor Hastie, Rob Tibshirani (JMLR vol 11, 2010)


if (nargin<2) 
disp('Error: require at least two inputs');    
Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];  obj_vals=[];
return
end

if  (nargin==3) && ~isempty(OPTS) 
     if ~isstruct(OPTS);
disp('Error: OPTS must be a structure');    
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

if  ( isempty(GXobs) || isempty(lambda) )
Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];obj_vals=[];
      disp('Error: requires observed data AND lambda');
return
end

if  (min(lambda) <0) || (length(lambda)>1) 
disp('Error: lambda needs to be non-negative and a scalar');
Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];obj_vals=[];
return;
end

if ~issparse(GXobs)
Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];obj_vals=[];
disp('Error: requires Input observed matrix to be a sparse matrix');
return
end

dim_check=size(GXobs);
if length(dim_check)~=2
Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];obj_vals=[];
disp('Error: Incorrect dimensions for Observed matrix');
return
end

nrow=dim_check(1); ncol=dim_check(2); clear dim_check; 

%% Declare defaults
MAX_RANK=min(nrow,ncol);MAXITER=100; TOLERANCE=10^-4; INCREMENT=10;

%% check if OPTIONS are supplied, by the user
%% Parse OPTS struct

if isstruct(OPTS)
    c = fieldnames(OPTS);
    for i=1:length(c)  
        if any(strcmpi(c(i),'TOLERANCE')); TOLERANCE = double(getfield(OPTS,'TOLERANCE'));  end
        if any(strcmpi(c(i),'MAX_RANK'));  MAX_RANK= double(getfield(OPTS,'MAX_RANK')); end
        if any(strcmpi(c(i),'MAXITER')); MAXITER = double(getfield(OPTS,'MAXITER')); end
        if any(strcmpi(c(i),'INCREMENT')); INCREMENT = double(getfield(OPTS,'INCREMENT')); end
    end
end
clear OPTS c

obj_vals=zeros(MAXITER,1);
GXobs=sparse(GXobs);
[i_row, j_col, data]=find(GXobs); number_observed=length(data);

%% check if starting point factorization is supplied

if isstruct(INIT)

   if ( (~isempty(INIT.U))&(~isempty(INIT.D)) & (~isempty(INIT.V)) )
           Glr_mat_u =INIT.U; Glr_mat_v=INIT.V; Glr_mat_d=INIT.D;
           clear INIT
         dim_check1=size(Glr_mat_u); dim_check2=size(Glr_mat_v); dim_check3= size(Glr_mat_d); 
         if ( (dim_check1(1)~=nrow) || (dim_check2(1)~=ncol) || (dim_check2(2)~= dim_check2(2) ) )  || ( dim_check3(1) ~= dim_check3(2) )
         disp('Error: wrong dimensions in Input starting point'); 
         Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];obj_vals=[];
         return;
         end         
         if (dim_check2(2) > MAX_RANK)
         disp('Error: Input starting point has rank larger than MAX_RANK'); 
         Glr_mat_u=[];Glr_mat_d=[];Glr_mat_v=[];obj_vals=[];
         return;
         end
         
   else
   disp('Error: Not proper starting point');   
   end
   
else

Glr_mat_u=zeros(nrow,1); Glr_mat_v=zeros(ncol,1);
Glr_mat_d=spalloc(1,1,1);
GPmZ_old=sparse([],[],[],nrow,ncol,number_observed);  % projected imputation of matrix (Initialized to zero) 

end

%% Prepare to intialize "GPmZ_old"
Glr_mat_u=Glr_mat_u*Glr_mat_d; 
tttemp=sum(Glr_mat_u(i_row,:).*Glr_mat_v(j_col,:),2); 
GPmZ_old=sparse(i_row,j_col,tttemp,nrow,ncol,number_observed); 
GZ_old=Glr_mat_u*Glr_mat_v'; %current estimate of the matrix
tttemp=data-tttemp; train_err= tttemp'*tttemp;  
soft_singvals=max(diag(Glr_mat_d)-lambda,0); 
objval_new=train_err/2 + lambda*sum(soft_singvals);

clear dim_check1 dim_check2 dim_check3

tol_curr=10;

objval_old=data'*data; 

tol_curr=10^5;
i=0; 

while (tol_curr>TOLERANCE)&(i<MAXITER) 
i=i+1 ;
target_old= GXobs+GZ_old-GPmZ_old;

%%% this is for svd
[a,b,c] = svd(target_old,'econ'); 
sing_vals=diag(b); clear b;

%%early exit if iterate turns to be zero.
if (max(sing_vals) <= lambda)
disp('lambda-too-large');
Glr_mat_u=sparse(nrow,1); Glr_mat_d=0; Glr_mat_v=sparse(ncol,1); obj_vals=[];
return
end

%% soft-threshold, singular values
soft_singvals=max(sing_vals-lambda,0); 
soft_singvals=soft_singvals(soft_singvals>0);
no_singvals=length(soft_singvals);

Glr_mat_d=sparse([1:no_singvals],[1:no_singvals],soft_singvals,no_singvals,no_singvals); 
Glr_mat_u=a(:,1:no_singvals);  clear a
Glr_mat_u=Glr_mat_u*Glr_mat_d; u_vec=Glr_mat_u(i_row,:); %% clear Glr_mat_u 

Glr_mat_v=c(:,1:no_singvals); clear c; v_vec=Glr_mat_v(j_col,:);

tttemp=sum(u_vec.*v_vec,2); clear u_vec v_vec
GPmZ_old=sparse(i_row,j_col,tttemp);  
tttemp=data-tttemp;
train_err= tttemp'*tttemp;  

GZ_old=Glr_mat_u*Glr_mat_v';


objval_new=train_err/2 + lambda*sum(soft_singvals);
obj_vals(i)=objval_new;
tol_curr= abs(-objval_new+objval_old)/(objval_old+10^-6);

objval_old=objval_new;

end

obj_vals=obj_vals(1:i);

Glr_mat_u=Glr_mat_u*sparse(1:no_singvals,1:no_singvals,1./soft_singvals);

















