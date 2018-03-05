function [Glr_mat_u,Glr_mat_d,Glr_mat_v,obj_vals]=soft_impute_path(GXobs,lambda,OPTS)
% This function performs SoftImpute, matrix completion with noisy entries:
% min_X  0.5*|| GXobs - P(X) ||_F^2 + lambda ||X||_* --------------------- (A)
% P(X) with size=[nrow, ncol] is a sparse matrix with zeros in the unobserved locations.
% GXobs is the observed data matrix with exactly the same sparsity pattern as P(X). 
% lambda: tuning parameter
% ||X||_* = nuclear norm of X ie sum(singular values of X)
%==================================================================
% soft_impute_path(..) creates a sequence \lambda_max > ..> \lambda (see OPTS below) and calls  
% soft_impute(..) with warm-starts to solve  (A). 
%==================================================================
% This is done to ensure that the successive iterates remain 
% low-rank provided the solution at lambda is low rank as well. 
% OUTPUTS:
%   Glr_mat_u :  left singular matrix,
%   Glr_mat_d : singular values (vector)
%   Glr_mat_v : Right singular matrix.
%   obj_vals  : sequence of objective values across iterations 
% INPUTS:
%1) GXobs    : sparse matrix 0's correspond to the missing values (REQUIRED)
%2) lambda   : the value of the tuning parameter (REQUIRED)
%3) Structure OPTS (Optional) with fields
%     NUMBER_GRID (optional) : number of grid values in [lambda_max, lambda], to obtain the solution at lambda. default=10; 
%     TOLERANCE  (optional)  : convergence criterion --lack of progress of succ iterates), default=10^-4
%     MAXITER    (optional)  : max no. of iterations reqd. for convergence,   default =500
%     MAX_RANK   (optional)  : max no of sing-vectors that can be computed,   default=min(nrow,ncol);
%     SMALL_SCALE(optional)  :=1 means small-scale, direct factorization based svd will be used; 
%                              default  (program decides to go small scale if (min(nrow,ncol) < 2000) 
%     INCREMENT  (optional) : increase the number of sing-vectors to be computed as a part of PROPACK by this amount, default=10;
% For any questions/ suggestions/ comments/ bugs please report to rahulm@stanford.edu

% Matlab code written by Rahul Mazumder <rahulm@stanford.edu>
% Reference: "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
% by Rahul Mazumder, Trevor Hastie, Rob Tibshirani (JMLR vol 11, 2010)


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
MAX_RANK=min(nrow,ncol);MAXITER=100; TOLERANCE=10^-4; INCREMENT=10; SMALL_SCALE=0; NUMBER_GRID=10;
%% check if OPTS are supplied, by the user
%% Parse OPTS structure

if isstruct(OPTS)
    c = fieldnames(OPTS);
    for i=1:length(c)  
        if any(strcmpi(c(i),'NUMBER_GRID')); NUMBER_GRID = double(getfield(OPTS,'NUMBER_GRID'));  end
        if any(strcmpi(c(i),'TOLERANCE')); TOLERANCE = double(getfield(OPTS,'TOLERANCE'));  end
        if any(strcmpi(c(i),'MAX_RANK'));  MAX_RANK= double(getfield(OPTS,'MAX_RANK')); end
        if any(strcmpi(c(i),'MAXITER')); MAXITER = double(getfield(OPTS,'MAXITER')); end
        if any(strcmpi(c(i),'INCREMENT')); INCREMENT = double(getfield(OPTS,'INCREMENT')); end
        if any(strcmpi(c(i),'SMALL_SCALE')); SMALL_SCALE = double(getfield(OPTS,'SMALL_SCALE')); end
    end
end


%% find the value of lambda_max such that the solution is zero at that lambda

%% define the grid of lambda values lambda_seq
lambda_max=spectral_norm(GXobs);
lambda_max=(lambda_max*.9 + 0.1*lambda);
lambda_seq=exp(linspace(log(lambda_max),log(lambda),NUMBER_GRID));

% Intialize OPTS_temp, convergence criterion

OPTS_temp=[];
OPTS_temp.TOLERANCE=10^-2;
OPTS_temp.MAXITER=20;OPTS_temp.MAX_RANK=MAX_RANK;
[Glr_mat_u,Glr_mat_d,Glr_mat_v,obj_vals]=soft_impute(GXobs,lambda_seq(1),OPTS_temp);

% Assign warm-start
INIT_temp=struct('U', Glr_mat_u , 'D', Glr_mat_d,  'V', Glr_mat_v );
clear Glr_mat_u Glr_mat_d Glr_mat_v

for lambda_index =2 : (NUMBER_GRID-1)
[Glr_mat_u,Glr_mat_d,Glr_mat_v,obj_vals]=soft_impute(GXobs,lambda_seq(lambda_index),OPTS_temp,INIT_temp);
%%lambda_index
if nnz(Glr_mat_d) > MAX_RANK 
    disp('Warning!: User supplied MAX_RANK exceeded prior to lambda\n')
    disp(['Rank exceeds MAX_RANK at lambda=',num2str(lambda_seq(lambda_index))])
    return;
end

%% reassign warm-start
INIT_temp=struct('U', Glr_mat_u , 'D', Glr_mat_d,  'V', Glr_mat_v );
clear  Glr_mat_u Glr_mat_d Glr_mat_v

end

OPTS=rmfield(OPTS,'NUMBER_GRID');

%% finally for the solution at lambda
[Glr_mat_u,Glr_mat_d,Glr_mat_v,obj_vals]=soft_impute(GXobs,lambda,OPTS,INIT_temp);












