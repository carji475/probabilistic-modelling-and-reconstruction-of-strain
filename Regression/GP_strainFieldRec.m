function [epsxx_pred,epsxy_pred,epsyy_pred,varargout]=...
    GP_strainFieldRec(obs,y,pred,m_1,m_2,Lx,Ly,nrSegs,addPrevSegs,sigma_f,M,sigma_n,E,v,covFunc,varargin)
% GP_strainFieldRec Reconstructed 2D strain field components.
%   [epsxx_pred,epsxy_pred,epsyy_pred,epsxx_var,epsxy_var,epsyy_var]=GP_strainFieldRec(obs,y,pred,m_1,m_2,Lx,Ly,mu_att,nrSegs,addPrevSegs,sigma_f,M,sigma_n,E,v) 
%   returns the reconstructed components for the 2D strain tensor, using
%   the approximation method proposed by Solin/Särkkä
%   
%   INPUTS
%       obs - 4x(N+totAddSegs) matrix with entry and exit points for the 
%           measurements each column having the form [x0; x1; y0; y1]. If 
%           the ray passes through multiple segments, place the columns 
%           left to right corresponding to first to last segment.
%                 N is the number of measurements and totAddSegs is the
%           total number of additional segments of all measurements (ie.g 
%           if a ray passes through 3 segments, 2 of them are considered addtional)
%       y - column vector of measured values
%       pred - Px2 matrix with coordinates for the prediction on the form [X(:) Y(:)]
%       m_1,m_2 - number of basis functions in x/y-direction
%       Lx,Ly - domain size in x/y-direction to solve for the eigenvalues
%       mu_att - attenuation coefficient
%       nrSegs - Nx1 vector specifying the number of segments the ray passes 
%           through for each measurement
%       addPrevSegs - Nx1 vector specifying the total number of additional 
%           segments of all previous measurements 
%       sigma_f - signal variance
%       M - length scale, either general with separate components for x and y 
%           or just a single common 
%       sigma_n - noise level
%       E - Young's modulus
%       v - Poisson's ratio
%       covFunc - structure specifying the covariance function
%           Options
%               covFunc=struct('type','SE') - squared exponential
%               covFunc=struct('type','Matern','nu',nu_val) - Matern with any nu
%   OUTPUTS
%       epsxx_pred,epsxy_pred,epsyy_pred - row vectors containing the
%           corresponding predictions
%       epsxx_var,epsxy_var,epsyy_var - row vectors containing the
%           corresponding variances

if length(M)==1                 % check if generalised length scale
    M=[M M];
end

n_obs=length(y);                % #observations
n_pred=size(pred,1);            % #points for prediction

lambda = v*E/((1+v)*(1-2*v));   % lamé constants
mu = (1-v)*lambda/(2*v); 
b=lambda; c=2*mu; a=b+c;        % constraint parameters
A=a/b; B=-(a+b)/b;              % implementational form

%% approximation
% place the basis functions in an eliptical domain
[mm1,mm2] = meshgrid(1:m_1,1:m_2);
insideEllipse = mm1.^2/m_1^2+mm2.^2/m_2^2<1;
mm1 = mm1(insideEllipse);
mm2 = mm2(insideEllipse);
mm_adj = length(mm1(:));              % #basis functions

mm1=mm1(:); mm2=mm2(:);

% build Phi and predPhi^T
[Phi, predPhi_T]=eqBothcalc(n_obs,n_pred,mm_adj,[mm1 mm2],...
     obs(:),pred(:,1),pred(:,2),Lx,Ly,A,B,nrSegs,addPrevSegs);

% spectral density
sq_lambda = [(mm1*pi/(2*Lx))   (mm2*pi/(2*Ly))]; % angular frequencies
switch covFunc.type
    case 'SE'
        invLambda = 1./(2*pi*M(1)*M(2)*exp(-0.5*(M(1)^2*sq_lambda(:,1).^2+...
            M(2)^2*sq_lambda(:,2).^2))*sigma_f^2);
    case 'Matern'
        nu = covFunc.nu;
        invLambda = 1./...
            ( sigma_f^2*( (4*pi*gamma(nu+1)*(2*nu)^nu) / (gamma(nu)*(M(1)*M(2))^(nu)) )*...
            ((2*nu+sq_lambda(:,1).^2*M(1)^2+sq_lambda(:,2).^2*M(2)^2)/(M(1)*M(2))).^(-(nu+1)) );
end

disp(['Minimum inverted SD: ' num2str(min(invLambda))])
disp(['Maximum inverted SD: ' num2str(max(invLambda))])

% % numerical tweak
Q = ( Phi'*Phi+sigma_n^2*diag(invLambda) );
Q = 0.5*(Q+Q');
Q = Q+2*abs(min([0; eig(Q)]))*eye(size(Q));

%% predict
mean_pred = reshape(predPhi_T*( Q\(Phi'*y) ),3,n_pred);
epsxx_pred = mean_pred(1,:)';
epsxy_pred = mean_pred(2,:)';
epsyy_pred = mean_pred(3,:)';

if isempty(varargin)
    % variance
    var_pred = reshape( sum( sigma_n^2*predPhi_T.*(( Q\(predPhi_T') )'), 2), 3,n_pred);
    varargout{1} = var_pred(1,:)';
    varargout{2} = var_pred(2,:)';
    varargout{3} = var_pred(3,:)';
end
end
