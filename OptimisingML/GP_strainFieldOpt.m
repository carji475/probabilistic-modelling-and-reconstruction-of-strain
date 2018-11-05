function par_opt = GP_strainFieldOpt(obs,y,m_1,m_2,Lx,Ly,mu_att,nrSegs,addPrevSegs,E,v,options,start_guesses,covFunc)
%GP_strainFieldOpt Optimises hyperparamters for 2D strain field reconstruction.
%   par_opt = GP_strainFieldOpt(obs,y,m_1,m_2,Lx,Ly,mu_att,nrSegs,addPrevSegs,E,v,options,start_guesses)
%   returns the best solution found after optimising with supplied start guesses, using
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
%       m_1,m_2 - number of basis functions in x/y-direction
%       Lx,Ly - domain size in x/y-direction to solve for the eigenvalues
%       mu_att - attenuation coefficient
%       nrSegs - Nx1 vector specifying the number of segments the ray passes 
%           through for each measurement
%       addPrevSegs - Nx1 vector specifying the total number of additional 
%           segments of all previous measurements 
%       sigma_f - signal variance
%       E - Young's modulus
%       v - Poisson's ratio
%       options - options for fminunc
%       start_guesses - Mx4 matrix with start guesses each column of the
%           form [signal variance, x-length scale, y-length scale, noise level].
%           In case of common length scale, set the last element to NaN.
%       covFunc - structure specifying the covariance function
%           Options
%               covFunc=struct('type','SE') - squared exponential
%               covFunc=struct('type','Matern','nu',nu_val) - Matern with any nu
%
%   OUTPUTS
%       par_opt - row-vector with optimal values

n_obs=length(y);                % #observations

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

% build the basis functions
[Phi,~]=eqBothcalc_Att(n_obs,0,mm_adj,[mm1 mm2],obs(:),[],[],Lx,Ly,A,B,mu_att,nrSegs,addPrevSegs);
sq_lambda = [(mm1*pi/(2*Lx))   (mm2*pi/(2*Ly))]; % frequencies

val_opt=realmax; % optimal value of ML, initialise large
par_opt=NaN;     % optimal parameter values
for qq=1:size(start_guesses,1)
    sg=abs(start_guesses(qq,:)); sg=sg(~isnan(sg));
    try
        % call the solver
        [sol,val] = fminunc(@(thetas) hypOpFunc(...
            thetas, n_obs, mm_adj, y, Phi, sq_lambda,covFunc), log(sg), options);
        sol = exp(sol);
        if val<val_opt % change if better
            val_opt=val;
            par_opt=sol;
        end
    catch errorObj % problem occurred
        disp(['===== PROBLEM WITH GUESS NR ' num2str(qq) ' ======'])
        disp(' ')
        disp(getReport(errorObj,'extended','hyperlinks','on'));
    end
end
if isnan(par_opt)   % throw an error if all start guesses failed
    error('No valid start guess');
end
end
