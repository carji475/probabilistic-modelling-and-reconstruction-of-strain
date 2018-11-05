function [ L , grad] = hypOpFunc( thetas, n_obs, nr_base, y, Phi, sq_lambda, covFunc )
% optimisation function
% returns the function value and gradient
% implements solin/särkkä

thetas=exp(thetas); % optimising the logarithm

% common length scale?
if length(thetas)==4
    genLS=1;
else
    genLS=0;
end

%% extract hyperparameters
sigma_f = thetas(1);            % signal variance
if genLS
    lx = thetas(2);             % length scale x
    ly = thetas(3);             % length scale y
    sigma_n = thetas(4);        % noise
else
    ll = thetas(2);             % length scale common
    sigma_n = thetas(3);        % noise
end

switch covFunc.type
    case 'SE'
        try
            if genLS
                invLambda = 1./(2*pi*lx*ly*exp(-0.5*(lx^2*sq_lambda(:,1).^2+ly^2*...
                    sq_lambda(:,2).^2))*sigma_f^2);
            else
                invLambda = 1./(2*pi*ll^2*exp(-0.5*ll^2*sum(sq_lambda.^2,2))*sigma_f^2);
            end
            Z = sigma_n^2*diag(invLambda)+Phi'*Phi;
            Z = 0.5*(Z+Z'); Z = Z+2*abs(min([0; eig(Z)]))*eye(size(Z));
            invZ = Z\eye(nr_base);
            
            % terms involving log|Q|
            logQ = (n_obs-nr_base)*log(sigma_n^2)+2*sum(log(diag(chol(Z))))+sum(log(1./invLambda));
            dlogQ_dsigma_f = 2*nr_base-sigma_n^2*sum(sum(invZ.*diag(invLambda*2)',2));
            if genLS
                dlogQ_dlx = sum(1-lx^2*sq_lambda(:,1).^2)-sigma_n^2*...
                    sum(sum(invZ.*diag(invLambda.*(1-lx^2*sq_lambda(:,1).^2))',2));
                dlogQ_dly = sum(1-ly^2*sq_lambda(:,2).^2)-sigma_n^2*...
                    sum(sum(invZ.*diag(invLambda.*(1-ly^2*sq_lambda(:,2).^2))',2));
            else
                dlogQ_dll = sum(2-ll^2*sum(sq_lambda.^2,2))-sigma_n^2*...
                    sum(sum(invZ.*diag(invLambda.*(2-ll^2*sum(sq_lambda.^2,2)))',2));
            end
            dlogQ_dsigma_n =  2*(n_obs-nr_base)+2*sigma_n^2*sum(sum(invZ.*diag(invLambda)',2));
            
            % terms involving invQ
            yTinvQy1 = (1/sigma_n^2)*(y'*y);
            yTinvQy2 = (1/sigma_n^2)*-y'*Phi*(invZ*Phi')*y;
            yTinvQy = yTinvQy1+yTinvQy2;
            dyTinvQy_dsigma_f = -y'*Phi*invZ*diag(invLambda*2)*invZ*Phi'*y;
            if genLS
                dyTinvQy_dlx = -y'*Phi*invZ*diag(invLambda.*...
                    (1-lx^2*sq_lambda(:,1).^2))*invZ*Phi'*y;
                dyTinvQy_dly = -y'*Phi*invZ*diag(invLambda.*...
                    (1-ly^2*sq_lambda(:,2).^2))*invZ*Phi'*y;
            else
                dyTinvQy_dll = -y'*Phi*invZ*diag(invLambda.*...
                    (2-ll^2*sum(sq_lambda.^2,2)))*invZ*Phi'*y;
            end
            invZPhi = (invZ*(Phi'*y)).*sqrt(invLambda);
            dyTinvQy_dsigma_n = ( 2*(invZPhi'*invZPhi)+(2/sigma_n^2)*y'*Phi*(invZ*Phi')*y-...
                (2/sigma_n^2)*(y'*y) );
            
            % neg log ML
            L = 0.5*(logQ+yTinvQy+n_obs*log(2*pi));
            
            % partial derivatives
            grad_sigma_f = 0.5*(dlogQ_dsigma_f+dyTinvQy_dsigma_f);
            if genLS
                grad_lx = 0.5*(dlogQ_dlx+dyTinvQy_dlx);
                grad_ly = 0.5*(dlogQ_dly+dyTinvQy_dly);
            else
                grad_ll = 0.5*(dlogQ_dll+dyTinvQy_dll);
            end
            grad_sigma_n = 0.5*(dlogQ_dsigma_n+dyTinvQy_dsigma_n);
            if genLS
                grad = [grad_sigma_f; grad_lx; grad_ly; grad_sigma_n];
            else
                grad = [grad_sigma_f; grad_ll; grad_sigma_n];
            end
        catch
            L=NaN; grad=nan(3+genLS,1);
            return;
        end
    case 'Matern'
        nu = covFunc.nu;
        SD_fac = (4*pi*gamma(nu+1)*(2*nu)^nu)/gamma(nu);
        try
            if genLS
                SD_par = ((2*nu+lx^2*sq_lambda(:,1).^2+ly^2*sq_lambda(:,2).^2)/(lx*ly));
                invLambda = 1./( sigma_f^2*( SD_fac*(lx*ly)^(-nu) )*SD_par.^(-(nu+1)) );
            else
                SD_par=(2*nu/(ll^2)+sum(sq_lambda.^2,2));
                invLambda = 1./( sigma_f^2*( SD_fac*(ll)^(-2*nu) )*SD_par.^(-(nu+1)) );
            end
            Z = sigma_n^2*diag(invLambda)+Phi'*Phi;
            Z = 0.5*(Z+Z'); Z = Z+2*abs(min([0; eig(Z)]))*eye(size(Z)); 
            invZ = Z\eye(nr_base);
            
            % terms involving log|Q|
            logQ = (n_obs-nr_base)*log(sigma_n^2)+2*sum(log(diag(chol(Z))))+sum(log(1./invLambda));
            dlogQ_dsigma_f = 2*nr_base-sigma_n^2*sum(sum(invZ.*diag(invLambda*2)',2));
            if genLS
                dlogQ_dlx = lx*( sum( (( (-nu-1)*(lx*ly)^(-nu).*(2*sq_lambda(:,1).^2./(ly*SD_par)-1/lx)...
                            -ly*nu*(lx*ly)^(-nu-1)))./(( (lx*ly)^(-nu) )) )...
                -sigma_n^2*...
                    sum(sum(invZ.*diag( ...
                    SD_par.^nu.*( (-nu-1)*(lx*ly)^(nu).*(2*sq_lambda(:,1).^2/ly-SD_par/lx)...
                            -ly*nu*(lx*ly)^(nu-1)*SD_par)...
                    ./( sigma_f^2*SD_fac )...
                    )',2)) );
                
                dlogQ_dly = ly*( sum( (( (-nu-1)*(lx*ly)^(-nu).*(2*sq_lambda(:,2).^2./(lx*SD_par)-1/ly)...
                            -lx*nu*(lx*ly)^(-nu-1)))./(( (lx*ly)^(-nu) )) )...
                -sigma_n^2*...
                    sum(sum(invZ.*diag( ...
                    SD_par.^nu.*( (-nu-1)*(lx*ly)^(nu).*(2*sq_lambda(:,2).^2/lx-SD_par/ly)...
                            -lx*nu*(lx*ly)^(nu-1)*SD_par)...
                    ./( sigma_f^2*SD_fac )...
                    )',2)) );
            else
                dlogQ_dll = ll*( sum( (2*nu*(-2*(-nu-1)*ll^(-2*nu-3)./SD_par...
                            -ll^(-2*nu-1)))./(( (ll)^(-2*nu) )) )...
                -sigma_n^2*...
                    sum(sum(invZ.*diag( ...
                    (2*nu*(-2*(-nu-1)*ll^(2*nu-3)*SD_par.^nu -ll^(2*nu-1)*SD_par.^(nu+1)))...
                    ./(  sigma_f^2*SD_fac ) ...
                    )',2)) );
            end
            dlogQ_dsigma_n =  2*(n_obs-nr_base)+2*sigma_n^2*sum(sum(invZ.*diag(invLambda)',2));
            
            % terms involving invQ
            yTinvQy1 = (1/sigma_n^2)*(y'*y);
            yTinvQy2 = (1/sigma_n^2)*-y'*Phi*(invZ*Phi')*y;
            yTinvQy = yTinvQy1+yTinvQy2;
            dyTinvQy_dsigma_f = -y'*Phi*invZ*diag(invLambda*2)*invZ*Phi'*y;
            if genLS
                dyTinvQy_dlx = lx*(-y'*Phi*invZ*...
                    diag( ...
                    SD_par.^nu.*( (-nu-1)*(lx*ly)^(nu).*(2*sq_lambda(:,1).^2/ly-SD_par/lx)...
                            -ly*nu*(lx*ly)^(nu-1)*SD_par)...
                    ./( sigma_f^2*SD_fac ))...
                    *invZ*Phi'*y );
                
                dyTinvQy_dly = ly*(-y'*Phi*invZ*...
                    diag( ...
                    SD_par.^nu.*( (-nu-1)*(lx*ly)^(nu).*(2*sq_lambda(:,2).^2/lx-SD_par/ly)...
                            -lx*nu*(lx*ly)^(nu-1)*SD_par)...
                    ./( sigma_f^2*SD_fac ))...
                    *invZ*Phi'*y );
            else
                dyTinvQy_dll = ll*(-y'*Phi*invZ*...
                    diag((2*SD_fac*nu*(-2*(-nu-1)*ll^(-2*nu-3)*SD_par.^(-nu-2)...
                    -ll^(-2*nu-1).*SD_par.^(-nu-1))).*invLambda.^2*sigma_f^2)*invZ*Phi'*y );
            end
            invZPhi = (invZ*(Phi'*y)).*sqrt(invLambda);
            dyTinvQy_dsigma_n = ( 2*(invZPhi'*invZPhi)+(2/sigma_n^2)*y'*Phi*(invZ*Phi')*y-...
                (2/sigma_n^2)*(y'*y) );
            
            % neg log ML
            L = 0.5*(logQ+yTinvQy+n_obs*log(2*pi));
            
            % partial derivatives
            grad_sigma_f = 0.5*(dlogQ_dsigma_f+dyTinvQy_dsigma_f);
            if genLS
                grad_lx = 0.5*(dlogQ_dlx+dyTinvQy_dlx);
                grad_ly = 0.5*(dlogQ_dly+dyTinvQy_dly);
            else
                grad_ll = 0.5*(dlogQ_dll+dyTinvQy_dll);
            end

            grad_sigma_n = 0.5*(dlogQ_dsigma_n+dyTinvQy_dsigma_n);
            if genLS
                grad = [grad_sigma_f; grad_lx; grad_ly; grad_sigma_n];
            else
                grad = [grad_sigma_f; grad_ll; grad_sigma_n];
            end
        catch
            L=NaN; grad=nan(3+genLS,1);
            return;
        end
end
end