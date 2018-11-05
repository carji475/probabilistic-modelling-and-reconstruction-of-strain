clearvars;
% close all;

addpath(genpath('../'))

load cantilevered_processed
load rot_noise0as_noise_paperdata_96proj

%% choose a random subset of the data (if not, set n_red=n_obs)
load ang_red
n_projs = 96;
projs_ind = round(linspace(1,96,n_projs));
indices = [];
for qq=projs_ind
    indices = [indices proj_indices(qq,1):proj_indices(qq,2)];
end
n_obs = length(indices);

addPrevSegs = [0; cumsum(nrSegs(1:end-1)-1)]';
indices_tweak = indices+addPrevSegs(indices);

nrSegs_tweak = [indices_tweak; (indices_tweak+1).*(nrSegs(indices)'==2)];
nrSegs_tweak = nrSegs_tweak(:)';
nrSegs_tweak = nrSegs_tweak(nrSegs_tweak~=0);

y_meas = y_meas(indices);
nrSegs = nrSegs(indices);

obs = obs(:,nrSegs_tweak);

%% noise
noise_std = 1e-4;
rng default
y_meas=y_meas+noise_std*randn(size(y_meas));

%% Approximation parameters
Cx=3; 
Cy=Cx;

% #basis functions
m_1 = 15;       % x
m_2 = m_1;      % y

%% covariance function
covFunc = struct('type','Matern','nu',5/2);

%% material parameters
E = 200e9;  % Pa
v = 0.3;

%% mesh
nx = 40; ny = 20;
xmin = -10e-3;
xmax = 10e-3;
ymin = -0.5e-2;
ymax = 0.5e-2;
x = linspace(xmin,xmax,nx);
y = linspace(ymin,ymax,ny);
[X,Y] = meshgrid(x,y);

%% expand domain
Lx= Cx*xmax; Ly=Cy*ymax;

% specify segment information
addPrevSegs=[0; cumsum(nrSegs(1:end-1)-1)]; % #"additional" segments before previous measurement

%% compatibility
nx_comp = 5; ny_comp=5;
x_comp = linspace(xmin,xmax,nx_comp);
y_comp = linspace(ymin,ymax,ny_comp);
[Xcomp,Ycomp] = meshgrid(x_comp,y_comp);

%% find hyperparameters by optimising the marginal likelihood
% start guesses, try a few different
start_guesses = [1 1 0.01 noise_std]; % set last element to NaN if common length scale

% set optimisation options for fminunc
options = optimoptions('fminunc','Algorithm','quasi-newton','Display','iter-detailed',...
    'GradObj','on','TolFun',1e-8,'TolX',1e-8);
% call the optimising routine
par_opt = GP_strainFieldOpt(obs,y_meas,m_1,m_2,Lx,Ly,1e-6,nrSegs,addPrevSegs,E,v,options,start_guesses,covFunc);

% par_opt = [1 1 1 noise_std];

%% compute the reconstruction
% call the computational routine
[epsxx_pred,epsxy_pred,epsyy_pred,epsxx_var,epsxy_var,epsyy_var]=...
    GP_strainFieldRec(obs,y_meas,[X(:) Y(:)],m_1,m_2,Lx,Ly,1e-6,nrSegs,...
    addPrevSegs,par_opt(1),par_opt(2:end-1),par_opt(end),E,v,covFunc);

% reshape into matrices
epsxx_pred = reshape(epsxx_pred,ny,nx);
epsxy_pred = reshape(epsxy_pred,ny,nx);
epsyy_pred = reshape(epsyy_pred,ny,nx);

epsxx_std = sqrt( reshape(epsxx_var,ny,nx) );
epsxy_std = sqrt( reshape(epsxy_var,ny,nx) );
epsyy_std = sqrt( reshape(epsyy_var,ny,nx) );

%% "true" (fea) solution
epsxx_fea = Fxx([X(:) Y(:)]);
epsxy_fea = Fxy([X(:) Y(:)]);
epsyy_fea = Fyy([X(:) Y(:)]);

% reshape into matrices
epsxx_fea = reshape(epsxx_fea,ny,nx);
epsxy_fea = reshape(epsxy_fea,ny,nx);
epsyy_fea = reshape(epsyy_fea,ny,nx);

%% plot fea and reconstruction
load cmocean_bal
cmap=colormap(cmocean_bal);
clims_pred = 'auto';
clims_var = 'auto';
FigHandle = figure(1);
set(FigHandle, 'Position', [8 1 2049 895]);

figure(1)
% epsxx
subplot(3,3,1); surf(X,Y,epsxx_fea); view(0,90); colorbar; colormap(cmap); caxis(clims_pred); shading interp; axis image; title('FEA $\epsilon_{xx}$','Interpreter','latex','FontSize',16);
subplot(3,3,4); surf(X,Y,epsxx_pred); view(0,90); colorbar; colormap(cmap); caxis(clims_pred); shading interp; axis image; title('Predicted $\epsilon_{xx}$','Interpreter','latex','FontSize',16);
subplot(3,3,7); surf(X,Y,epsxx_std); view(0,90); colorbar; colormap(cmap); caxis(clims_var); shading interp; axis image; title('Predicted std','Interpreter','latex','FontSize',16);

% epsxy
subplot(3,3,2); surf(X,Y,epsxy_fea); view(0,90); colorbar; colormap(cmap); caxis([-5e-4 5e-4]); shading interp; axis image; title('FEA $\epsilon_{xy}$','Interpreter','latex','FontSize',16);
subplot(3,3,5); surf(X,Y,epsxy_pred); view(0,90); colorbar; colormap(cmap); caxis([-5e-4 5e-4]); shading interp; axis image; title('Predicted $\epsilon_{xy}$','Interpreter','latex','FontSize',16);
subplot(3,3,8); surf(X,Y,epsxy_std); view(0,90); colorbar; colormap(cmap); caxis(clims_var); shading interp; axis image; title('Predicted std','Interpreter','latex','FontSize',16);

% epsyy
subplot(3,3,3); surf(X,Y,epsyy_fea); view(0,90); colorbar; colormap(cmap); caxis(clims_pred); shading interp; axis image; title('FEA $\epsilon_{yy}$','Interpreter','latex','FontSize',16);
subplot(3,3,6); surf(X,Y,epsyy_pred); view(0,90); colorbar; colormap(cmap); caxis(clims_pred); shading interp; axis image; title('Predicted $\epsilon_{yy}$','Interpreter','latex','FontSize',16);
subplot(3,3,9); surf(X,Y,epsyy_std); view(0,90); colorbar; colormap(cmap); caxis(clims_var); shading interp; axis image; title('Predicted std','Interpreter','latex','FontSize',16);

%% error std
disp(['Error std: ' num2str(std([epsxx_pred(:); epsxy_pred(:); epsyy_pred(:)]...
    -[epsxx_fea(:); epsxy_fea(:); epsyy_fea(:)])-2e-5)])
disp(['Error rel: ' num2str(norm([epsxx_pred(:); epsxy_pred(:); epsyy_pred(:)]...
    -[epsxx_fea(:); epsxy_fea(:); epsyy_fea(:)])/norm([epsxx_fea(:); epsxy_fea(:); epsyy_fea(:)]))])

%% remove path
rmpath(genpath('../'))
rmpath(genpath('../../Cantilevered'))