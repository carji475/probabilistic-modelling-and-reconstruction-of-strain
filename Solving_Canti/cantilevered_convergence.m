clearvars;

load rot_noise0as_noise_paperdata_96proj
clearvars -except Fxx Fxy Fyy
addpath(genpath('../'))

load cantilevered_processed
load ang_red

%% save the full data
y_meas_full = y_meas;
nrSegs_full = nrSegs;
obs_full    = obs;

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
x = sort(Fxx.Points(abs(Fxx.Points(:,2)-min(Fxx.Points(:,2)))<1e-6,1))';
y = sort(Fxx.Points(abs(Fxx.Points(:,1)-min(Fxx.Points(:,1)))<1e-6,2))';
[X,Y] = meshgrid(x,y);

%% "true" (fea) solution
epsxx_fea = Fxx([X(:) Y(:)]);
epsxy_fea = Fxy([X(:) Y(:)]);
epsyy_fea = Fyy([X(:) Y(:)]);
epsxx_fea = reshape(epsxx_fea,ny,nx);
epsxy_fea = reshape(epsxy_fea,ny,nx);
epsyy_fea = reshape(epsyy_fea,ny,nx);

%% expand domain
Lx= Cx*xmax; Ly=Cy*ymax;

noise_std_vec = [0.25 0.5 0.75 1 1.25]*1e-4;
n_projs_vec   = 1:5:96;

ERROR_STD = zeros(length(n_projs_vec),length(noise_std_vec));
ERROR_REL = ERROR_STD;

h = waitbar(0,'Please wait...');
for noise_std_vec_index=1:length(noise_std_vec)
    noise_std = noise_std_vec(noise_std_vec_index);
    for n_projs_vec_index=1:length(n_projs_vec)
        n_projs = n_projs_vec(n_projs_vec_index);
        
        %% pick out the data
        projs_ind = round(linspace(1,96,n_projs));
        indices = [];
        for qq=projs_ind
            indices = [indices proj_indices(qq,1):proj_indices(qq,2)];
        end
        n_obs = length(indices);
        
        addPrevSegs = [0; cumsum(nrSegs_full(1:end-1)-1)]';
        indices_tweak = indices+addPrevSegs(indices);
        
        nrSegs_tweak = [indices_tweak; (indices_tweak+1).*(nrSegs_full(indices)'==2)];
        nrSegs_tweak = nrSegs_tweak(:)';
        nrSegs_tweak = nrSegs_tweak(nrSegs_tweak~=0);
        
        y_meas = y_meas_full(indices);
        nrSegs = nrSegs_full(indices);
        
        obs = obs_full(:,nrSegs_tweak);
        
        %% noise
        rng default
        y_meas=y_meas+noise_std*randn(size(y_meas));
        
        %% specify segment information
        addPrevSegs=[0; cumsum(nrSegs(1:end-1)-1)]; % #"additional" segments before previous measurement
        
        %% find hyperparameters by optimising the marginal likelihood
        % start guesses, try a few different
        start_guesses = [1 1 1 1]; % set last element to NaN if common length scale
        
        % set optimisation options for fminunc
        options = optimoptions('fminunc','Algorithm','quasi-newton','Display','none',...
            'GradObj','on','TolFun',1e-8,'TolX',1e-8);
        % call the optimising routine
        par_opt = GP_strainFieldOpt(obs,y_meas,m_1,m_2,Lx,Ly,nrSegs,addPrevSegs,E,v,options,start_guesses,covFunc);
        
        %% compute the reconstruction
        % call the computational routine
        [epsxx_pred,epsxy_pred,epsyy_pred]=...
            GP_strainFieldRec(obs,y_meas,[X(:) Y(:)],m_1,m_2,Lx,Ly,nrSegs,...
            addPrevSegs,par_opt(1),par_opt(2:end-1),par_opt(end),E,v,covFunc,0);
        
        % reshape into matrices
        epsxx_pred = reshape(epsxx_pred,ny,nx);
        epsxy_pred = reshape(epsxy_pred,ny,nx);
        epsyy_pred = reshape(epsyy_pred,ny,nx);
        
        %% error std
        ERROR_STD(n_projs_vec_index,noise_std_vec_index) = std([epsxx_pred(:); epsxy_pred(:); epsyy_pred(:)]-...
            [epsxx_fea(:); epsxy_fea(:); epsyy_fea(:)]);
        ERROR_REL(n_projs_vec_index,noise_std_vec_index) = norm([epsxx_pred(:); epsxy_pred(:); epsyy_pred(:)]...
            -[epsxx_fea(:); epsxy_fea(:); epsyy_fea(:)])/norm([epsxx_fea(:); epsxy_fea(:); epsyy_fea(:)]);
        
        %% waitbar
        waitbar(((noise_std_vec_index-1)*length(n_projs_vec)+n_projs_vec_index)/(length(n_projs_vec)*length(noise_std_vec)),h)
    end
end
delete(h)

%% plot
symbols=['v'; '+'; 'x'; '*'; '^'];
figure(1); 
for qq=1:length(noise_std_vec)
    plot(n_projs_vec(:),ERROR_STD(:,qq)-2e-5,['' symbols(qq) '-'])
    hold on; 
end
axis([1 96 0 0.5*1e-4])
legend([num2str(1e4*noise_std_vec(:),'%0.2f') repmat('e-4',length(noise_std_vec),1)],'location','Best')
xlabel('Number of Projections - N [-]','Interpreter','latex')
ylabel('Error, Standard Dev [-]','Interpreter','latex')
% grid on

%% relative error
symbols=['v'; '+'; 'x'; '*'; '^'];
figure(3); 
for qq=4%1:length(noise_std_vec)
    plot(n_projs_vec(:),100*ERROR_REL(:,qq),['' symbols(qq) '-'])
    hold on; 
end
% axis([1 96 0 2])
% legend([num2str(1e4*noise_std_vec(:),'%0.2f') repmat('e-4',length(noise_std_vec),1)],'location','Best')
xlabel('Number of Projections - N [-]','Interpreter','latex')
ylabel('Relative Error [\%]','Interpreter','latex')

%% remove path
rmpath(genpath('../'))