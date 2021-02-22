clear all;
close all;

addpath(genpath('../'))

load JPARC_data_moreProcessed
y_meas = y_meas-mean(y_meas); 

%% choose a random subset of the data (if not, set n_red=n_obs)
rng default
n_red = n_obs;
indices = sort(randperm(n_obs,n_red));

addPrevSegs = [0; cumsum(nrSegs(1:end-1)-1)]';
indices_tweak = indices+addPrevSegs(indices);

nrSegs_tweak = [indices_tweak; (indices_tweak+1).*(nrSegs(indices)'==2)];
nrSegs_tweak = nrSegs_tweak(:)';
nrSegs_tweak = nrSegs_tweak(nrSegs_tweak~=0);

y_meas = y_meas(indices);
nrSegs = nrSegs(indices);

obs = obs(:,nrSegs_tweak);
n_obs=n_red;

%% Approximation parameters
Cx = 2.5;
Cy = 2.5;

% #basis functions
m_1 = 30;       % x
m_2 = 30;      % y

%% covariance function
covFunc = struct('type','Matern','nu',2.5);

%% material parameters
E = 200e9;  % Pa
v = 0.3;

%% mesh
nx = 100; ny = 100;
xmin = -10e-3;   xmax = 10e-3;
ymin = -10e-3;   ymax = 10e-3;
x = linspace(xmin,xmax,nx);
y = linspace(ymin,ymax,ny);
if ~any(y)==0 || isempty(y); y=[y(y<0) 0 y(y>0)]; ny=ny+1; end
y_zero_index = find(y==0);

numr = 100;
numtheta = 360;
r1 = 3.5e-3;
r2 = 10e-3;
[r, theta] = meshgrid(linspace(r1,r2,numr),linspace(45,360-45,numtheta));
X = r.*cosd(theta);
Y = r.*sind(theta);

nx=size(X,2); ny=size(X,1);
indices = X<realmax;

xslice=linspace(-r2,-r1,100)'; 
yslice=zeros(100,1);

%% expand domain
Lx= Cx*r2; Ly=Cy*r2;

% specify segment information
addPrevSegs=[0; cumsum(nrSegs(1:end-1)-1)]; % #"additional" segments before previous measurement

%% find hyperparameters by optimising the marginal likelihood
% start guess
start_guesses = [1 0.001 0.001 1.5e-4];
% set optimisation options for fminunc
options = optimoptions('fminunc','Algorithm','quasi-newton','Display','iter-detailed',...
    'GradObj','on','TolFun',1e-8,'TolX',1e-8);
% call the optimising routine
par_opt = GP_strainFieldOpt(obs,y_meas,m_1,m_2,Lx,Ly,nrSegs,addPrevSegs,E,v,options,start_guesses,covFunc);

%% compute the reconstruction
% call the computational routine
[epsxx_preda,epsxy_preda,epsyy_preda,epsxx_vara,epsxy_vara,epsyy_vara]=...
    GP_strainFieldRec(obs,y_meas,[X(:) Y(:); xslice yslice],m_1,m_2,Lx,Ly,nrSegs,...
    addPrevSegs,par_opt(1),par_opt(2:end-1),par_opt(end),E,v,covFunc);

epsxx_pred = nan(ny,nx);     
epsxy_pred = nan(ny,nx);  
epsyy_pred = nan(ny,nx);
epsxx_pred(indices) = epsxx_preda(1:end-100);     
epsxy_pred(indices) = epsxy_preda(1:end-100);  
epsyy_pred(indices) = epsyy_preda(1:end-100);

epsxx_std = nan(ny,nx);     
epsxy_std = nan(ny,nx);  
epsyy_std = nan(ny,nx);
epsxx_std(indices) = sqrt( epsxx_vara(1:end-100) );     
epsxy_std(indices) = sqrt( epsxy_vara(1:end-100) );  
epsyy_std(indices) = sqrt( epsyy_vara(1:end-100) );

%% "true" (fea) solution
load JPARC_FEA
sc = 1.15; % scaling constant
epsxx_fea = nan(ny,nx);     
epsxy_fea = nan(ny,nx);  
epsyy_fea = nan(ny,nx);
epsxx_fea(indices) = sc*Fxx([X(:) Y(:)]);     
epsxy_fea(indices) = sc*Fxy([X(:) Y(:)]);  
epsyy_fea(indices) = sc*Fyy([X(:) Y(:)]);  

%% plot theory and reconstruction
load cmocean_bal
cmap=cmocean_bal;
clims_pred = [-1e-3, 1e-3];
clims_var = 1e-5*[1 10];

f1=figure(1);
clf
f1.Position = 0.8*[100 100 950 750];
xinsetl = 0.05;
xinsetr = 0.11;
yinsetb = 0.05;
yinsett = 0.06;
xgap = 0.000;
ygap = 0.030;
width = (1- xinsetl -xinsetr - 2*xgap)/3;
height = (1 - yinsetb -yinsett - 2*ygap)/3;
l1 = xinsetl;
l2 = l1+width+xgap;
l3 = l2+width+xgap;
b3 = yinsetb;
b2 = b3 + height+ygap;
b1 = b2+ygap+height;

% frame
k = boundary(X(:),Y(:));
Xb = X(k);
Yb = Y(k);

% epsxx
subplot('position',[l1,b1,width,height]); pcolor(X,Y,epsxx_fea); axis off;  colormap(cmap); caxis(clims_pred); shading interp; axis equal; line(Xb,Yb,'Color','k'); title('$\epsilon_{xx}$','Interpreter','latex','FontSize',16);
ylim=get(gca,'YLim');
xlim=get(gca,'XLim');
text(xlim(1)-0.001,ylim(1)+0.010,'FEA',...
'VerticalAlignment','middle',...
'HorizontalAlignment','center', 'rotation', 90,'interpreter','latex','fontsize',16)
subplot('position',[l1,b2,width,height]); pcolor(X,Y,epsxx_pred); axis off;  colormap(cmap); caxis(clims_pred); shading interp; axis equal; line(Xb,Yb,'Color','k');
text(xlim(1)-0.001,ylim(1)+0.010,'GP mean',...
'VerticalAlignment','middle',...
'HorizontalAlignment','center', 'rotation', 90,'interpreter','latex','fontsize',16)
subplot('position',[l1,b3,width,height]); pcolor(X,Y,epsxx_std); axis off;  colormap(cmap); caxis(clims_var); shading interp; axis equal; line(Xb,Yb,'Color','k');
text(xlim(1)-0.001,ylim(1)+0.010,'Predicted std',...
'VerticalAlignment','middle',...
'HorizontalAlignment','center', 'rotation', 90,'interpreter','latex','fontsize',16)

% epsxy
subplot('position',[l2,b1,width,height]); pcolor(X,Y,epsxy_fea); axis off;  colormap(cmap); caxis(clims_pred); shading interp; axis equal; line(Xb,Yb,'Color','k');title('$\epsilon_{xy}$','Interpreter','latex','FontSize',16);
subplot('position',[l2,b2,width,height]); pcolor(X,Y,epsxy_pred); axis off;  colormap(cmap); caxis(clims_pred); shading interp; axis equal; line(Xb,Yb,'Color','k');
subplot('position',[l2,b3,width,height]); pcolor(X,Y,epsxy_std); axis off;  colormap(cmap); caxis(clims_var); shading interp; axis equal; line(Xb,Yb,'Color','k');

% epsyy
subplot('position',[l3,b1,width,height]); pcolor(X,Y,epsyy_fea); axis off;  colormap(cmap); caxis(clims_pred); shading interp; axis equal; line(Xb,Yb,'Color','k');title('$\epsilon_{yy}$','Interpreter','latex','FontSize',16);
subplot('position',[l3,b2,width,height]); pcolor(X,Y,epsyy_pred); axis off;  colormap(cmap); caxis(clims_pred); shading interp; axis equal; line(Xb,Yb,'Color','k');
% colorbar1
legl = +0.065;
legb = -0.0225;
legw = +0.007;
legh = 0.35;
temp = get(gca,'position');
c = colorbar;
c.Position = c.Position + [legl legb legw legh];
c.TickLabelInterpreter = 'latex';
c.FontSize = 12;
% c.Label.String = 'Error $\times 10^4$ [-]';
% c.Label.Interpreter = 'latex';
% c.Label.FontSize = 18;
set(gca, 'Position', temp);

subplot('position',[l3,b3,width,height]); pcolor(X,Y,epsyy_std); axis off;  colormap(cmap); caxis(clims_var); shading interp; axis equal; line(Xb,Yb,'Color','k');
% colorbar2
legl = +0.065;
legb = -0.05;
legw = +0.007;
legh = 0.06;
temp = get(gca,'position');
c = colorbar;
c.Position = c.Position + [legl legb legw legh];
c.TickLabelInterpreter = 'latex';
c.FontSize = 12;
% c.Label.String = 'Error $\times 10^4$ [-]';
% c.Label.Interpreter = 'latex';
% c.Label.FontSize = 18;
set(gca, 'Position', temp);


%% slices
load COMP

% epsyy
figure
subplot(1,2,1)
plot(xslice,[1e6*epsyy_preda(end-99:end)],'-r','LineWidth',2); 
title('$\epsilon_{yy}$','Interpreter','latex','FontSize',16)
hold on; plot(radius_kow,1e6*EPSyy_kow,'og',radius_rec,1e6*EPSyy_rec,'-.k','LineWidth',2)%,radius_rec,EPSyy_fea)
plot(xslice,1e6*sc*Fyy([xslice yslice]),'--b','LineWidth',2);
grid on
legend GP KOW LS FEA location best
xlabel('$x$','Interpreter','latex')

% epsxx
subplot(1,2,2)
plot(xslice,[1e6*epsxx_preda(end-99:end)],'-r','LineWidth',2);% 1e6*epsxx_preda(end-99:end)+2*1e6*sqrt( epsxx_vara(end-99:end) ) 1e6*epsxx_preda(end-99:end)-2*1e6*sqrt( epsxx_vara(end-99:end) )],'-r','LineWidth',2); 
title('$\epsilon_{xx}$','Interpreter','latex','FontSize',16)
hold on; plot(radius_kow,1e6*EPSxx_kow,'og',radius_rec,1e6*EPSxx_rec,'-.k','LineWidth',2)%,radius_rec,EPSxx_fea)
plot(xslice,1e6*sc*Fxx([xslice yslice]),'--b','LineWidth',2);
grid on
legend GP KOW LS FEA location best
xlabel('$x$','Interpreter','latex')

%% error std
diff=[epsxx_pred(:); epsxy_pred(:); epsyy_pred(:)]...
    -[epsxx_fea(:); epsxy_fea(:); epsyy_fea(:)]-2e-5;
diff=diff(~isnan(diff));
disp(['Error std: ' num2str(std(diff))])
disp(['Error rel: ' num2str(norm(diff)/sqrt(nansum([epsxx_fea(:); epsxy_fea(:); epsyy_fea(:)].^2)))])