clearvars; clc

addpath(genpath('../'))

load conv_res_predPointsSameAsScattered.mat; clearvars -except ERROR_REL n_projs_vec
load beam_bound_data.mat

semilogy([angs_bound(1); angs_bound(4:end)],100*[relres_bound(1); relres_bound(4:end)],'-x')
hold on

symbols=['v'; '+'; 'x'; '*'; '^'];
qq=4;
semilogy(n_projs_vec(:),100*ERROR_REL(:,qq),['' symbols(qq) '-'])
% axis([1 96 0 2])
% legend([num2str(1e4*noise_std_vec(:),'%0.2f') repmat('e-4',length(noise_std_vec),1)],'location','Best')
xlabel('Number of Projections - N [-]','Interpreter','latex','FontSize',14)
ylabel('Relative Error [\%]','Interpreter','latex','FontSize',14)

grid on
legend 'Wensrich et al' 'GP'