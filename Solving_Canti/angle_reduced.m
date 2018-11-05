clearvars

load cantilevered_processed

proj_indices = zeros(96,2);
ang_red = ang(1);
ind     = 1;

proj_indices(1,1)=1;
proj_indices(end,end)=length(ang);

for qq=2:length(ang)
    if abs(ang(qq)-ang_red(ind))>1e-4
        proj_indices(ind,2) = qq-1;
        ind = ind+1;
        proj_indices(ind,1) = qq;
        ang_red(ind) = ang(qq);
    end
end
ang_red=ang_red(:);

save ang_red ang_red proj_indices