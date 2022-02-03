%%%  This code generate REFERENCE SOLUTIONS for comparison  %%%
clear;
addpath(genpath('../src'))
%%  Domain parametes  %%
Lx = 1.0; Nx = 2^(8); dx = Lx/Nx; x0 = 0:dx:Lx; Nx = length(x0); 
Ly = 1.0; Ny = 2^(8); dy = Ly/Ny; y0 = 0:dy:Ly; Ny = length(y0);

%%  Equation parameters  %%

p = 6;

eps1 = 1/5; eps2 = 1/13; eps3 = 1/17; eps4 = 1/31; eps5 = 1/65;

a = @(x,y) 1/6*( (1.1+sin(2*pi*x/eps1))./(1.1+sin(2*pi*y/eps1)) ...
               + (1.1+sin(2*pi*y/eps2))./(1.1+cos(2*pi*x/eps2)) ...
               + (1.1+cos(2*pi*x/eps3))./(1.1+sin(2*pi*y/eps3)) ...
               + (1.1+sin(2*pi*y/eps4))./(1.1+cos(2*pi*x/eps4)) ...
               + (1.1+cos(2*pi*x/eps5))./(1.1+sin(2*pi*y/eps5)) ...
               + sin(4*x.^2.*y.^2) + 1);
a_n = 1;

% bdy_D_n = @(x) sin(2*pi*x); bdy_no_n = 1;
% bdy_D_s = @(x) -sin(2*pi*x); bdy_no_s = 1;
% bdy_D_w = @(y) sin(2*pi*y); bdy_no_w = 1;
% bdy_D_e = @(y) -sin(2*pi*y);  bdy_no_e = 1;
% 
% bdy_D_n = @(x) sin(4*pi*x); bdy_no_n = 2;
% bdy_D_s = @(x) -sin(4*pi*x); bdy_no_s = 2;
% bdy_D_w = @(y) sin(4*pi*y); bdy_no_w = 2;
% bdy_D_e = @(y) -sin(4*pi*y);  bdy_no_e = 2;

bdy_D_n = @(x) ones(size(x)); bdy_no_n = 3;
bdy_D_s = @(x) -ones(size(x)); bdy_no_s = 3;
bdy_D_w = @(y) -1+2*y.^2; bdy_no_w = 3;
bdy_D_e = @(y) -1+2*y.^2;  bdy_no_e = 3;




%%  Solver  %%

bdy_w = bdy_D_w(y0)';
bdy_e = bdy_D_e(y0)';
bdy_n = bdy_D_n(x0)';
bdy_s = bdy_D_s(x0)';

tic

u_ref = pLaplacian_homo_PGD_pPoi_LineSearch(x0,y0,dx,...
                                        a,p,bdy_w,bdy_e,bdy_s,bdy_n);
                               
t_ref = toc;


%%  Plot  %%
[xx,yy] = meshgrid(x0,y0);
figure(1)
mesh(xx,yy,u_ref); xlim([0,Lx]); ylim([0,Ly]);
xlabel('x'); ylabel('y'); zlabel('u');

%%  Save  %%

save(fullfile('data_pPoi',['u_ref_pPoi_p',int2str(p),...
    '_bdy',int2str(bdy_no_n),int2str(bdy_no_s),int2str(bdy_no_w),int2str(bdy_no_e),...
    '_a',int2str(a_n),'_dx',num2str(dx),'.mat']),'u_ref','t_ref');

