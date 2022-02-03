% Online iteration of the neural network-based Schwarz iteration
clear;
addpath(genpath('../src'))

%%  Domain Parameters  %%
Lx = 1.0; Nx = 2^(8); Dx_overlap = 2^(-5); 
dx = Lx/Nx; x0 = 0:dx:Lx; Nx = length(x0); Mx = 8; 
Ly = 1.0; Ny = 2^(8); Dy_overlap = 2^(-5); 
dy = Ly/Ny; y0 = 0:dy:Ly; Ny = length(y0); My = 8; 

x_sw_o = 0:Lx/Mx:Lx-Lx/Mx; x_sw_o = max(x_sw_o-Dx_overlap,0);
y_sw_o = 0:Ly/My:Ly-Ly/My; y_sw_o = max(y_sw_o-Dy_overlap,0);
x_ne_o = Lx/Mx:Lx/Mx:Lx; x_ne_o = min(x_ne_o+Dx_overlap,Ly);
y_ne_o = Ly/My:Ly/My:Ly; y_ne_o = min(y_ne_o+Dy_overlap,Ly);

Ny_patch = (y_ne_o-y_sw_o)/dx+1;
Nx_patch = (x_ne_o-x_sw_o)/dx+1;

[xx,yy] = meshgrid(x0,y0);                    


% Interior patch & boundary patch
j_int = repmat(2:Mx-1,My-2,1); k_int = repmat((2:My-1)',1,Mx-2);
label_int = [j_int(:),k_int(:)];

j = repmat(1:Mx,My,1); k = repmat((1:My)',1,Mx);
j(2:end-1,2:end-1) = 0; k(2:end-1,2:end-1) = 0; 
label_bdy = [j(j~=0),k(k~=0)];


%%  Equation Parameters  %%
p = 6;

eps1 = 1/5; eps2 = 1/13; eps3 = 1/17; eps4 = 1/31; eps5 = 1/65;

a = @(x,y) 1/6*( (1.1+sin(2*pi*x/eps1))./(1.1+sin(2*pi*y/eps1)) ...
               + (1.1+sin(2*pi*y/eps2))./(1.1+cos(2*pi*x/eps2)) ...
               + (1.1+cos(2*pi*x/eps3))./(1.1+sin(2*pi*y/eps3)) ...
               + (1.1+sin(2*pi*y/eps4))./(1.1+cos(2*pi*x/eps4)) ...
               + (1.1+cos(2*pi*x/eps5))./(1.1+sin(2*pi*y/eps5)) ...
               + sin(4*x.^2.*y.^2) + 1);
a_n = 1;

bdy_D_n = @(x) sin(2*pi*x); bdy_no_n = 1;
bdy_D_s = @(x) -sin(2*pi*x); bdy_no_s = 1;
bdy_D_w = @(y) sin(2*pi*y); bdy_no_w = 1;
bdy_D_e = @(y) -sin(2*pi*y);  bdy_no_e = 1;

% bdy_D_n = @(x) sin(4*pi*x); bdy_no_n = 2;
% bdy_D_s = @(x) -sin(4*pi*x); bdy_no_s = 2;
% bdy_D_w = @(y) sin(4*pi*y); bdy_no_w = 2;
% bdy_D_e = @(y) -sin(4*pi*y);  bdy_no_e = 2;
 
% bdy_D_n = @(x) ones(size(x)); bdy_no_n = 3;
% bdy_D_s = @(x) -ones(size(x)); bdy_no_s = 3;
% bdy_D_w = @(y) -1+2*y.^2; bdy_no_w = 3;
% bdy_D_e = @(y) -1+2*y.^2;  bdy_no_e = 3;


%%  Reference Solution  %%
% dx_ref = Lx*2^(-8);
% x_ref = 0:dx_ref:Lx; y_ref = 0:dx_ref:Ly;
% Nx_ref = length(x_ref); Ny_ref = length(y_ref);
% [xx_ref,yy_ref] = meshgrid(x_ref,y_ref);
% 
% load(fullfile('data_pPoi',['u_ref_pPoi_p',int2str(p),...
%     '_bdy',int2str(bdy_no_n),int2str(bdy_no_s),int2str(bdy_no_w),int2str(bdy_no_e),...
%     '_a',int2str(a_n),'_dx',num2str(dx_ref),'.mat']),'u_ref','t_ref');
% 
% figure(110)
% mesh(xx_ref,yy_ref,u_ref); 
% xlabel('$x$','FontSize',14,'Interpreter','latex'); 
% ylabel('$y$','FontSize',14,'Interpreter','latex'); 
% zlabel('$u^\ast$','FontSize',14,'Interpreter','latex');


%%         Load NNs         %%

%%%   Training Info of NN  %%%
N_train = 1000;       
Dx_buffer = 2^(-4) + 2^(-5);

%%%  Load Pretrained NN  %%%
Op_bdy = cell(1,size(label_int,1));
for i = 1:size(label_int,1)
        j = label_int(i,1); k = label_int(i,2);
        
        load(fullfile('data_pPoi',['NN_param',...
            '_Mx',int2str(Mx),'_My',int2str(My),'_(',int2str(j),',',int2str(k),')',...
            '_Ntrain',int2str(N_train),'_dxb',sprintf('%.3e',Dx_buffer),...
            '.mat']),'fc1_bias','fc1_weight','fc2_bias','fc2_weight');
        
        Op_bdy{i} = @(x)NN_Twolayers(x,fc1_weight,fc2_weight,...
                                                    fc1_bias',fc2_bias',dx);

end




%%  Schwartz ieration  %%
tic
% NN-Schwarz / Reduced Schwarz
[u,qq,flag] = deep_Schwarz_elliptic_pPoi(x_sw_o,y_sw_o,x_ne_o,y_ne_o,dx,...
                                          Op_bdy,label_int,label_bdy,...
                                          p,a,...
                                          bdy_D_s,bdy_D_n,bdy_D_w,bdy_D_e);
t_total = toc;


%% Plot
figure(301)
mesh(xx,yy,u);
xlabel('$x$','FontSize',14,'Interpreter','latex'); 
ylabel('$y$','FontSize',14,'Interpreter','latex'); 
title('Patched Approximate Solution');









