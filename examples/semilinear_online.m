% Online iteration of the neural network-based Schwarz iteration
clear;
addpath(genpath('../src'))

%%  Domain Parameters  %%
Lx = 1.0; Nx = 2^(8); Dx_overlap = 2^(-4); 
dx = Lx/Nx; x0 = 0:dx:Lx; Nx = length(x0); Mx = 4;
Ly = 1.0; Ny = 2^(8); Dy_overlap = 2^(-4); 
dy = Ly/Ny; y0 = 0:dy:Ly; Ny = length(y0); My = 4;

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
n = 4;
epsilon = 2^(-n);

f = @(u) u.^3;
del_f = @(u) 3*u.^2; 
f_no = 1;

a = @(x,y) 2+sin(2*pi*x).*cos(2*pi*y)...
    +(2+1.8*sin(2*pi*x/epsilon))./(2+1.8*cos(2*pi*y/epsilon))...
    +(2+sin(2*pi*y/epsilon))./(2+1.8*cos(2*pi*x/epsilon));
a_n = 1;

% bdy_D_n = @(x) sin(2*pi*x); bdy_no_n = 1;
% bdy_D_s = @(x) -sin(2*pi*x); bdy_no_s = 1;
% bdy_D_w = @(y) sin(2*pi*y); bdy_no_w = 1;
% bdy_D_e = @(y) -sin(2*pi*y);  bdy_no_e = 1;

% bdy_D_n = @(x) 50+50*sin(2*pi*x); bdy_no_n = 2;
% bdy_D_s = @(x) 50-50*sin(2*pi*x); bdy_no_s = 2;
% bdy_D_w = @(y) 50+50*sin(2*pi*y); bdy_no_w = 2;
% bdy_D_e = @(y) 50-50*sin(2*pi*y);  bdy_no_e = 2;

% bdy_D_n = @(x) 10+sin(2*pi*x); bdy_no_n = 3;
% bdy_D_s = @(x) 10-sin(2*pi*x); bdy_no_s = 3;
% bdy_D_w = @(y) 10+sin(2*pi*y); bdy_no_w = 3;
% bdy_D_e = @(y) 10-sin(2*pi*y);  bdy_no_e = 3;

% bdy_D_n = @(x) 30+10*cos(2*pi*x); bdy_no_n = 4;
% bdy_D_s = @(x) 30+10*cos(2*pi*x); bdy_no_s = 4;
% bdy_D_w = @(y) 30+10*cos(2*pi*y); bdy_no_w = 4;
% bdy_D_e = @(y) 30+10*cos(2*pi*y);  bdy_no_e = 4;

% bdy_D_n = @(x) 30+10*cos(2*pi*x)+5*sin(4*pi*x); bdy_no_n = 9;
% bdy_D_s = @(x) 30+10*cos(2*pi*x)-5*sin(4*pi*x); bdy_no_s = 9;
% bdy_D_w = @(y) 30+10*cos(2*pi*y)-5*sin(4*pi*y); bdy_no_w = 9;
% bdy_D_e = @(y) 30+10*cos(2*pi*y)+5*sin(4*pi*y);  bdy_no_e = 9;

% bdy_D_n = @(x) 35*ones(size(x)); bdy_no_n = 8;
% bdy_D_s = @(x) 10*ones(size(x)); bdy_no_s = 8;
% bdy_D_w = @(y) 10+25*y; bdy_no_w = 8;
% bdy_D_e = @(y) 10+25*y;  bdy_no_e = 8;

% bdy_D_n = @(x) 25+10*cos(2*pi*x); bdy_no_n = 7;
% bdy_D_s = @(x) 25+10*cos(2*pi*x); bdy_no_s = 7;
% bdy_D_w = @(y) 25+10*cos(2*pi*y); bdy_no_w = 7;
% bdy_D_e = @(y) 25+10*cos(2*pi*y);  bdy_no_e = 7;

bdy_D_n = @(x) 40*ones(size(x)); bdy_no_n = 6;
bdy_D_s = @(x) 40*ones(size(x)); bdy_no_s = 6;
bdy_D_w = @(y) 40*ones(size(y)); bdy_no_w = 6;
bdy_D_e = @(y) 40*ones(size(y));  bdy_no_e = 6;



%%  Reference Solution  %%
% dx_ref = Lx*2^(-8);
% x_ref = 0:dx_ref:Lx; y_ref = 0:dx_ref:Ly;
% Nx_ref = length(x_ref); Ny_ref = length(y_ref);
% [xx_ref,yy_ref] = meshgrid(x_ref,y_ref);
% 
% load(fullfile('data_semilinear',['u_ref_fno',int2str(f_no),'_eps',int2str(n),...
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
N_train = 10000;      
Dx_buffer = 2^(-4);

%%%  Load NN  %%%
Op_bdy = cell(1,size(label_int,1));
for i = 1:size(label_int,1)
        j = label_int(i,1); k = label_int(i,2);
        load(fullfile('data_semilinear',['NN_param',...
            '_Mx',int2str(Mx),'_My',int2str(My),'_(',int2str(j),',',int2str(k),')',...
            '_Ntrain',int2str(N_train),'_dxb',sprintf('%.3e',Dx_buffer),...
            '.mat']),'fc1_bias','fc1_weight','fc2_bias','fc2_weight');
            
            Op_bdy{i} = @(x)fc2_weight*(max(fc1_weight*x+fc1_bias',0))+fc2_bias';
            
end




%%  Schwartz ieration  %%

tic
% NN-Schwarz
[u,qq,flag] = deep_Schwarz_elliptic(x_sw_o,y_sw_o,x_ne_o,y_ne_o,dx,...
                                          Op_bdy,label_int,label_bdy,...
                                          f,del_f,a,...
                                          bdy_D_s,bdy_D_n,bdy_D_w,bdy_D_e);

t_total = toc;







%% Plot
figure(302)
mesh(xx,yy,u);
xlabel('$x$','FontSize',14,'Interpreter','latex'); 
ylabel('$y$','FontSize',14,'Interpreter','latex'); 
title('Patched Approximate Solution');









