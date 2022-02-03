%%%  This code generate REFERENCE SOLUTIONS for comparison  %%%
clear;
addpath(genpath('../src'))
%%  Domain parametes  %%
n = 4;
epsilon = 2^(-n);

Lx = 1.0; Nx = 2^(8); dx = Lx/Nx; x0 = 0:dx:Lx; Nx = length(x0); 
Ly = 1.0; Ny = 2^(8); dy = Ly/Ny; y0 = 0:dy:Ly; Ny = length(y0);

%%  Equation parameters  %%

f = @(u) u.^3;
del_f = @(u) 3*u.^2;   
f_no = 1;

a = @(x,y) 2+sin(2*pi*x).*cos(2*pi*y)...
    +(2+1.8*sin(2*pi*x/epsilon))./(2+1.8*cos(2*pi*y/epsilon))...
    +(2+sin(2*pi*y/epsilon))./(2+1.8*cos(2*pi*x/epsilon));
a_n = 1;

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

% bdy_D_n = @(x) sin(2*pi*x); bdy_no_n = 1;
% bdy_D_s = @(x) -sin(2*pi*x); bdy_no_s = 1;
% bdy_D_w = @(y) sin(2*pi*y); bdy_no_w = 1;
% bdy_D_e = @(y) -sin(2*pi*y);  bdy_no_e = 1;

% bdy_D_n = @(x) ones(size(x)); bdy_no_n = 5;
% bdy_D_s = @(x) ones(size(x)); bdy_no_s = 5;
% bdy_D_w = @(y) ones(size(y)); bdy_no_w = 5;
% bdy_D_e = @(y) ones(size(y));  bdy_no_e = 5;

bdy_D_n = @(x) 40*ones(size(x)); bdy_no_n = 6;
bdy_D_s = @(x) 40*ones(size(x)); bdy_no_s = 6;
bdy_D_w = @(y) 40*ones(size(y)); bdy_no_w = 6;
bdy_D_e = @(y) 40*ones(size(y));  bdy_no_e = 6;

% bdy_D_n = @(x) 25+10*cos(2*pi*x); bdy_no_n = 7;
% bdy_D_s = @(x) 25+10*cos(2*pi*x); bdy_no_s = 7;
% bdy_D_w = @(y) 25+10*cos(2*pi*y); bdy_no_w = 7;
% bdy_D_e = @(y) 25+10*cos(2*pi*y);  bdy_no_e = 7;

% bdy_D_n = @(x) 35*ones(size(x)); bdy_no_n = 8;
% bdy_D_s = @(x) 10*ones(size(x)); bdy_no_s = 8;
% bdy_D_w = @(y) 10+25*y; bdy_no_w = 8;
% bdy_D_e = @(y) 10+25*y;  bdy_no_e = 8;

% bdy_D_n = @(x) 30+10*cos(2*pi*x)+5*sin(4*pi*x); bdy_no_n = 9;
% bdy_D_s = @(x) 30+10*cos(2*pi*x)-5*sin(4*pi*x); bdy_no_s = 9;
% bdy_D_w = @(y) 30+10*cos(2*pi*y)-5*sin(4*pi*y); bdy_no_w = 9;
% bdy_D_e = @(y) 30+10*cos(2*pi*y)+5*sin(4*pi*y);  bdy_no_e = 9;



%%  Solver  %%

bdy_w = bdy_D_w(y0)';
bdy_e = bdy_D_e(y0)';
bdy_n = bdy_D_n(x0)';
bdy_s = bdy_D_s(x0)';

tic

u_ref = semilinear_elliptic_newton(x0,y0,dx,f,del_f,a,...
                                   bdy_w,bdy_e,bdy_s,bdy_n);
      
t_ref = toc;


%%  Plot  %%   
[xx,yy] = meshgrid(x0,y0);
figure(1)
mesh(xx,yy,u_ref); xlim([0,Lx]); ylim([0,Ly]);
xlabel('x'); ylabel('y'); zlabel('u');

%%  Save  %%

save(fullfile('data_semilinear',['u_ref_fno',int2str(f_no),'_eps',int2str(n),...
    '_bdy',int2str(bdy_no_n),int2str(bdy_no_s),int2str(bdy_no_w),int2str(bdy_no_e),...
    '_a',int2str(a_n),'_dx',num2str(dx),'.mat']),'u_ref','t_ref');


%% Plot   %%
figure(666)
imagesc(x0,y0,a(x0,y0')); 
set(gca,'YDir','normal');
xlabel('$x$','Interpreter','latex','FontSize',14);
ylabel('$y$','Interpreter','latex','FontSize',14);

