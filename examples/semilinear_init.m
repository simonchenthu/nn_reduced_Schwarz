%  Generate initial weights for two-layer NNs using linear elliptic Green's
%  functions
clear;
addpath(genpath('../src'))
%%  Domain parameters  %%

Lx = 1.0; Nx = 2^(8); Dx_overlap = 2^(-4); 
dx = Lx/Nx; x0 = 0:dx:Lx; Nx = length(x0); Mx = 4;
Ly = 1.0; Ny = 2^(8); Dy_overlap = 2^(-4); 
dy = Ly/Ny; y0 = 0:dy:Ly; Ny = length(y0); My = 4;

x_sw_o = 0:Lx/Mx:Lx-Lx/Mx; x_sw_o = max(x_sw_o-Dx_overlap,0);
y_sw_o = 0:Ly/My:Ly-Ly/My; y_sw_o = max(y_sw_o-Dy_overlap,0);
x_ne_o = Lx/Mx:Lx/Mx:Lx; x_ne_o = min(x_ne_o+Dx_overlap,Ly);
y_ne_o = Ly/My:Ly/My:Ly; y_ne_o = min(y_ne_o+Dy_overlap,Ly);


%%  Equation parameters  %%

n = 4;
epsilon = 2^(-n);

f = @(u) zeros(size(u));
del_f = @(u) zeros(size(u));

a = @(x,y) 2+sin(2*pi*x).*cos(2*pi*y)...
    +(2+1.8*sin(2*pi*x/epsilon))./(2+1.8*cos(2*pi*y/epsilon))...
    +(2+sin(2*pi*y/epsilon))./(2+1.8*cos(2*pi*x/epsilon));
a_n = 1;

%% Thresholding
% N_neuron = 300;
svd_threshold = 1e-2;

          
%%  Iteration starts  %%
for k = 2:My-1
    for j = 2:Mx-1
        
        t_start = tic;
        
        x_patch_o = x_sw_o(j):dx:x_ne_o(j);
        y_patch_o = y_sw_o(k):dx:y_ne_o(k);
        
        Nx_patch_o = length(x_patch_o);
        Ny_patch_o = length(y_patch_o);
        
        N_train = 2*Nx_patch_o + 2*Ny_patch_o;
        
        %% sample boundary conditions
        bdy_temp = eye(2*Nx_patch_o+2*Ny_patch_o,N_train);
        
        bdy_s = bdy_temp(1:Nx_patch_o,:);
        bdy_n = bdy_temp(Nx_patch_o+1:2*Nx_patch_o,:);
        bdy_w = bdy_temp(2*Nx_patch_o+1:2*Nx_patch_o+Ny_patch_o,:);
        bdy_e = bdy_temp(2*Nx_patch_o+Ny_patch_o+1:end,:);
        
        %%       Standard Solver     %%
        DNx_o1 = (2*Dx_overlap)/dx;
        DNx_o2 = (2*Dx_overlap)/dx;
        DNy_o1 = (2*Dy_overlap)/dx;
        DNy_o2 = (2*Dy_overlap)/dx;
        
        
        phi_int = zeros(N_train,2*Ny_patch_o+2*Nx_patch_o);
        for i=1:N_train
            
            u_temp = semilinear_elliptic_newton(x_patch_o,y_patch_o,dx,f,del_f,a,...
                                        bdy_w(:,i),bdy_e(:,i),bdy_s(:,i),bdy_n(:,i));
            
            % [South, North, West, East]
            phi_int(i,:) = [u_temp(DNy_o1+1,:),...
                        u_temp(end-DNy_o2,:),...
                        u_temp(:,DNx_o1+1)',...
                        u_temp(:,end-DNx_o2)'];
            
        end
        
        [U,S,V] = svd(phi_int);
        
        rk = find(diag(S)<svd_threshold,1)-1;
        
        S_init = diag(S); S_init = S_init(1:rk);
        
        U_init = [U(:,1:rk).*sqrt(S_init)',-U(:,1:rk).*sqrt(S_init)'];
        V_init = [V(:,1:rk).*sqrt(S_init)',-V(:,1:rk).*sqrt(S_init)'];
        
        N_neuron = 2*rk;
        
        t_dic = toc(t_start);
            
        save(fullfile('data_semilinear',['init',...
            '_Mx',int2str(Mx),'_My',int2str(My),'_(',int2str(j),',',int2str(k),')',...
            '.mat']),'U_init','V_init','U','V','S','N_neuron','t_dic');
        
    end
end





