%  Generate solutions on each interior patch, and save them on separated files
clear;
addpath(genpath('../src'))
%%  Domain parameters  %%%%
Lx = 1.0; Nx = 2^(8); Dx_buffer = 2^(-5)+2^(-4); Dx_overlap = 2^(-5); 
dx = Lx/Nx; x0 = 0:dx:Lx; Nx = length(x0); Mx = 8; 
Ly = 1.0; Ny = 2^(8); Dy_buffer = 2^(-5)+2^(-4); Dy_overlap = 2^(-5); 
dy = Ly/Ny; y0 = 0:dy:Ly; Ny = length(y0); My = 8; 

x_sw_o = 0:Lx/Mx:Lx-Lx/Mx; x_sw_o = max(x_sw_o-Dx_overlap,0);
y_sw_o = 0:Ly/My:Ly-Ly/My; y_sw_o = max(y_sw_o-Dy_overlap,0);
x_ne_o = Lx/Mx:Lx/Mx:Lx; x_ne_o = min(x_ne_o+Dx_overlap,Ly);
y_ne_o = Ly/My:Ly/My:Ly; y_ne_o = min(y_ne_o+Dy_overlap,Ly);

x_sw_b = 0:Lx/Mx:Lx-Lx/Mx; x_sw_b = max(x_sw_b-Dx_overlap-Dx_buffer,0);
y_sw_b = 0:Ly/My:Ly-Ly/My; y_sw_b = max(y_sw_b-Dy_overlap-Dy_buffer,0);
x_ne_b = Lx/Mx:Lx/Mx:Lx; x_ne_b = min(x_ne_b+Dx_overlap+Dx_buffer,Ly);
y_ne_b = Ly/My:Ly/My:Ly; y_ne_b = min(y_ne_b+Dy_overlap+Dy_buffer,Ly);


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

%%  Sample  Parameters  %%

N_train = 1000;

radius_n = 10;
          
dim_r = 3;
          

          
%%  Iteration starts  %%
% rng('default');
rng('shuffle');
for k = 2:My-1                           
    for j = 2:Mx-1                       
        
        t_start = tic;
        
        %% Generate random boundary conditions
        % x/y range for buffered patches
        x_patch_b = x_sw_b(j):dx:x_ne_b(j);
        y_patch_b = y_sw_b(k):dx:y_ne_b(k);
        
        [bdy_s,bdy_n,bdy_w,bdy_e] = rand_bdy_H12(N_train,dim_r,radius_n,...
                                                x_patch_b,y_patch_b,dx);
        
        
        %%       Local Solver     %%
        
        x_patch_o = x_sw_o(j):dx:x_ne_o(j);
        y_patch_o = y_sw_o(k):dx:y_ne_o(k);
        
        Nx_patch_o = length(x_patch_o);
        Ny_patch_o = length(y_patch_o);
        
        DNx_b1 = Dx_buffer/dx; 
        DNx_b2 = Dx_buffer/dx;
        DNy_b1 = Dy_buffer/dx;
        DNy_b2 = Dy_buffer/dx;
        DNx_bo1 = (Dx_buffer+2*Dx_overlap)/dx;
        DNx_bo2 = (Dx_buffer+2*Dx_overlap)/dx;
        DNy_bo1 = (Dy_buffer+2*Dy_overlap)/dx;
        DNy_bo2 = (Dy_buffer+2*Dy_overlap)/dx;
        
        
        phi = zeros(N_train, 2*Nx_patch_o+2*Ny_patch_o);
        phi_int = zeros(N_train, 2*Nx_patch_o+2*Ny_patch_o);
        
        for i=1:N_train
            
            u_temp = pLaplacian_homo_PGD_pPoi_LineSearch(x_patch_b,y_patch_b,dx,...
                a,p,bdy_w(:,i),bdy_e(:,i),bdy_s(:,i),bdy_n(:,i));
            
            
            % [South, North, West, East]
            phi(i,:) = [u_temp(DNy_b1+1,DNx_b1+1:end-DNx_b2),...
                        u_temp(end-DNy_b2,DNx_b1+1:end-DNx_b2),...
                        u_temp(DNy_b1+1:end-DNy_b2,DNx_b1+1)',...
                        u_temp(DNy_b1+1:end-DNy_b2,end-DNx_b2)'];
            
            phi_int(i,:) = [u_temp(DNy_bo1+1,DNx_b1+1:end-DNx_b2),...
                            u_temp(end-DNy_bo2,DNx_b1+1:end-DNx_b2),...
                            u_temp(DNy_b1+1:end-DNy_b2,DNx_bo1+1)',...
                            u_temp(DNy_b1+1:end-DNy_b2,end-DNx_bo2)'];
            
        end
        
        t_dic = toc(t_start);
        
        
        %% Save
        save(fullfile('data_pPoi',['data',...
            '_Mx',int2str(Mx),'_My',int2str(My),'_(',int2str(j),',',int2str(k),')',...
            '_Ntrain',int2str(N_train),'_dxb',sprintf('%.3e',Dx_buffer),...
            '.mat']),'phi','phi_int','t_dic');
        
    end
end





