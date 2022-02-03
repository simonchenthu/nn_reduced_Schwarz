function [u,q,flag] = deep_Schwarz_elliptic_pPoi(x_sw,y_sw,x_ne,y_ne,dx,...
                                          Op_bdy,label_int,label_bdy,...
                                          p,a,...
                                          bdy_D_s,bdy_D_n,bdy_D_w,bdy_D_e)
% This function peforms the Schwarz iteration acclerated by NNs
% The NN is a map from boundary to boundary
% The input of NN is the full BC, the output is the four neighboring
% boundaries


%%%  Auxiliary parameters  %%%
% Whole domain
x0 = x_sw(1):dx:x_ne(end); y0 = y_sw(1):dx:y_ne(end);
Nx = length(x0); Ny = length(y0);
% Number of patches
Mx = length(x_sw); My = length(y_sw); 
% Number of interior patches
N_int = size(label_int,1); N_bdy = size(label_bdy,1);
% Number of grid points of patches
Ny_patch = (y_ne-y_sw)/dx+1; 
Nx_patch = (x_ne-x_sw)/dx+1; 
% Location of patches
ny_sw = y_sw/dx+1; ny_ne = y_ne/dx+1;
nx_sw = x_sw/dx+1; nx_ne = x_ne/dx+1; 
% Patch coordinates
x_patch = cell(1,Mx); y_patch = cell(1,My);
for j = 1:Mx
    x_patch{j} = x_sw(j):dx:x_ne(j);
end
for k = 1:My
    y_patch{k} = y_sw(k):dx:y_ne(k);
end
% Location of neighboring boundaries
DNx_o =  (x_ne(1) - x_sw(2))/dx;
DNy_o = (y_ne(1) - y_sw(2))/dx;




%%%  Intial boundary on each patch  %%%
% Cells with zero boundary on each patch
Patch_bdy_s = cell(My,Mx); Patch_bdy_n = cell(My,Mx);
Patch_bdy_w = cell(My,Mx); Patch_bdy_e = cell(My,Mx);
for j = 1:Mx
    for k = 1:My
        Patch_bdy_s{k,j} = zeros(Nx_patch(j),1);
        Patch_bdy_n{k,j} = zeros(Nx_patch(j),1);
        Patch_bdy_w{k,j} = zeros(Ny_patch(k),1);
        Patch_bdy_e{k,j} = zeros(Ny_patch(k),1);
    end
end
% Physical boundary
for j = 1:Mx
    Patch_bdy_s{1,j} = bdy_D_s(x_patch{j})';
    Patch_bdy_n{end,j} = bdy_D_n(x_patch{j})';
end
for k = 1:My
    Patch_bdy_w{k,1} = bdy_D_w(y_patch{k})';
    Patch_bdy_e{k,end} = bdy_D_e(y_patch{k})';
end
% Cells used to update boundary conditions
Patch_bdy_s_new = Patch_bdy_s;
Patch_bdy_n_new = Patch_bdy_n;
Patch_bdy_w_new = Patch_bdy_w;
Patch_bdy_e_new = Patch_bdy_e;








% outputs = cell(My,Mx);
res = 1; q = 0;
tol = 1e-4;
while res > tol && q+1<=2000 % q = 1:iter
    
    q = q+1;
    
    
%     for j = 1:Mx
%         for k = 1:My
%         figure(1001)
%         plot3(x_patch{j},y_sw(k)*ones(Nx_patch(j)),Patch_bdy_s{k,j});
%         hold on;
%         plot3(x_patch{j},y_ne(k)*ones(Nx_patch(j)),Patch_bdy_n{k,j});
%         plot3(x_sw(j)*ones(Ny_patch(k)),y_patch{k},Patch_bdy_w{k,j});
%         plot3(x_ne(j)*ones(Ny_patch(k)),y_patch{k},Patch_bdy_e{k,j});
%         xlim([0,1]); ylim([0,1]);
%         pause;
%         end
%     end
%     hold off;
    
    
    % Compute interior patches by NNs
    for i = 1:N_int
        j = label_int(i,1); k = label_int(i,2);
        % Assemble input boundaries
        inputs = [Patch_bdy_s{k,j};Patch_bdy_n{k,j};...
                    Patch_bdy_w{k,j};Patch_bdy_e{k,j}];
            
        % Compute boundary updating operator
        outputs = Op_bdy{i}(inputs);
            
        % Update neighboring boundaries
        Patch_bdy_n_new{k-1,j} = outputs(1:Nx_patch(j));
        Patch_bdy_s_new{k+1,j} = outputs(1+Nx_patch(j):2*Nx_patch(j));
        Patch_bdy_e_new{k,j-1} = outputs(1+2*Nx_patch(j):2*Nx_patch(j)+Ny_patch(k));
        Patch_bdy_w_new{k,j+1} = outputs(1+2*Nx_patch(j)+Ny_patch(k):end);
    end
    
    % Compute boundary patches by classical solver
    for i = 1:N_bdy
        j = label_bdy(i,1); k = label_bdy(i,2);
%         u_temp = semilinear_elliptic_newton(x_patch{j},y_patch{k},dx,...
%                                                     f,del_f,a,...
%                                                    Patch_bdy_w{k,j},Patch_bdy_e{k,j},...
%                                                    Patch_bdy_s{k,j},Patch_bdy_n{k,j});
        u_temp = pLaplacian_homo_PGD_pPoi_LineSearch(x_patch{j},y_patch{k},dx,a,p,...
                                                   Patch_bdy_w{k,j},Patch_bdy_e{k,j},...
                                                   Patch_bdy_s{k,j},Patch_bdy_n{k,j});
        % Update neighboring boundaries
        if k ~= 1
            Patch_bdy_n_new{k-1,j} = u_temp(DNy_o+1,:)';
        end
        if k ~= My
            Patch_bdy_s_new{k+1,j} = u_temp(end-DNy_o,:)';
        end
        if j ~= 1
            Patch_bdy_e_new{k,j-1} = u_temp(:,DNx_o+1);
        end
        if j ~= Mx
            Patch_bdy_w_new{k,j+1} = u_temp(:,end-DNx_o);
        end
    end
    
    %%%       Compute Residual     %%%
    res_s = cellfun(@(x,y)x-y,Patch_bdy_s,Patch_bdy_s_new,'UniformOutput',false);
    res_n = cellfun(@(x,y)x-y,Patch_bdy_n,Patch_bdy_n_new,'UniformOutput',false);
    res_w = cellfun(@(x,y)x-y,Patch_bdy_w,Patch_bdy_w_new,'UniformOutput',false);
    res_e = cellfun(@(x,y)x-y,Patch_bdy_e,Patch_bdy_e_new,'UniformOutput',false);
    res = ones(1,Mx)*...
        sqrt(dx*cellfun(@(x1,x2,x3,x4)x1'*x1+x2'*x2+x3'*x3+x4'*x4,res_s,res_n,res_w,res_e))...
        *ones(My,1);

    
    %%%     Switch the variables    %%%
    Patch_bdy_s = Patch_bdy_s_new;
    Patch_bdy_n = Patch_bdy_n_new;
    Patch_bdy_w = Patch_bdy_w_new;
    Patch_bdy_e = Patch_bdy_e_new;
    
end




%%%   Compute the Local Solution on Each Patch  %%%
u_patch = cell(My,Mx);
for j = 1:Mx
    for k = 1:My

%         u_patch{k,j} = semilinear_elliptic_newton(x_patch{j},y_patch{k},dx,f,del_f,a,...
%                                                    Patch_bdy_w{k,j},Patch_bdy_e{k,j},...
%                                                    Patch_bdy_s{k,j},Patch_bdy_n{k,j});
        u_patch{k,j} = pLaplacian_homo_PGD_pPoi_LineSearch(x_patch{j},y_patch{k},dx,a,p,...
                                                   Patch_bdy_w{k,j},Patch_bdy_e{k,j},...
                                                   Patch_bdy_s{k,j},Patch_bdy_n{k,j});
                                               
        
                                               
%         % test
%         figure(999)
%         [xx_test,yy_test] = meshgrid(x_patch{j},y_patch{k});
%         mesh(xx_test,yy_test,u_patch{k,j}); hold on;
%         xlim([0,1]); ylim([0,1]);
%         pause;
    end
end







%%%   Output   %%%
% Generate Partition of Unity
POU = Partition_of_Unity(x_sw,y_sw,x_ne,y_ne,Nx,Ny);

% Patch the local solutions using POU
u = zeros(Ny,Nx);
for k = 1:My
    for j = 1:Mx
        
        u(ny_sw(k):ny_ne(k),nx_sw(j):nx_ne(j)) ...
                  = u(ny_sw(k):ny_ne(k),nx_sw(j):nx_ne(j))...
                    + u_patch{k,j}.*POU(ny_sw(k):ny_ne(k),nx_sw(j):nx_ne(j),j,k);
        
    end
end


%%%   Flag   %%%
if res <= tol 
    flag = 0;
else
    flag = -1;
end

% %%%   Plot   %%%
% [xx,yy] = meshgrid(x0,y0);
% figure(202)
% mesh(xx,yy,u); % zlim([-2,2]);
% xlabel('x'); ylabel('y'); zlabel('u');
% title('Patched Approximate Solution');


end