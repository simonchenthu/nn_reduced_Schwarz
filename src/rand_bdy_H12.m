function [bdy_s,bdy_n,bdy_w,bdy_e] = rand_bdy_H12(N_train,dim_r,radius_n,...
                                                x_patch,y_patch,dx)
% This function generates the random boundary condition on local patch
% (j,k) (Free boundary patches)
                                            
                                            
Nx_patch_b = length(x_patch);
Ny_patch_b = length(y_patch);


% compute weights W (order: SENW) and linear transform C
bdy_x = [x_patch,x_patch(end)*ones(1,Ny_patch_b-1),...
    fliplr(x_patch(1:end-1)),x_patch(1)*ones(1,Ny_patch_b-2)];

bdy_y = [y_patch(1)*ones(1,Nx_patch_b),y_patch(2:end),...
    y_patch(end)*ones(1,Nx_patch_b-1),fliplr(y_patch(2:end-1))];

disq =(bdy_x-bdy_x').^2+(bdy_y-bdy_y').^2;

W = 2*dx^2./disq; W(isnan(W)|isinf(W)) = 0;
W = diag( ones(1,2*(Nx_patch_b-1)+2*(Ny_patch_b-1))*W + dx ) - W;

C = chol(W);

% generate random samples
rho = rand(1,N_train); rho = nthroot(rho,dim_r);

% Generate unit vectors
x = randn(2*(Nx_patch_b-1)+2*(Ny_patch_b-1),N_train);
x_norm = sqrt(ones(1,2*(Nx_patch_b-1)+2*(Ny_patch_b-1))*x.^2);
x = C\x;
x_unit = x./x_norm;

% Boundary conditions
bdy_rand = radius_n*rho.*x_unit;

bdy_s = bdy_rand(1:Nx_patch_b,:);
bdy_e = bdy_rand(Nx_patch_b:(Nx_patch_b-1)+(Ny_patch_b-1)+1,:);
bdy_n = flipud(bdy_rand((Nx_patch_b-1)+(Ny_patch_b-1)+1:2*(Nx_patch_b-1)+(Ny_patch_b-1)+1,:));
bdy_w = [bdy_rand(1,:);...
    flipud(bdy_rand(2*(Nx_patch_b-1)+(Ny_patch_b-1)+1:end,:))];



end