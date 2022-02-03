function POU = Partition_of_Unity(x_nw,y_nw,x_se,y_se,Nx,Ny)
% Lx = 1; Ly = 1; Mx = 4; My = 4; Dx_overlap = 2^(-3); Dy_overlap = 2^(-3);
% Nx = 2^(4+4); Ny = 2^(4+4);
% x_nw = 0:Lx/Mx:Lx-Lx/Mx; x_nw = max(x_nw-Dx_overlap,0);
% y_nw = 0:Ly/My:Ly-Ly/My; y_nw = max(y_nw-Dy_overlap,0);
% x_se = Lx/Mx:Lx/Mx:Lx; x_se = min(x_se+Dx_overlap,Ly);
% y_se = Ly/My:Ly/My:Ly; y_se = min(y_se+Dy_overlap,Ly);

Mx = length(x_nw); My = length(y_nw);
Lx = x_se(end); Ly = y_se(end);
x = linspace(0,Lx,Nx); y = linspace(0,Ly,Ny);
[xx,yy] = meshgrid(x,y); 

%L = @(x,alpha) max(0,1-x/alpha);          %  Prototype function
L = @(x,alpha) exp(-1./(1-min(x/alpha,1)));

POU = zeros(Ny,Nx,Mx,My);

for k = 1:My
    for j = 1:Mx
        xx_dis = abs(xx-(x_nw(j)+x_se(j))/2);
        yy_dis = abs(yy-(y_nw(k)+y_se(k))/2);
        
        alpha_x = (x_se(j)-x_nw(j))/2;
        alpha_y = (y_se(k)-y_nw(k))/2;
        
        POU(:,:,j,k) = L(xx_dis,alpha_x).*L(yy_dis,alpha_y);
        
%         figure(99)
%         mesh(xx,yy,POU(:,:,j,k));
    end
end

weight = sum(sum(POU,3),4);
POU = POU./weight;
POU(isnan(POU)|isinf(POU)) = 0;

POU(1,:,:,:) = POU(2,:,:,:);
POU(end,:,:,:) = POU(end-1,:,:,:);
POU(:,1,:,:) = POU(:,2,:,:);
POU(:,end,:,:) = POU(:,end-1,:,:);

% for k = 1:My
%     for j = 1:Mx
%         figure(99)
%         mesh(xx,yy,POU(:,:,j,k));
%         pause;
%     end
% end

end