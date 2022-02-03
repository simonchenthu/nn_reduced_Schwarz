function u = LinearElliptic(x0,y0,a,D_w,D_e,D_s,D_n)
%%% Solution map for equation  -\nabla\cdot(a\nabla u) =0. 
%%% Finite volume scheme is used.

Nx = length(x0); Ny = length(y0);
[xx,yy] = meshgrid(x0,y0);

D_n = D_n'; D_s = D_s';
% if boundary datas are functions
% u(:,1) = data_left(y0); u(:,end) = data_right(y0);
% u(1,:) = data_down(x0); u(end,:) = data_up(x0);
% if boundary datas are matrices

% initial u
u = zeros(Ny,Nx);
u(:,1) = D_w; u(:,end) = D_e;
u(1,:) = D_s; u(end,:) = D_n;

% u_old = u(2:end-1,2:end-1); u_old = u_old(:);

% permeability matrix
a_e = a((xx(2:end-1,3:end)+xx(2:end-1,2:end-1))/2,(yy(2:end-1,3:end)+yy(2:end-1,2:end-1))/2);
a_w = a((xx(2:end-1,1:end-2)+xx(2:end-1,2:end-1))/2,(yy(2:end-1,1:end-2)+yy(2:end-1,2:end-1))/2);
a_n = a((xx(3:end,2:end-1)+xx(2:end-1,2:end-1))/2,(yy(3:end,2:end-1)+yy(2:end-1,2:end-1))/2);
a_s = a((xx(1:end-2,2:end-1)+xx(2:end-1,2:end-1))/2,(yy(1:end-2,2:end-1)+yy(2:end-1,2:end-1))/2);

co_c = a_e+a_w+a_n+a_s; co_c = co_c(:);

co_n = -a_n; co_n(end,:) = 0; co_n = co_n(:); co_n = [0;co_n(1:end-1)];
co_s = -a_s; co_s(1,:) = 0; co_s = co_s(:); co_s = [co_s(2:end);0];
co_e = -a_e; co_e(:,end) = 0; co_e = co_e(:); co_e = [zeros(Ny-2,1);co_e(1:end-Ny+2)];
co_w = -a_w; co_w(:,1) = 0; co_w = co_w(:); co_w = [co_w(Ny-1:end);zeros(Ny-2,1)];

JF = spdiags([co_w co_s co_c co_n co_e],[-(Ny-2) -1 0 1 Ny-2],(Nx-2)*(Ny-2),(Nx-2)*(Ny-2));



% compute stiffness vector
F = zeros(Ny-2,Nx-2);
F(end,:) = a_n(end,:).*D_n(2:end-1);
F(1,:) = F(1,:) + a_s(1,:).*D_s(2:end-1);
F(:,1) = F(:,1) + a_w(:,1).*D_w(2:end-1);
F(:,end) = F(:,end) + a_e(:,end).*D_e(2:end-1);
F = F(:);

% solve increments
u_new = JF\F;
u(2:end-1,2:end-1) = reshape(u_new,Ny-2,Nx-2); 


% figure(1)
% mesh(xx,yy,u); % xlim([0,1]); ylim([0,1]); % zlim([-plotz,plotz]);



    
end
