function u = semilinear_elliptic_newton(x0,y0,dx,f,del_f,a,...
                                                   bdy_w,bdy_e,bdy_s,bdy_n)
%%% Solver for equation  -\nabla\cdot(a\nabla u) + f(u) =0 with Dirichlet
%%% Boundary. Finite volume scheme is used, Newton iteration is used to
%%% solve the nonlinear system.

Nx = length(x0); Ny = length(y0);
[xx,yy] = meshgrid(x0,y0);

u = zeros(Ny,Nx);

% if boundary datas are functions
% u(:,1) = data_left(y0); u(:,end) = data_right(y0);
% u(1,:) = data_down(x0); u(end,:) = data_up(x0);
% if boundary datas are matrices
u(:,1) = bdy_w; u(:,end) = bdy_e;
u(1,:) = bdy_s'; u(end,:) = bdy_n';

a_e = a((xx(2:end-1,3:end)+xx(2:end-1,2:end-1))/2,(yy(2:end-1,3:end)+yy(2:end-1,2:end-1))/2);
a_w = a((xx(2:end-1,1:end-2)+xx(2:end-1,2:end-1))/2,(yy(2:end-1,1:end-2)+yy(2:end-1,2:end-1))/2);
a_n = a((xx(3:end,2:end-1)+xx(2:end-1,2:end-1))/2,(yy(3:end,2:end-1)+yy(2:end-1,2:end-1))/2);
a_s = a((xx(1:end-2,2:end-1)+xx(2:end-1,2:end-1))/2,(yy(1:end-2,2:end-1)+yy(2:end-1,2:end-1))/2);

co_c = a_e+a_w+a_n+a_s; co_c = co_c(:);

co_n = -a_n; co_n(end,:) = 0; co_n = co_n(:); co_n = [0;co_n(1:end-1)];
co_s = -a_s; co_s(1,:) = 0; co_s = co_s(:); co_s = [co_s(2:end);0];
co_e = -a_e; co_e(:,end) = 0; co_e = co_e(:); co_e = [zeros(Ny-2,1);co_e(1:end-Ny+2)];
co_w = -a_w; co_w(:,1) = 0; co_w = co_w(:); co_w = [co_w(Ny-1:end);zeros(Ny-2,1)];

JF_base = spdiags([co_w co_s co_c co_n co_e],[-(Ny-2) -1 0 1 Ny-2],(Nx-2)*(Ny-2),(Nx-2)*(Ny-2));

% initial u
% u(2:end-1,2:end-1) = zeros(Ny-2,Nx-2);

%Newton iteration
TOL = 1e-8; err = 1;
while err > TOL
    % compute Jacobian
    dJF = dx^2*del_f(u(2:end-1,2:end-1)); dJF = dJF(:);
    dJF = dJF + co_c;
    JF = spdiags(dJF,0,JF_base);
    
    % compute stiffness vector
    F = (a_w+a_s+a_n+a_e).*u(2:end-1,2:end-1)-a_w.*u(2:end-1,1:end-2)-a_s.*u(1:end-2,2:end-1)...
        -a_n.*u(3:end,2:end-1)-a_e.*u(2:end-1,3:end)+dx^2*f(u(2:end-1,2:end-1));
    F = F(:);
    
    % solve increments
    du = JF\F; 
    du = reshape(du,Ny-2,Nx-2); 
    u(2:end-1,2:end-1) = u(2:end-1,2:end-1)-du;
    
    % error
    err = norm(du(:),2)/norm(u(:),2);
    
end

% figure(1)
% Lx = 1; Ly = 1; %plotz = 10;
% mesh(xx,yy,u);
% xlabel('x'); ylabel('y'); zlabel('u');
% xlim([0,Lx]); ylim([0,Ly]); zlim([-1,1]);
% pause;
% u = u(:);

end
