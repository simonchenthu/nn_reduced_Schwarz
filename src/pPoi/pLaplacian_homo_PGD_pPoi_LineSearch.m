function u = pLaplacian_homo_PGD_pPoi_LineSearch(x0,y0,dx,a,p,bdy_w,bdy_e,bdy_s,bdy_n)
%%% Solver for equation  -\nabla\cdot(a|\nabla u|^{p-2}\nabla u) = 0 with 
%%% Dirichlet Boundary. The Ritz form \int a*|\nabla (w+g)|^p dx 
%%% with finite elements on uniform grid is used. The preconditioned 
%%% gradient desent (with line searching using fminunc and p-Poisson 
%%% preconditioner) is used to minimize the functional. (u = w + g)

Nx = length(x0); Ny = length(y0);
% [xx,yy] = meshgrid(x0,y0);

% barycenter of traingle
x_t = (x0(2:end)+x0(1:end-1)+x0(1:end-1))/3; 
y_t = (y0(2:end)+y0(2:end)+y0(1:end-1))/3;
x_b = (x0(2:end)+x0(2:end)+x0(1:end-1))/3; 
y_b = (y0(2:end)+y0(1:end-1)+y0(1:end-1))/3;

% a at the barycenter of traingle
a_t = a(x_t,y_t'); a_b = a(x_b,y_b');

% Extension of BC
g = LinearElliptic(x0,y0,a,bdy_w,bdy_e,bdy_s,bdy_n);

% Initial u
w = zeros(Ny,Nx);
u = w + g;

% Regularization Constant
epsilon = nthroot(1e-6,p-2);

% Initial step size
lambda = 1;
% options = optimoptions(@fmincon,'MaxIterations',1500,...
%     'OptimalityTolerance',1e-12,'StepTolerance',1e-12,'ConstraintTolerance',1e-12,...
%     'MaxFunctionEvaluations',10000);
options = optimoptions(@fminunc,'MaxIterations',1500,...
    'OptimalityTolerance',1e-12,'StepTolerance',1e-12,'MaxFunctionEvaluations',10000,...
    'Display','off');

% Initial preconditioner
P = speye((Nx-2)*(Ny-2));

% Initial functional value
J_old = J_homo(w(2:end-1,2:end-1),g,a_t,a_b,p,Ny,Nx,dx);
    
% Iteration begins
TOL = 1e-13; err = 1; iter = 0;
while err > TOL && iter <=2000
    % Compute gradient in each triangle
    dx_u = (u(:,2:end)-u(:,1:end-1))/dx;
    dy_u = (u(2:end,:)-u(1:end-1,:))/dx;
    
    dx_u_sq = dx_u.^2; dy_u_sq = dy_u.^2;

    % Compute norm of gradient in each triangle
    grad_u_t = sqrt(dx_u_sq(2:end,:)+dy_u_sq(:,1:end-1)); % triangle on the top
    grad_u_b = sqrt(dx_u_sq(1:end-1,:)+dy_u_sq(:,2:end)); % triangle on the bottom
    
    % Compute grad J
    a_grad1_u_t = a_t.*(grad_u_t.^(p-2));
    a_grad1_u_b = a_b.*(grad_u_b.^(p-2));
    
    a_grad1_u_dxu_t = a_grad1_u_t.*dx_u(2:end,:);
    a_grad1_u_dyu_t = a_grad1_u_t.*dy_u(:,1:end-1);
    
    a_grad1_u_dxu_b = a_grad1_u_b.*dx_u(1:end-1,:);
    a_grad1_u_dyu_b = a_grad1_u_b.*dy_u(:,2:end);
    
    grad_J = - a_grad1_u_dyu_t(2:end,2:end) ...                                    % Triangle #1
             - a_grad1_u_dxu_b(2:end,2:end) ...                                    % Triangle #2
             + a_grad1_u_dyu_t(1:end-1,2:end) - a_grad1_u_dxu_t(1:end-1,2:end)...  % Triangle #3
             + a_grad1_u_dyu_b(1:end-1,1:end-1)...                                 % Triangle #4
             + a_grad1_u_dxu_t(1:end-1,1:end-1)...                                 % Triangle #5
             - a_grad1_u_dyu_b(2:end,1:end-1) + a_grad1_u_dxu_b(2:end,1:end-1);    % Triangle #6
    grad_J = grad_J*(dx/2);                                                      % There is no p in this term!!
    grad_J = grad_J(:);
    
    
    % p-Poisson preconditioner
    grad_u_t_reg = max(grad_u_t,epsilon); % triangle on the top
    grad_u_b_reg = max(grad_u_b,epsilon); % triangle on the bottom
    
    a_grad1_u_t_reg = a_t.*(grad_u_t_reg.^(p-2));
    a_grad1_u_b_reg = a_b.*(grad_u_b_reg.^(p-2));
    
    co_c = (a_grad1_u_t_reg(2:end,2:end) ...
            + a_grad1_u_b_reg(2:end,2:end) ...
            + 2*a_grad1_u_t_reg(1:end-1,2:end) ...
            + a_grad1_u_b_reg(1:end-1,1:end-1)...
            + a_grad1_u_t_reg(1:end-1,1:end-1) ...
            + 2*a_grad1_u_b_reg(2:end,1:end-1))/2;
    co_c = co_c(:);
    
    co_n = -(a_grad1_u_t_reg(2:end,2:end)+a_grad1_u_b_reg(2:end,1:end-1))/2;
    co_n(end,:) = 0; co_n = co_n(:); co_n = [0;co_n(1:end-1)];
    
    co_s = -(a_grad1_u_t_reg(1:end-1,2:end)+a_grad1_u_b_reg(1:end-1,1:end-1))/2;
    co_s(1,:) = 0; co_s = co_s(:); co_s = [co_s(2:end);0];
    
    co_e = -(a_grad1_u_t_reg(1:end-1,2:end)+a_grad1_u_b_reg(2:end,2:end))/2;
    co_e(:,end) = 0; co_e = co_e(:); co_e = [zeros(Ny-2,1);co_e(1:end-Ny+2)];
    
    co_w = -(a_grad1_u_t_reg(1:end-1,1:end-1)+a_grad1_u_b_reg(2:end,1:end-1))/2;
    co_w(:,1) = 0; co_w = co_w(:); co_w = [co_w(Ny-1:end);zeros(Ny-2,1)];
    
    P = spdiags([co_w co_s co_c co_n co_e],[-(Ny-2) -1 0 1 Ny-2],P);

    % Preconditioning
    P_grad_J = P\grad_J; P_grad_J = reshape(P_grad_J,Ny-2,Nx-2);
    
    % Line searching
    fun = @(lambda)J_homo(w(2:end-1,2:end-1)-lambda*P_grad_J,...
        g,a_t,a_b,p,Ny,Nx,dx);
%     lambda = fmincon(fun,lambda,[],[],[],[],0,[],[],options);
    [lambda,J_new] = fminunc(fun,lambda,options);
    
%     % First optimality (residual in variational form)
%     err = norm(grad_J,'fro')
    
    % Relative Residual
    err = abs((J_old-J_new)/J_old);
    
    % Update solution
    w(2:end-1,2:end-1) = w(2:end-1,2:end-1) - lambda*P_grad_J;
    u = w + g;
    
    % Update functional value
    J_old = J_new;
    
    iter = iter + 1;
    
end

end
