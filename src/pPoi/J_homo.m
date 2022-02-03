function J = J_homo(w,g,a_t,a_b,p,Ny,Nx,dx)
% Compute the variational functional of 1/p * \int a*|\nabla (w+g)|^p

w = reshape(w,Ny-2,Nx-2);
g(2:end-1,2:end-1) = g(2:end-1,2:end-1) + w;

dux_t = (g(2:end,2:end) - g(2:end,1:end-1))/dx;
duy_t = (g(2:end,1:end-1) - g(1:end-1,1:end-1))/dx;

dux_b = (g(1:end-1,2:end) - g(1:end-1,1:end-1))/dx;
duy_b = (g(2:end,2:end) - g(1:end-1,2:end))/dx;

J = dx^2/(2*p)*ones(1,Ny-1)*(a_t.*(sqrt(dux_t.^2+duy_t.^2).^p) ...
                           + a_b.*(sqrt(dux_b.^2+duy_b.^2).^p))*ones(Nx-1,1);

end