function x = NN_Twolayers(x,fc1_weight,fc2_weight,fc1_bias,fc2_bias,dx)
% Two layer NN with normalization layers

Ex = mean(x,1);
x = x-Ex;
Varx = max(sqrt(dx)*sqrt(sum(x.^2,1)),1e-8);
x = x./Varx;

x = max(fc1_weight*x+fc1_bias,0);
x = fc2_weight*x+fc2_bias;

x = Varx.*x + Ex;

end