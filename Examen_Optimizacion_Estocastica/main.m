
X = [1.5;1.5];
Alpha = [0.6; 0.4]
epsilon = 0.5 ;
delta = 0.1;
DF = Example1Jacobian(X);

NPoints = 100000;
Xtrue = linspace(-1.5,1.5,NPoints); %-2 + (4)*rand(100, 1);
Ytrue = -Xtrue;

%Xtrue = -2 + (4)*rand(NPoints , 1);
%Ytrue = -2 + (4)*rand(NPoints , 1);

FParetoOptimal = zeros(NPoints ,2);
for i = 1:NPoints 
   FParetoOptimal(i,:) = Example1( [Xtrue(i); Ytrue(i)]); 
end


%%Solving Weight vector....
[FX, X, Alpha] = BiObjective_Continuation(X, Alpha, epsilon, delta, @Example1, @Example1Jacobian);



%%%%Drawing....
scatter(FParetoOptimal(:,1), FParetoOptimal(:,2), 'c');
hold all;
scatter(FX(1), FX(2),'r');


%Bi-Objective Continuation
function [FX, X, Alpha] = BiObjective_Continuation(X, Alpha, epsilon, delta, F, JF)

  n = length(X); 
  tol = 10;
  %while (1-Alpha(2)) > delta
      %%compute q2 as in equation (32) ..
      [Q R] = qr(Alpha);
      %%compute v as in (33) ..
      DF = JF(X);
      V =  linsolve(DF ,Q(:,2:n));
      %V = V/norm(V)
      %%compute t as in (40)
      t =  epsilon./(abs((DF*V)));
      t = min(t);
       P =  X - sign(Q(2,2))*t*V;       
%       while abs( dot(Alpha,(F(P) - F(X)) )) < tol
 %          t = t*2;
  %          P =  X - sign(Q(2,2))*t*V;
   %    end
      %%compute x_{i+1} by solving (IVP_alpha) with initial value p_i and
      %%using alpha_i
      X = Directed_Search_Method(P, Alpha, F, JF);
       %%compute alpha_{i+1} as in (28)
        DF = JF(X);
   %   Alpha = lsqlin(transpose(DF),zeros(n,1),-eye(n), zeros(n,1), ones(1,n), 1, zeros(n,1), ones(n,1));
      
%return;
  %end
  FX = F(X);
end

function X = Directed_Search_Method(X, Alpha, F, JF)
   tol = 1e8;
   cont = 0;
   DF = JF(X);
      F(X)
   while cond(DF) < tol && cont < 100
       %compute pseudo inverse..
       v = -pinv(DF)*Alpha;
       v = v/norm(v);
       t = 100.0;
       %armijo condition
       while any(F(X + t*v) >  (F(X) + 0.0001*t*DF*v) )
           t = t*0.5;
       end
       X = X + t*v;
       DF = JF(X);
       cont = cont+1;
   end
   F(X)
end
function JF = Example1Jacobian(X)
   lambda = 0.85;
   syms x y 
   f =[0.5*( sqrt(1.0+(x + y)*(x + y)) + sqrt(1.0 + (x-y)*(x-y)) + x - y ) + lambda * exp(-(x-y)*(x-y)), 0.5*( sqrt(1.0+(x + y)*(x + y)) + sqrt(1.0 + (x-y)*(x-y)) - x + y ) + lambda * exp(-(x-y)*(x-y)) ];
   v = [x, y];
   fp = jacobian(f,v);
   JF = double(subs(fp, [x y], transpose(X)));
end
function F = Example1(X)
   F = [];
   x = X(1);
   y = X(2);
   lambda = 0.85;
   F(1) = 0.5*( sqrt(1.0+(x + y)*(x + y)) + sqrt(1.0 + (x-y)*(x-y)) + x - y ) + lambda * exp(-(x-y)*(x-y));
   F(2) = 0.5*( sqrt(1.0+(x + y)*(x + y)) + sqrt(1.0 + (x-y)*(x-y)) - x + y ) + lambda * exp(-(x-y)*(x-y));
end