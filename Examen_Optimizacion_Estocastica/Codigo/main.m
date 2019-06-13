
X = [1.5;1.5];
Alpha = [0.9; 0.1]
epsilon = 0.5 ;
delta = 0.1;
DF = Example1Jacobian(X);



scatter(0,0);
pause(5);
%%Solving Weight vector....
[FX, X, Alpha] = BiObjective_Continuation(X, Alpha, epsilon, delta, @Example1, @Example1Jacobian);


%Bi-Objective Continuation
function [FX, X, Alpha] = BiObjective_Continuation(X, Alpha, epsilon, delta, F, JF)

  n = length(X); 
  tol = 10;
  ParetoPointRecord = [0, 0];
  while (1-Alpha(2)) > delta
      %%compute q2 as in equation (32) ..
      [Q R] = qr(Alpha);
      %%compute v as in (33) ..
      DF = JF(X);
      V =  linsolve(DF ,Q(:,2:n));
      V = V/norm(V);
      %%compute t as in (40)
      t =  epsilon./(abs((DF*V)));
      t = min(t);
       P =  X - sign(Q(2,2))*t*V;       

      %%compute x_{i+1} by solving (IVP_alpha) with initial value p_i and
      %%using alpha_i
      X = Directed_Search_Method(P, Alpha, F, JF);
       %%compute alpha_{i+1} as in (28)
      DF = JF(X);
     Alpha = lsqlin(transpose(DF),zeros(n,1),-eye(n), zeros(n,1), ones(1,n), 1, zeros(n,1), ones(n,1)); 
     %%%%%%%%%%%%Drawing points...
     FParetoOptimal = computeParetoFrontExample1(200);
     scatter(FParetoOptimal(:,1), FParetoOptimal(:,2), 'c');
      hold all; 
      scatter(ParetoPointRecord(:,1), ParetoPointRecord(:,2), 'k');
     FX = F(X); 
     scatter(FX(1), FX(2), 'b');
     f1 = 0:0.01:7;
     f2 = 0:0.01:7;
     [F1,F2] = meshgrid(f1,f2);
     Z = F1*Alpha(1) + F2*Alpha(2);
     str_f = sprintf('\\leftarrow Point Pareto point of the with weight vector \\alpha = [%0.2f, %0.2f]',Alpha(1), Alpha(2));
     text(FX(1)+0.1, FX(2),str_f, 'Color', 'r');
     contour(F1,F2,Z);
     xlim([0 7]); ylim([0 7]);
     xlabel('F_1(x)');
     ylabel('F_2(x)');
     ParetoPointRecord = [ParetoPointRecord; FX];
     pause(0.5);
     hold off; 
  end

  
  
end

function X = Directed_Search_Method(X, Alpha, F, JF)
   tol = 1e8;
   cont = 0;
   DF = JF(X);
      F(X)
   while cond(DF) < tol && cont < 10
       %compute pseudo inverse..
       v = -pinv(DF)*Alpha;
       v = v/norm(v);
       t = 100.0;
       %decreasing procedure based in armijo condition
       while any(F(X + t*v) >  (F(X) + 0.0001*t*DF*v) )
           t = t*0.5;
       end
       X = X + t*v;
       FX = F(X);
     
%hold all;
       DF = JF(X);
       cont = cont+1
   end
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
function FParetoOptimal = computeParetoFrontExample1(NPoints)
   Xtrue = linspace(-1.5,1.5,NPoints); %-2 + (4)*rand(100, 1);
   Ytrue = -Xtrue;

%Xtrue = -2 + (4)*rand(NPoints , 1);
%Ytrue = -2 + (4)*rand(NPoints , 1);

FParetoOptimal = zeros(NPoints ,2);
for i = 1:NPoints 
   FParetoOptimal(i,:) = Example1( [Xtrue(i); Ytrue(i)]); 
end


end
