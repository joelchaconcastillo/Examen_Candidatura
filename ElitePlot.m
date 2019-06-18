x = linspace(-1*pi,8*pi);
y = linspace(0,10*pi);
a = linspace(0,1);
[X,Y] = meshgrid(x,y);
Z = sin(X)+cos(Y);

figure
contour(X,Y,Z)
N =100;
NSelect = 10;
min = 0;
max = 25;
POOL = min + (max-min)*rand(N, 2);
F = zeros(N, 1);
for i =1:N
   viscircles(POOL(i,:), 0.2, 'Color', 'r');
   F(i, 1) = evaluate(POOL(i,:));
end
active = zeros(N,1);
penalized = zeros(N, 1);
selected = zeros(N, 1);

 [P bestindex] = min(F(active<1, 1))
   selected(bestindex, 1) =1;
 active(bestindex, 1) = 1;
 return
while length(selected(selected>0, 1)) < NSelect && length(active(active<1, 1)) > 0%|| length(penalized(penalized>0)) < NSelect 
        %%select the best point
        [P bestindex] = min(F(active<1, 1))
        selected(bestindex, 1) =1;
        for i = 1:N
            if active(i, 1) > 0
               continue;
            end
             for i = 1:N
                 if i == bestindex
                    continue;
                 end
                 if norm(POOL(bestindex,:) - POOL(i,:)) < 10
                     penalized(i, 1) = 1;
                     active(i, 1) = 1;
                 end
             end

        end
                    active(bestindex, 1) = 1;
end



%viscircles([14 19], 0.2, 'Color', 'r')
%viscircles([15 25], 0.2, 'Color', 'gr')
%viscircles([14 19], 8,'Color', 'b', 'LineStyle', ':')
function F = evaluate(X)
 F = sin(X(1)) + cos(X(2))
 
end