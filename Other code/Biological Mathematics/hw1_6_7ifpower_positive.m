L=[0 1 0 4;
   .2 0 0 0;
   0 .8 0 0;
   0 0 .3 0];
power_positive = 0;
%L is power_positive if L(1,3)==2 and not so if L(1,3)==0
%Please modify on your own by change the parameter "power_positive"
%%%%%%%%%%%%%%%%%%%%%%
% For the exercise
%A non-positive matrix will lead to vanishment of some dimensions(some age groups)
%in this case, age2&4 group are missing starting from a single new-born
%%%%%%%%%%%%%%%%%%%%%%
if power_positive
    L(1,3) = 2;
end

n=size(L);
ifpp = all(all(L^(ind(n(1)))));
disp(ifpp);
disp(L^(ind(n(1))));
function power_index = ind(n)
    power_index = (n-1).^2+1;
end

