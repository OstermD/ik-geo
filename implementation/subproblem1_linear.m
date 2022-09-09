function [theta, is_LS] = subproblem1_linear(p1, p2, k)
%  p2 = rot(k, theta)*p1

KxP = cross(k, p1);
A = [KxP -cross(k,KxP)];

x = A'*p2;

theta = atan2(x(1),x(2));

if nargout > 1
    is_LS = abs(norm(p1) - norm(p2)) > 1e-8 || abs(dot(k,p1) - dot(k,p2)) > 1e-8;
end
end