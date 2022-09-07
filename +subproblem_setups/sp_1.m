classdef sp_1
    methods (Static)
        function [P, S] = setup()
            P.p1 = rand_vec;
            P.k = rand_normal_vec;
            S.theta = rand_angle;

            P.p2 = rot(P.k,S.theta)*P.p1;
        end

        function S = run(P)
            S.theta = subproblem1_linear(P.p1,P.p2,P.k);
        end

        function S = run_grt(P)
            S.theta = subproblem1(P.p1,P.p2,P.k);
        end

        function S = run_mex(P)
            S.theta = subproblem1_mex(P.p1,P.p2,P.k);
        end

        function e = error(P, S)
            norm(P.p2 - rot(P.k,S.theta)*P.p1)
        end
    end
end