classdef linear_forward_kalman_filter
    %{
    This class sets up the forward problem for the linear forward Kalman
    filter. Given the system matrices and 
    %}
    properties
        A = [];
        B = [];
        C = [];
        Q = [];
        R = [];
        n = [];
        m = [];
        p = [];
        x = [];
        y = [];
        x_trajectory = {};
        u_trajectory = {};
        y_trajectory = {};
        system_trajectory = {};
        K_trajectory = {};
        noise_trajectory = {};
        P0 = [];
        L = [];
        P = [];
        Qnominal = [];
        Rnominal = [];
        xhat = [];
        L_trajectory = {};
        P_trajectory = {};
        xhat_trajectory = {};
    end
    
    methods
        function obj = linear_forward_kalman_filter(A,B,C,Q,R,P0,Qnominal,Rnominal,...
                x0,r,K)
            obj.A = A;
            obj.B = B;
            obj.C = C;
            obj.Q = Q;
            obj.R = R;
            obj.n = size(A,1);
            obj.m = size(B,2);
            obj.p = size(C,1);
            obj = obj.clear_trajectories();
            obj.P0 = P0;
            obj = obj.update_P(P0);
            obj = obj.update_Qnominal(Qnominal);
            obj = obj.update_Rnominal(Rnominal);
            if exist('x0','var') && exist('r','var') && exist('K','var')
                obj = obj.auto_generate_forward_problem(x0,r,K);
            end
        end

        function obj = update_A(obj,A)
            if size(A,1) == obj.n && size(A,2) == obj.n
                obj.A = A;
            else
                error(['A must be of dimension ' num2str(obj.n) 'x' ...
                    num2str(obj.n)]);
            end
        end

        function obj = update_B(obj,B)
            if size(B,1) == obj.n && size(B,2) == obj.m
                obj.B = B;
            else
                error(['B must be of dimension ' num2str(obj.m) 'x' ... 
                    num2str(obj.m)]);
            end
        end

        function obj = update_C(obj,C)
            if size(C,1) == obj.p && size(C,2) == obj.n
                obj.C = C;
            else
                error(['C must be of dimension ' num2str(obj.p) 'x' ... 
                    num2str(obj.n)]);
            end
        end

        function obj = update_Q(obj,Q)
            if ~(size(Q,1) == obj.n && size(Q,2) == obj.n)
                error(['Q must be of dimension ' num2str(obj.n) 'x' ...
                    num2str(obj.n)]);
            elseif ~(min(eig(Q)) >= 0)
                error('Q must be positive semi-definite');
            elseif ~(min(min(Q == Q')) == 1)
                error('Q must be symmetric');
            else
                obj.Q = Q;
            end
        end

        function obj = update_R(obj,R)
            if ~(size(R,1) == obj.p && size(R,2) == obj.p)
                error(['R must be of dimension ' num2str(obj.p) 'x' ...
                    num2str(obj.p)]);
            elseif ~(min(eig(R)) > 0)
                error('R must be positive definite');
            elseif ~(min(min(R == R')) == 1)
                error('R must be symmetric');
            else
                obj.R = R;
            end
        end

        function obj = clear_trajectories(obj)
            obj.x_trajectory = {};
            obj.u_trajectory = {};
            obj.y_trajectory = {};
            obj.system_trajectory = {};
            obj.noise_trajectory = {};
        end

        function obj = initialize_state(obj,x0)
            obj = obj.clear_trajectories();
            obj.x = x0;
            obj.y = obj.C*x0;
            obj.x_trajectory{1} = obj.x;
            obj.y_trajectory{1} = obj.y;
            obj.system_trajectory{1} = {obj.A,obj.B,obj.C};
            obj.noise_trajectory{1} = {obj.Q,obj.R};
        end

        function obj = update_state(obj,u)
            if ~isempty(obj.x)
                w = mvnrnd(zeros(obj.n,1),obj.Q)';
                v = mvnrnd(zeros(obj.p,1),obj.R)';
                obj.x = obj.A*obj.x + obj.B*u + w;
                obj.y = obj.C*obj.x + v;
                obj = obj.update_trajectories(u);
            else
                error('State needs to be initialized');
            end
        end

        function obj = update_trajectories(obj,u)
            obj.x_trajectory{length(obj.x_trajectory)+1} = obj.x;
            obj.u_trajectory{length(obj.u_trajectory)+1} = u;
            obj.y_trajectory{length(obj.y_trajectory)+1} = obj.y;
            obj.system_trajectory{length(obj.system_trajectory)+1} = ... 
                {obj.A, obj.B, obj.C};
            obj.noise_trajectory{length(obj.noise_trajectory)+1} = ...
                {obj.Q, obj.R};
        end

        function [obj,x_trajectory,u_trajectory,y_trajectory,system_trajectory] ... 
                = generate_trajectories(obj,r,K)
            x_trajectory = cell(r+1,1);
            u_trajectory = cell(r,1);
            y_trajectory = cell(r+1,1);
            system_trajectory = cell(r+1,1);

            x_trajectory{1} = obj.x;
            y_trajectory{1} = obj.y;
            system_trajectory{1} = {obj.A, obj.B, obj.C, obj.Q, obj.R};
            for i = 1:r
                u = -K*obj.x;
                obj = obj.update_state(u);
                x_trajectory{i+1} = obj.x_trajectory{end};
                u_trajectory{i} = obj.u_trajectory{end};
                y_trajectory{i} = obj.y_trajectory{end};
                system_trajectory = obj.system_trajectory{end};
            end
        end

        function obj = update_P(obj,P)
            if ~(size(P,1) == obj.n && size(P,2) == obj.n)
                error(['P must be of dimension ' num2str(obj.n) 'x' ...
                    num2str(obj.n)]);
            elseif ~(min(eig(P)) >= 0)
                error('P must be positive semi-definite');
            elseif ~(min(min(P == P')) == 1)
                error('P must be symmetric');
            else
                obj.P = P;
            end
        end

        function obj = update_Qnominal(obj,Qnominal)
            if ~(size(Qnominal,1) == obj.n && size(Qnominal,2) == obj.n)
                error(['Q nominal must be of dimension ' num2str(obj.n) 'x' ...
                    num2str(obj.n)]);
            elseif ~(min(eig(Qnominal)) >= 0)
                error('Q nominal must be positive semi-definite');
            elseif ~(min(min(Qnominal == Qnominal')) == 1)
                error('Q nominal must be symmetric');
            else
                obj.Qnominal = Qnominal;
            end
        end
       
        function obj = update_Rnominal(obj,Rnominal)
            if ~(size(Rnominal,1) == obj.p && size(Rnominal,2) == obj.p)
                error(['R nominal must be of dimension ' num2str(obj.n) 'x' ...
                    num2str(obj.n)]);
            elseif ~(min(eig(Rnominal)) >= 0)
                error('R nominal must be positive semi-definite');
            elseif ~(min(min(Rnominal == Rnominal')) == 1)
                error('R nominal must be symmetric');
            else
                obj.Rnominal = Rnominal;
            end
        end

        function obj = update_kalman_filter(obj)
            obj.P = obj.A*obj.P*obj.A' - ... 
                obj.A*obj.P*obj.C'/(obj.C*obj.P*obj.C' + obj.R)*obj.C*obj.P*obj.A' + obj.Q;
            obj.L = obj.A*obj.P*obj.C'/(obj.C*obj.P*obj.C' + obj.R);
            obj.P_trajectory{length(obj.P_trajectory)+1} = obj.P;
            obj.L_trajectory{length(obj.L_trajectory)+1} = obj.L;
        end

        function obj = update_state_estimate(obj,u)
            obj.xhat = obj.A*obj.xhat + obj.B*u + ... 
                obj.L*(obj.y - obj.C*(obj.A*obj.xhat + obj.B*u));
            obj.xhat_trajectory{length(obj.xhat_trajectory)+1} = obj.xhat;
        end

        function obj = initialize_state_estimate(obj,xhat0)
            obj.xhat = xhat0;
            obj.xhat_trajectory{length(obj.xhat_trajectory)+1} = obj.xhat;
            obj.L_trajectory{length(obj.L_trajectory)+1} = obj.A*obj.P*obj.C'/(obj.C*obj.P*obj.C' + obj.R);
            obj.P_trajectory{length(obj.P_trajectory)+1} = obj.P;
        end

        function obj = generate_estimate_trajectory(obj,r,K)
            for i = 1:r
                u = -K*obj.xhat;
                obj.K_trajectory{length(obj.K_trajectory)+1} = K;
                obj = obj.update_state(u);
                obj = obj.update_kalman_filter();
                obj = obj.update_state_estimate(u);
            end
        end

        function obj = auto_generate_forward_problem(obj,x0,r,K)
            obj = obj.initialize_state(x0);
            obj = obj.initialize_state_estimate(x0);
            obj.K_trajectory{length(obj.K_trajectory)+1} = K;
            obj = obj.generate_estimate_trajectory(r,K);
        end
    end
end

