function [Q,out] = l_ls_ikf(system_trajectory,L_trajectory,xhat_trajectory,...
    K_trajectory,y_trajectory,P0,t,test)
    [n,p,r] = find_dimensions(xhat_trajectory,y_trajectory);
    Qmap = create_map(n);
    Rmap = create_map(p);
    Bqbar = create_Bbar(Qmap);
    Brbar = create_Bbar(Rmap);
    [Abar_trajectory,Cbar_trajectory,Lbar_trajectory] = ...
        find_augmented_matrix_trajectories(system_trajectory,L_trajectory,K_trajectory,r);
    [Psi,Phiq,Phir] = create_markov_parameters(Abar_trajectory,Cbar_trajectory,Lbar_trajectory,Bqbar,Brbar,r,t);
    Phiq = sum_Phi(Phiq,Bqbar,r-t);
    [~,z_trajectory_data] = find_innovation_trajectory(xhat_trajectory,y_trajectory,system_trajectory,r,p);
    y = find_innovation_covariance_trajectory(z_trajectory_data,t,r,p);

    %...............
    % S = test.C*test.P*test.C' + test.R;
    % y = TEST_find_theoretical_output_trajectory(S,t,r);
    Phipinv = psuedo_inv(Phiq);
    Pv = vec(P0);
    Rv = vec(test.R);
    dy = y - Psi*Pv - Phir*Rv;
    ue = Phipinv*dy;
    Q = reconstruct_Q(ue,Qmap);

    out.ue = ue;
    out.Psi = Psi;
    out.Phiq = Phiq;
    out.Phir = Phir;
    out.Abar_traj = Abar_trajectory;
    out.Cbar_traj = Cbar_trajectory;
    out.Lbar_traj = Lbar_trajectory;
    out.y = y;
end

function [x] = vec(x)
    x = reshape(x,[],1);
end

function [x] = psuedo_inv(x)
    [U,Sigma,V] = svd(x);
    x = V*pinv(Sigma)*U';
end

function [n,p,r] = find_dimensions(xhat_trajectory,y_trajectory)
    n = length(xhat_trajectory{1});
    p = length(y_trajectory{1});
    r = length(xhat_trajectory) - 1;
end

function map = create_map(n)
    nn = n*n;
    idx = 1:nn;
    idx = reshape(idx,n,n);
    idxt = idx';
    change = zeros(nn,1);
    map = zeros(nn,1);
    j = 1;
    for i = 1:nn
        if change(idx(i)) == 0
            map(i) = j;
            change(idx(i)) = j;
            change(idxt(i)) = j;
            j = j + 1;
        else
            map(i) = change(idx(i));
        end
    end
end

function Bbar = create_Bbar(Qmap)
    nn = length(Qmap);
    Bbar = zeros(nn,max(Qmap));
    for i = 1:max(Qmap)
        Bbari = zeros(nn,1);
        Bbari(Qmap == i) = 1;
        Bbar(:,i) = Bbari;
    end
end

function [Abar_trajectory,Cbar_trajectory,Lbar_trajectory] = find_augmented_matrix_trajectories(system_trajectory,L_trajectory,K_trajectory,r)
    Abar_trajectory = cell(r,1);
    Cbar_trajectory = cell(r,1);
    Lbar_trajectory = cell(r,1);
    for i = 1:r
        sysi = system_trajectory{i};
        Ai = sysi{1};
        Bi = sysi{2};
        Ci = sysi{3};
        Ki = K_trajectory{i};
        Li = L_trajectory{i};
        Acli = Ai - Bi*Ki;
        Abar_trajectory{i} = kron(Acli-Li*Ci,Acli-Li*Ci);
        Cbar_trajectory{i} = kron(Ci,Ci);
        Lbar_trajectory{i} = kron(Li,Li);
    end
end

function [Psi,Phiq,Phir] = create_markov_parameters(Abar_trajectory,Cbar_trajectory,Lbar_trajectory,Bqbar,Brbar,r,t)
    C = Cbar_trajectory{1};
    Psi = C;
    Phiiq = zeros(size(C,1),size(Bqbar,2)*r);
    Phiq = Phiiq;
    Phiir = eye(size(C,1));
    Phir = Phiir;
    for i = 1:r-t
        A = Abar_trajectory{i};
        C = Cbar_trajectory{i};
        L = Lbar_trajectory{i};
        Psi = [Psi ; C*A^i];
        Phiiq = [C*A^(i-1)*Bqbar Phiiq];
        Phiiq = Phiiq(:,1:end-size(Bqbar,2));
        Phiq = [Phiq ; Phiiq];
        Phiir = Phiir + C*A^(i-1)*L;
        Phir = [Phir; Phiir];
    end
    p = size(C,1);
    Psi = Psi(p+1:end,:);
    Phiq = Phiq(p+1:end,:);
    Phir = Phir(p+1:end,:);
end

function [Phi] = sum_Phi(Phi,B,r)
    Phi = Phi(:,1:r*size(B,2));
    for i = 1:r-1
        Phi_i = Phi(:,1:size(B,2)) + Phi(:,size(B,2)+1:2*size(B,2));
        Phi = [Phi_i Phi(:,2*size(B,2)+1:end)];
    end
end

function [z_trajectory,z_trajectory_data] = find_innovation_trajectory(xhat_trajectory,y_trajectory,system_trajectory,r,p)
    z_trajectory = cell(r,1);
    z_trajectory_data = zeros(r,p);
    for i = 2:r+1
        sysi = system_trajectory{i};
        Ci = sysi{3};
        z_trajectory{i} = y_trajectory{i} - Ci*xhat_trajectory{i-1};
        z_trajectory_data(i,:) = z_trajectory{i}';
    end
end

function [y] = find_innovation_covariance_trajectory(z_trajectory_data,t,r,p)
    y = zeros(p*p*(r-t),1);
    for i = 1:r-t
        % y(p*p*(i-1)+1:p*p*i) = vec(robustcov(z_trajectory_data(i:i+t,:)));
        y(p*p*(i-1)+1:p*p*i) = vec(robustcov(z_trajectory_data));
    end
end

function [y] = TEST_find_theoretical_output_trajectory(S,t,r)
    y = [];
    for i = 1:r-t
        y = [y ; vec(S)];
    end
end

function Q = reconstruct_Q(Qin,Qmap)
    nn = length(Qmap);
    n = sqrt(nn);
    Q = zeros(n);
    for i = 1:nn
        Q(i) = Qin(Qmap(i));
    end
end