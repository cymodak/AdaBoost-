function [eigenface] = get_eigen_face(data, histEqFlag, normFlag, nrows, ncolumns)
%% Input arguments:
% data: D x N matrix there each column is a stretched face vector
% histEqFlag: logical flag (true/false) indicating whether histeq is used
% normFlag: logical flag (true/false) indicating whether data norm is used

%% Output arguments:
% eigenFace: D x 1 eigenface vector 

%% Get input dimensions
[dim , N] = size(data);

%% Histogram Equalization
if(histEqFlag)
    % Fill your histogram eq code here
    for i=1:1:N
        data(:,i) = double(histeq(uint8(data(:,i))));
    end
end

%% Data Normalization
if(normFlag)
    % Fill your data normalization code here
    for i=1:1:N
        data(:,i) = (data(:,i)-mean(data(:,i)))/norm(data(:,i));
    end
end

%% Get Final Eigen Face (With either SVD or Eig Decomp)
% Fill your code to compute eigen-face
% Eigenvectors of covariance matrix
[U,S,V] = svds(data,1);

% First eigenface
eigenface = -1*reshape(U(:,1),nrows,ncolumns);

end