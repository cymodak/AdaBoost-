close all
clear all
clc

% Variable Definitions
feig = 10;

%%
% Move to folder of code
if(~isdeployed)
    cd(fileparts(which(mfilename)));
end

%%
cd ../data/lfw1000
filenames = dir('*.pgm');
nimages = length(filenames);

% Reading in the images
for i = 1:1
    currentfile = filenames(i).name;
    image{i} = double(imread(currentfile));
    % Normalizing each image to neutralize variance in illuminance
    image{i} = (image{i}-mean2(image{i}))/std2(image{i});
end

[nrows , ncolumns] = size(image{1});
%%
% % Matrix of image vectors
% Y = [];
% for i = 1:nimages
%     Y = [Y image{i}(:)];
% end
% 
% cd ..
% save('Y.mat','Y');

%%
if(~isdeployed)
    cd(fileparts(which(mfilename)));
end

% Eigenface data
Y = load('Y.mat');
Y  = cell2mat(struct2cell(Y));

% Subtract the mean face
for i=1:1:size(Y,2)
    Y(:,i) = Y(:,i)-mean(Y,2);
end

% Eigenvectors of covariance matrix
[Uf,Sf,Vf] = svds(Y,feig);

for i = 1:1:feig
    eigenface(i,:,:) = reshape(Uf(:,i),nrows,ncolumns);
end

Uf = [];
for i = 1:1:feig
    image1 = squeeze(eigenface(i,:,:));
    temp = imresize(image1,19/64);
    Uf(:,i) = temp(:);
    Uf(:,i) = Uf(:,i)/norm(Uf(:,i));
end


%%
cd ../data/BoostingData/train/face
trainf = dir('*.pgm');
nfacestrain = length(trainf);

% Reading in the images
for i = 1:1
    currentfile = trainf(i).name;
    image{i} = double(imread(currentfile));
    % Normalizing each image to neutralize variance in illuminance
    image{i} = (image{i}-mean2(image{i}))/std2(image{i});
end

[nfacetrain1 , nfacetrain2] = size(image{1});
%%
% % Matrix of image vectors
% ftrain = [];
% for i = 1:nfacestrain
%     ftrain = [ftrain image{i}(:)];
% end
% 
% cd ../../..
% save('ftrain.mat','ftrain');

%%
%%
if(~isdeployed)
    cd(fileparts(which(mfilename)));
end
% Loading Training Faces
ftrain = load('ftrain.mat');
ftrain = cell2mat(struct2cell(ftrain));

%%
% Face Weights
for i = 1:1:feig
    wftrain (i,:) = Uf(:,i)'*ftrain;
end

err = ftrain - Uf*wftrain;
for i = 1:1:nfacestrain
    errf(i) = 1/(nfacetrain1*nfacetrain2)*norm(err(:,i));
end
wftrain = [wftrain ; errf];
wftrain = wftrain';


%%
cd ../data/BoostingData/train/non-face
trainnf = dir('*.pgm');
nnonfacestrain = length(trainnf);

% Reading in the images
for i = 1:1:1
    currentfile = trainnf(i).name;
    image{i} = double(imread(currentfile));
    % Normalizing each image to neutralize variance in illuminance
    image{i} = (image{i}-mean2(image{i}))/std2(image{i});
end

[nnonfacetrain1 , nnonfacetrain2] = size(image{1});

%%
% % Matrix of image vectors
% nftrain = [];
% for i = 1:nnonfacestrain
%     nftrain = [nftrain image{i}(:)];
% end
% 
% cd ../../..
% save('nftrain.mat','nftrain');

%%
%%
if(~isdeployed)
    cd(fileparts(which(mfilename)));
end
% Loading Training Non-faces
nftrain = load('nftrain.mat');
nftrain = cell2mat(struct2cell(nftrain));
%%
nftrain(isnan(nftrain)) = 0;

%%
% Non-face Weights
for i = 1:1:feig
    wnftrain (i,:) = Uf(:,i)'*nftrain;
end

err = nftrain - Uf*wnftrain;
for i = 1:1:nnonfacestrain
    errnf(i) = 1/(nnonfacetrain1*nnonfacetrain2)*norm(err(:,i));
end

wnftrain = [wnftrain ; errnf];
wnftrain = wnftrain';
feig = feig +1;

%% AdaBoost
% Concatenating to get a common matrix
wtrain = [wftrain ; wnftrain];
imagenumber = (1:size(wtrain,1))';
eignumber = (1:size(wtrain,2))';

% Sorting weights in a columnwise manner
[imvalue,imindex] = sort(wtrain,1);
threshold = zeros(size(imindex));

% imvalue: scores the sorted weights for each eigenvector
% imindex: book-keeping matrix that maps sorted  weights to original weights
% Probability: Updated at the end of each iteration
% alpha: Weight of each iteration
% h: Classification label at the end of each iteration
% sign_final: Direction of classification for each iteration
% totalerror: A metric for measuring training accuracy
iter = 1;
maxiter = 20;
prob = 1/size(imindex,1).*ones(size(imindex,1),maxiter);
alpha = zeros(maxiter,1);
h = ones(size(wtrain,1),maxiter);
y = [ones(nfacestrain,1); -1*ones(nnonfacestrain,1)];
test = zeros(maxiter,3);
totalerror = 0;

%%
while(iter <= maxiter)
    error1 = zeros(size(imindex));
    sign = zeros(size(imindex));
    for i = 1:1:feig
        for j=2:1:size(imindex,1)
            if (j == 2)
                if(imindex(j-1,i)<=nfacestrain)
                    error1(j,i) = prob(imindex(j-1,i),iter);
                end
                for k = 2:1:size(imvalue,1)
                    if(imindex(k,i)>nfacestrain)
                        error1(j,i) = error1(j,i)+prob(imindex(k,i),iter);
                    end
                end
            else
                if(imindex(j-1,i)>nfacestrain)
                    error1(j,i) = error1(j-1,i)-prob(imindex(j-1,i),iter);
                else
                    error1(j,i) = error1(j-1,i)+prob(imindex(j-1,i),iter);
                end
            end
        end
        for j = 1:1:size(imvalue,1)
            if (j<size(imvalue,1))
                threshold(j+1,i) = (imvalue(j,i) + imvalue(j+1,i))/2;
            end
            sign(j,i) = 1;
            if (error1(j,i)>0.5)
                error1(j,i) = 1 - error1(j,i);
                sign(j,i) = -1;
            end
        end
        threshold(1,i) = imvalue(j,i) - realmin;
    end
    error1(1,:) = 0.5;
    errormin = min(error1);
    final_error = min(errormin);
    [r , c] = find(error1 == final_error);
    test(iter,1) = r(1);
    test(iter,2) = c(1);
    signf = sign(r(1),c(1));
    sign_final(iter) = signf;
    alpha(iter) = 0.5*log((1-final_error)/final_error);
    if (signf == 1)
        for j = 1:1:r(1)-1
            h(imindex(j,c(1)),iter) = -1;
        end
    else
        for j = r(1):1:size(wtrain,1)
            h(imindex(j,c(1)),iter) = -1;
        end
    end
    for i=1:1:size(wtrain,1)
        prob(i,iter+1) = prob(i,iter)*exp(-1*alpha(iter)*y(i)*h(i,iter));
    end
    prob(:,iter+1) = (1/sum(prob(:,iter+1))).*prob(:,iter+1);
    iter = iter+1;
end

%%
% H: Final classifier as a sum up individual classifiers
H = h*alpha;
for i = 1:1:length(H)
    if (H(i) <= 0)
        H(i) = -1;
    else
        H(i) = 1;
    end
    if H(i) ~= y(i)
        totalerror = totalerror + 1;
    end
end

a = 100*totalerror/size(wtrain,1);
disp('Training Error');
disp(a);
