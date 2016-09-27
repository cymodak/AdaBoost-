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
cd ../data/lfw1000;

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
if(~isdeployed)
    cd(fileparts(which(mfilename)));
end
ftrain = load('ftrain.mat');
ftrain = cell2mat(struct2cell(ftrain));

%%
% Weights
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
if(~isdeployed)
    cd(fileparts(which(mfilename)));
end
nftrain = load('nftrain.mat');
nftrain = cell2mat(struct2cell(nftrain));
%%
nftrain(isnan(nftrain)) = 0;

%%
% Weights
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

wtrain = [wftrain ; wnftrain];
imagenumber = (1:size(wtrain,1))';
eignumber = (1:size(wtrain,2))';
[imvalue,imindex] = sort(wtrain,1);

threshold = zeros(size(imindex));

iter = 1;
maxiter = 10;
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
    test(iter,3) = threshold(r(1),c(1));
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

%
cd ../data/test_images
N = 19;
M = 19;
scale = 0.25;

I = imread('Beatles.jpg');
image2 = squeeze(mean(double(I),3));

for s = 1:1:length(scale)
    image1 = imresize(image2,scale(s));
    integralA = cumsum(cumsum(image1,1),2);
    integralB = cumsum(cumsum(image1.*image1,1),2);
    
    patchmeansofA = [];
    for i = 1:size(image1,1)-N+1
        for j = 1:size(image1,2)-M+1
            if (i == 1 || j == 1)
                a1 = 0;
                if (i==1)
                    a3 = 0;
                else
                    a3 = integralA(i-1,j+M-1);
                end
                if (j==1)
                    a2 = 0;
                else
                    a2 = integralA(i+N-1,j-1);
                end
            else
                a1 = integralA(i-1,j-1);
                a2 = integralA(i+N-1,j-1);
                a3 = integralA(i-1,j+M-1);
            end
            a4 = integralA(i+N-1,j+M-1);
            patchmeansofA(i,j) = (a4 + a1 - a2 - a3)/(N*M);
        end
    end
    
    % Second moment of Patch
    patchmeansqofA = [];
    for i = 1:size(image1,1)-N+1
        for j = 1:size(image1,2)-M+1
            if (i == 1 || j == 1)
                a1 = 0;
                if (i==1)
                    a3 = 0;
                else
                    a3 = integralB(i-1,j+M-1);
                end
                if (j==1)
                    a2 = 0;
                else
                    a2 = integralB(i+N-1,j-1);
                end
            else
                a1 = integralB(i-1,j-1);
                a2 = integralB(i+N-1,j-1);
                a3 = integralB(i-1,j+M-1);
            end
            a4 = integralB(i+N-1,j+M-1);
            patchmeansqofA(i,j) = (a4 + a1 - a2 - a3)/(N*M);
        end
    end
    
    % Variance of Patch
    patchvarsofA = patchmeansqofA - patchmeansofA.^2;
    patchvarsofA = sqrt(patchvarsofA);
    weights = [];
    patchscore = zeros(size(image1));
    
    for i = 1:size(image1,1)-N+1
        for j = 1:size(image1,2)-M+1
            patch = image1(i:i+N-1,j:j+N-1);
            patch = (patch(:)-patchmeansofA(i,j))/(patchvarsofA(i,j)+realmin);
            weights = Uf'*patch;
            errpatch = patch - Uf*weights;
            err = 1/(N*M)*norm(errpatch);
            weights = [weights ; err];
            for k = 1:1:maxiter
                output = 1;
                if (sign_final(k)  == -1)
                    if (threshold(test(k,1),test(k,2)) < weights(test(k,2)))
                        output = -1;
                    end
                else
                    if (threshold(test(k,1),test(k,2)) > weights(test(k,2)))
                        output = -1;
                    end
                end
                wt(k) = alpha(k)*output;
            end
            patchscore(i,j) = sum(wt);            
        end
    end    
        bbox1 = floor(1/scale(s)*[22, 18, 19, 19]);
        bbox2 = floor(1/scale(s)*[90, 30, 19, 19]);
        bbox3 = floor(1/scale(s)*[67, 53, 19, 19]);
        I1 = rgb2gray(I);
        I1 = insertShape(I1,'FilledRectangle',bbox1,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox2,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox3,'Color','green');
        if(~isdeployed)
            cd(fileparts(which(mfilename)));
        end
        imwrite(I1,'result/Beatles@0.25.jpg');
end

%%
cd ../data/test_images
N = 19;
M = 19;
scale = 0.2;
I = imread('Big_3.jpg');
image2 = double(I);

for s = 1:1:length(scale)
    image1 = imresize(image2,scale(s));
    integralA = cumsum(cumsum(image1,1),2);
    integralB = cumsum(cumsum(image1.*image1,1),2);
    
    patchmeansofA = [];
    for i = 1:size(image1,1)-N+1
        for j = 1:size(image1,2)-M+1
            if (i == 1 || j == 1)
                a1 = 0;
                if (i==1)
                    a3 = 0;
                else
                    a3 = integralA(i-1,j+M-1);
                end
                if (j==1)
                    a2 = 0;
                else
                    a2 = integralA(i+N-1,j-1);
                end
            else
                a1 = integralA(i-1,j-1);
                a2 = integralA(i+N-1,j-1);
                a3 = integralA(i-1,j+M-1);
            end
            a4 = integralA(i+N-1,j+M-1);
            patchmeansofA(i,j) = (a4 + a1 - a2 - a3)/(N*M);
        end
    end
    
    % Second moment of Patch
    patchmeansqofA = [];
    for i = 1:size(image1,1)-N+1
        for j = 1:size(image1,2)-M+1
            if (i == 1 || j == 1)
                a1 = 0;
                if (i==1)
                    a3 = 0;
                else
                    a3 = integralB(i-1,j+M-1);
                end
                if (j==1)
                    a2 = 0;
                else
                    a2 = integralB(i+N-1,j-1);
                end
            else
                a1 = integralB(i-1,j-1);
                a2 = integralB(i+N-1,j-1);
                a3 = integralB(i-1,j+M-1);
            end
            a4 = integralB(i+N-1,j+M-1);
            patchmeansqofA(i,j) = (a4 + a1 - a2 - a3)/(N*M);
        end
    end
    
    % Variance of Patch
    patchvarsofA = patchmeansqofA - patchmeansofA.^2;
    patchvarsofA = sqrt(patchvarsofA);
    weights = [];
    patchscore = zeros(size(image1));
    
    for i = 1:size(image1,1)-N+1
        for j = 1:size(image1,2)-M+1
            patch = image1(i:i+N-1,j:j+N-1);
            patch = (patch(:)-patchmeansofA(i,j))/(patchvarsofA(i,j)+realmin);
            weights = Uf'*patch;
            errpatch = patch - Uf*weights;
            err = 1/(N*M)*norm(errpatch);
            weights = [weights ; err];
            for k = 1:1:maxiter
                output = 1;
                if (sign_final(k)  == -1)
                    if (threshold(test(k,1),test(k,2)) < weights(test(k,2)))
                        output = -1;
                    end
                else
                    if (threshold(test(k,1),test(k,2)) > weights(test(k,2)))
                        output = -1;
                    end
                end
                wt(k) = alpha(k)*output;
            end
            patchscore(i,j) = sum(wt);
        end
    end
    
    if (s==1)
     bbox1 = floor(1/scale(s)*[15, 50, 19, 19]);
     bbox2 = floor(1/scale(s)*[60, 30, 19, 19]);
     bbox3 = floor(1/scale(s)*[10, 25, 19, 19]);
     I1 = I;
     I1 = insertShape(I1,'FilledRectangle',bbox1,'Color','green');
     I1 = insertShape(I1,'FilledRectangle',bbox2,'Color','green');
     I1 = insertShape(I1,'FilledRectangle',bbox3,'Color','green');
     if(~isdeployed)
         cd(fileparts(which(mfilename)));
     end
     imwrite(I1,'result/Big_3@0.2.jpg');
    end   
        
    
end

%%
cd ../data/test_images
N = 19;
M = 19;
I = imread('G8.jpg');
image2 = squeeze(mean(double(I),3));
scale = 0.2;
for s = 1:1:length(scale)
    image1 = imresize(image2,scale(s));
    integralA = cumsum(cumsum(image1,1),2);
    integralB = cumsum(cumsum(image1.*image1,1),2);
    
    patchmeansofA = [];
    for i = 1:size(image1,1)-N+1
        for j = 1:size(image1,2)-M+1
            if (i == 1 || j == 1)
                a1 = 0;
                if (i==1)
                    a3 = 0;
                else
                    a3 = integralA(i-1,j+M-1);
                end
                if (j==1)
                    a2 = 0;
                else
                    a2 = integralA(i+N-1,j-1);
                end
            else
                a1 = integralA(i-1,j-1);
                a2 = integralA(i+N-1,j-1);
                a3 = integralA(i-1,j+M-1);
            end
            a4 = integralA(i+N-1,j+M-1);
            patchmeansofA(i,j) = (a4 + a1 - a2 - a3)/(N*M);
        end
    end
    
    % Second moment of Patch
    patchmeansqofA = [];
    for i = 1:size(image1,1)-N+1
        for j = 1:size(image1,2)-M+1
            if (i == 1 || j == 1)
                a1 = 0;
                if (i==1)
                    a3 = 0;
                else
                    a3 = integralB(i-1,j+M-1);
                end
                if (j==1)
                    a2 = 0;
                else
                    a2 = integralB(i+N-1,j-1);
                end
            else
                a1 = integralB(i-1,j-1);
                a2 = integralB(i+N-1,j-1);
                a3 = integralB(i-1,j+M-1);
            end
            a4 = integralB(i+N-1,j+M-1);
            patchmeansqofA(i,j) = (a4 + a1 - a2 - a3)/(N*M);
        end
    end
    
    % Variance of Patch
    patchvarsofA = patchmeansqofA - patchmeansofA.^2;
    patchvarsofA = sqrt(patchvarsofA);
    weights = [];
    patchscore = zeros(size(image1));
    
    for i = 1:size(image1,1)-N+1
        for j = 1:size(image1,2)-M+1
            patch = image1(i:i+N-1,j:j+N-1);
            patch = (patch(:)-patchmeansofA(i,j))/(patchvarsofA(i,j)+realmin);
            weights = Uf'*patch;
            errpatch = patch - Uf*weights;
            err = 1/(N*M)*norm(errpatch);
            weights = [weights ; err];
            for k = 1:1:maxiter
                output = 1;
                if (sign_final(k)  == -1)
                    if (threshold(test(k,1),test(k,2)) < weights(test(k,2)))
                        output = -1;
                    end
                else
                    if (threshold(test(k,1),test(k,2)) > weights(test(k,2)))
                        output = -1;
                    end
                end
                wt(k) = alpha(k)*output;
            end
            patchscore(i,j) = sum(wt);
        end
    end
    bbox1 = floor(1/scale(s)*[120, 65, 19, 19]);
    bbox2 = floor(1/scale(s)*[85, 20, 19, 19]);
    bbox3 = floor(1/scale(s)*[50, 60, 19, 19]);
    bbox4 = floor(1/scale(s)*[70, 20, 19, 19]);
    bbox5 = floor(1/scale(s)*[45, 20, 19, 19]);
    
    I1 = rgb2gray(I);
    I1 = insertShape(I1,'FilledRectangle',bbox1,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox2,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox3,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox4,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox5,'Color','green');
    if(~isdeployed)
        cd(fileparts(which(mfilename)));
    end
    imwrite(I1,'result/G8@0.2.jpg');
end

%%
cd ../data/test_images
N = 19;
M = 19;
scale = 0.2;
I = imread('Solvay.jpg');
image2 = squeeze(mean(double(I),3));

for s = 1:1:length(scale)
    image1 = imresize(image2,scale(s));
    integralA = cumsum(cumsum(image1,1),2);
    integralB = cumsum(cumsum(image1.*image1,1),2);
    
    patchmeansofA = [];
    for i = 1:size(image1,1)-N+1
        for j = 1:size(image1,2)-M+1
            if (i == 1 || j == 1)
                a1 = 0;
                if (i==1)
                    a3 = 0;
                else
                    a3 = integralA(i-1,j+M-1);
                end
                if (j==1)
                    a2 = 0;
                else
                    a2 = integralA(i+N-1,j-1);
                end
            else
                a1 = integralA(i-1,j-1);
                a2 = integralA(i+N-1,j-1);
                a3 = integralA(i-1,j+M-1);
            end
            a4 = integralA(i+N-1,j+M-1);
            patchmeansofA(i,j) = (a4 + a1 - a2 - a3)/(N*M);
        end
    end
    
    % Second moment of Patch
    patchmeansqofA = [];
    for i = 1:size(image1,1)-N+1
        for j = 1:size(image1,2)-M+1
            if (i == 1 || j == 1)
                a1 = 0;
                if (i==1)
                    a3 = 0;
                else
                    a3 = integralB(i-1,j+M-1);
                end
                if (j==1)
                    a2 = 0;
                else
                    a2 = integralB(i+N-1,j-1);
                end
            else
                a1 = integralB(i-1,j-1);
                a2 = integralB(i+N-1,j-1);
                a3 = integralB(i-1,j+M-1);
            end
            a4 = integralB(i+N-1,j+M-1);
            patchmeansqofA(i,j) = (a4 + a1 - a2 - a3)/(N*M);
        end
    end
    
    % Variance of Patch
    patchvarsofA = patchmeansqofA - patchmeansofA.^2;
    patchvarsofA = sqrt(patchvarsofA);
    weights = [];
    patchscore = zeros(size(image1));
    
    for i = 1:size(image1,1)-N+1
        for j = 1:size(image1,2)-M+1
            patch = image1(i:i+N-1,j:j+N-1);
            patch = (patch(:)-patchmeansofA(i,j))/(patchvarsofA(i,j)+realmin);
            weights = Uf'*patch;
            errpatch = patch - Uf*weights;
            err = 1/(N*M)*norm(errpatch);
            weights = [weights ; err];
            for k = 1:1:maxiter
                output = 1;
                if (sign_final(k)  == -1)
                    if (threshold(test(k,1),test(k,2)) < weights(test(k,2)))
                        output = -1;
                    end
                else
                    if (threshold(test(k,1),test(k,2)) > weights(test(k,2)))
                        output = -1;
                    end
                end
                wt(k) = alpha(k)*output;
            end
            patchscore(i,j) = sum(wt);
        end
    end
    
    bbox1 = floor(1/scale(s)*[40, 65, 19, 19]);
    bbox2 = floor(1/scale(s)*[75, 65, 19, 19]);
    bbox3 = floor(1/scale(s)*[110, 60, 19, 19]);
    bbox4 = floor(1/scale(s)*[140, 70, 19, 19]);
    bbox5 = floor(1/scale(s)*[180, 45, 19, 19]);
    bbox6 = floor(1/scale(s)*[200, 45, 19, 19]);
    bbox7 = floor(1/scale(s)*[130, 40, 19, 19]);
    bbox8 = floor(1/scale(s)*[75, 40, 19, 19]);
    bbox9 = floor(1/scale(s)*[50, 40, 19, 19]);
    bbox10 = floor(1/scale(s)*[30, 30, 19, 19]);
    bbox11 = floor(1/scale(s)*[25, 15, 19, 19]);
    
    I1 = rgb2gray(I);
    I1 = insertShape(I1,'FilledRectangle',bbox1,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox2,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox3,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox4,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox5,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox6,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox7,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox8,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox9,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox10,'Color','green');
    I1 = insertShape(I1,'FilledRectangle',bbox11,'Color','green');
    if(~isdeployed)
        cd(fileparts(which(mfilename)));
    end
    imwrite(I1,'result/Solvay@0.2.jpg');
end
