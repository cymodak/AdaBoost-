close all
clear all
clc

% Move to folder of code
if(~isdeployed)
    cd(fileparts(which(mfilename)));
end

filenames = dir('../data/lfw1000/*.pgm');
nimages = size(filenames,1);
data = [];
for i = 1:nimages
    face = double(imread(['../data/lfw1000/' filenames(i,1).name]))./255;
    data(:,i) = face(:);
end

[nrows , ncolumns] = size(face);

for i=1:1:nimages
    data(:,i) = (data(:,i)-mean(data(:,i)))/norm(data(:,i));
end
Y = data;

% % Subtract the mean face
% for i=1:1:size(Y,2)
%     Y(:,i) = Y(:,i)-mean(Y,2);
% end

% Eigenvectors of covariance matrix
[U,S,V] = svds(Y,1);

% First eigenface
eigenface = -1*reshape(U(:,1),nrows,ncolumns);

%%
% Reading group photos
scale = [0.5 0.75 1 1.5];
scalenum = length(scale);

% %
groupnames = '../data/test_images/Beatles.jpg';
I = imread(groupnames);
image1 = double(I);
for iter = 1:1:scalenum
    % RGB to grey-scale
    image1 = squeeze(mean(I,3));
    temp = image1;
    % Image Scaling
    [P,Q] = size(image1);
    [N,M] = size(eigenface);
    image1 = imresize(image1,scale(iter));
    [P,Q] = size(image1);
    
    % Functions used to calculate Patch Moments
    integralA = cumsum(cumsum(image1,1),2);
    integralB = cumsum(cumsum(image1.*image1,1),2);
    
    % Mean of Patch
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
    
    % Convolution of Patch and Eigenface
    tmpim = conv2(image1, fliplr(flipud(eigenface)));
    convolvedimage = tmpim(N:end, M:end);
    
    % Normalizing with Patch Mean and Variance
    sumE = sum(eigenface(:));
    patchscore = (convolvedimage(1:size(patchmeansofA,1),1:size(patchmeansofA,2)) - sumE*patchmeansofA)./patchvarsofA;
    
    s1 = P - size(patchscore,1);
    s2 = Q - size(patchscore,2);
    
    patchscore = [patchscore zeros(size(patchscore,1),s2)];
    patchscore = [patchscore ; zeros(s1,size(patchscore,2))];
    patchscore = patchscore./max(max(patchscore));
    
    if (iter == 1)
        bbox1 = floor(1/scale(iter)*[160, 40, 64 , 64]);
        bbox2 = floor(1/scale(iter)*[250, 20, 64 , 64]);
        bbox3 = floor(1/scale(iter)*[80, 80, 64 , 64]);
        bbox4 = floor(1/scale(iter)*[1, 100, 64 , 64]);
        I1 = rgb2gray(I);
        I1 = insertShape(I1,'FilledRectangle',bbox1,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox2,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox3,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox4,'Color','green');
        imwrite(I1,'result/Beatles_050.jpg');
    end
    if (iter == 2)
        bbox5 = floor(1/scale(iter)*[400, 100, 64, 64]);
        bbox6 = floor(1/scale(iter)*[290, 100, 64, 64]);
        bbox7 = floor(1/scale(iter)*[180, 130, 64, 64]);
        bbox8 = floor(1/scale(iter)*[20, 180, 64, 64]);
        I1 = rgb2gray(I);
        I1 = insertShape(I1,'FilledRectangle',bbox5,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox6,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox7,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox8,'Color','green');
        imwrite(I1,'result/Beatles_075.jpg');
    end   
    if (iter == 3)
        bbox9 = floor(1/scale(iter)*[380, 160, 64, 64]);
        bbox10 = floor(1/scale(iter)*[560, 110, 64, 64]);
        I1 = rgb2gray(I);
        I1 = insertShape(I1,'FilledRectangle',bbox9,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox10,'Color','green');
        imwrite(I1,'result/Beatles_100.jpg');
    end   
    if (iter == 4)
        bbox11 = floor(1/scale(iter)*[580, 260, 64, 64]);
        I1 = rgb2gray(I);
        I1 = insertShape(I1,'FilledRectangle',bbox11,'Color','green');
        imwrite(I1,'result/Beatles_150.jpg');
    end   
end

%%
groupnames = '../data/test_images/Big_3.jpg';
I2 = imread(groupnames);
I = double(imread(groupnames));
image1 = I;
for iter = 1:1:scalenum
    % RGB to grey-scale
    image1 = I;
    temp = image1;
    % Image Scaling
    [P,Q] = size(image1);
    [N,M] = size(eigenface);
    image1 = imresize(image1,scale(iter));
    [P,Q] = size(image1);
    
    % Functions used to calculate Patch Moments
    integralA = cumsum(cumsum(image1,1),2);
    integralB = cumsum(cumsum(image1.*image1,1),2);
    
    % Mean of Patch
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
    
    % Convolution of Patch and Eigenface
    tmpim = conv2(image1, fliplr(flipud(eigenface)));
    convolvedimage = tmpim(N:end, M:end);
    
    % Normalizing with Patch Mean and Variance
    sumE = sum(eigenface(:));
    patchscore = (convolvedimage(1:size(patchmeansofA,1),1:size(patchmeansofA,2)) - sumE*patchmeansofA)./patchvarsofA;
    
    s1 = P - size(patchscore,1);
    s2 = Q - size(patchscore,2);
    
    patchscore = [patchscore zeros(size(patchscore,1),s2)];
    patchscore = [patchscore ; zeros(s1,size(patchscore,2))];
    patchscore = patchscore./max(max(patchscore));
    if (iter == 1)
        bbox1 = floor(1/scale(iter)*[190, 50, 64 , 64]);
        bbox2 = floor(1/scale(iter)*[10, 90, 64 , 64]);
        I1 = I2;
        I1 = insertShape(I1,'FilledRectangle',bbox1,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox2,'Color','green');
        imwrite(I1,'result/Big_3_050.jpg');
    end
    if (iter == 2)
        bbox3 = floor(1/scale(iter)*[300, 100, 64, 64]);
        bbox4 = floor(1/scale(iter)*[200, 60, 64, 64]);
        bbox5 = floor(1/scale(iter)*[165, 150, 64, 64]);
        I1 = I2;
        I1 = insertShape(I1,'FilledRectangle',bbox3,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox4,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox5,'Color','green');
        imwrite(I1,'result/Big_3_075.jpg');
    end   
    if (iter == 3)
        bbox6 = floor(1/scale(iter)*[410, 160, 64, 64]);
        I1 = I2;
        I1 = insertShape(I1,'FilledRectangle',bbox6,'Color','green');
        imwrite(I1,'result/Big_3_100.jpg');
    end   
    if (iter == 4)
        bbox7 = floor(1/scale(iter)*[20, 150, 64, 64]);
        I1 = I2;
        I1 = insertShape(I1,'FilledRectangle',bbox7,'Color','green');
        imwrite(I1,'result/Big_3_150.jpg');
    end   
end

%%
groupnames = '../data/test_images/G8.jpg';
I = imread(groupnames);
image1 = double(I);
for iter = 1:1:scalenum
    % RGB to grey-scale
    image1 = squeeze(mean(I,3));
    temp = image1;
    % Image Scaling
    [P,Q] = size(image1);
    [N,M] = size(eigenface);
    image1 = imresize(image1,scale(iter));
    [P,Q] = size(image1);
    
    % Functions used to calculate Patch Moments
    integralA = cumsum(cumsum(image1,1),2);
    integralB = cumsum(cumsum(image1.*image1,1),2);
    
    % Mean of Patch
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
    
    % Convolution of Patch and Eigenface
    tmpim = conv2(image1, fliplr(flipud(eigenface)));
    convolvedimage = tmpim(N:end, M:end);
    
    % Normalizing with Patch Mean and Variance
    sumE = sum(eigenface(:));
    patchscore = (convolvedimage(1:size(patchmeansofA,1),1:size(patchmeansofA,2)) - sumE*patchmeansofA)./patchvarsofA;
    
    s1 = P - size(patchscore,1);
    s2 = Q - size(patchscore,2);
    
    patchscore = [patchscore zeros(size(patchscore,1),s2)];
    patchscore = [patchscore ; zeros(s1,size(patchscore,2))];
    patchscore = patchscore./max(max(patchscore));
    if (iter == 1)
        bbox1 = floor(1/scale(iter)*[200, 1, 64 , 64]);
        bbox2 = floor(1/scale(iter)*[260, 1, 64 , 64]);  
        bbox3 = floor(1/scale(iter)*[55, 100, 64 , 64]);
        bbox4 = floor(1/scale(iter)*[130, 90, 64 , 64]);
        I1 = rgb2gray(I);
        I1 = insertShape(I1,'FilledRectangle',bbox1,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox2,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox3,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox4,'Color','green');
        imwrite(I1,'result/G8_50.jpg');
    end
    if (iter == 2)
        bbox5 = floor(1/scale(iter)*[200, 150, 64, 64]);
        I1 = rgb2gray(I);
        I1 = insertShape(I1,'FilledRectangle',bbox5,'Color','green');
        imwrite(I1,'result/G8_75.jpg');
    end   
    if (iter == 3)
        bbox6 = floor(1/scale(iter)*[120, 450, 64, 64]);
        bbox7 = floor(1/scale(iter)*[180, 220, 64, 64]);
        bbox8 = floor(1/scale(iter)*[300, 250, 64, 64]);       
        I1 = rgb2gray(I);
        I1 = insertShape(I1,'FilledRectangle',bbox6,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox7,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox8,'Color','green');
        imwrite(I1,'result/G8_100.jpg');
    end   
    if (iter == 4)
        bbox9 = floor(1/scale(iter)*[680, 220, 64, 64]);
        bbox10 = floor(1/scale(iter)*[850, 190, 64, 64]);
        I1 = rgb2gray(I);
        I1 = insertShape(I1,'FilledRectangle',bbox9,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox10,'Color','green');
        imwrite(I1,'result/G8_150.jpg');
    end   
end

%%
%%
groupnames = '../data/test_images/Solvay.jpg';
I = imread(groupnames);
image1 = double(I);
for iter = 1:1:scalenum
    % RGB to grey-scale
    image1 = squeeze(mean(I,3));
    temp = image1;
    % Image Scaling
    [P,Q] = size(image1);
    [N,M] = size(eigenface);
    image1 = imresize(image1,scale(iter));
    [P,Q] = size(image1);
    
    % Functions used to calculate Patch Moments
    integralA = cumsum(cumsum(image1,1),2);
    integralB = cumsum(cumsum(image1.*image1,1),2);
    
    % Mean of Patch
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
    
    % Convolution of Patch and Eigenface
    tmpim = conv2(image1, fliplr(flipud(eigenface)));
    convolvedimage = tmpim(N:end, M:end);
    
    % Normalizing with Patch Mean and Variance
    sumE = sum(eigenface(:));
    patchscore = (convolvedimage(1:size(patchmeansofA,1),1:size(patchmeansofA,2)) - sumE*patchmeansofA)./patchvarsofA;
    
    s1 = P - size(patchscore,1);
    s2 = Q - size(patchscore,2);
    
    patchscore = [patchscore zeros(size(patchscore,1),s2)];
    patchscore = [patchscore ; zeros(s1,size(patchscore,2))];
    patchscore = patchscore./max(max(patchscore));
    if (iter == 1)
        bbox1 = floor(1/scale(iter)*[140, 50, 64 , 64]);
        I1 = rgb2gray(I);
        I1 = insertShape(I1,'FilledRectangle',bbox1,'Color','green');
        imwrite(I1,'result/Solvay_50.jpg');
    end
    if (iter == 2)
        bbox2 = floor(1/scale(iter)*[70, 60, 64, 64]);
        bbox3 = floor(1/scale(iter)*[130, 50, 64, 64]);
        bbox4 = floor(1/scale(iter)*[370, 60, 64, 64]);
        I1 = rgb2gray(I);
        I1 = insertShape(I1,'FilledRectangle',bbox2,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox3,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox4,'Color','green');
        imwrite(I1,'result/Solvay_75.jpg');
    end   
    if (iter == 3)
        bbox5 = floor(1/scale(iter)*[380, 220, 64, 64]);
        bbox6 = floor(1/scale(iter)*[670, 240, 64, 64]);
        bbox7 = floor(1/scale(iter)*[550, 240, 64, 64]);
        bbox8 = floor(1/scale(iter)*[600, 190, 64, 64]);
        bbox9 = floor(1/scale(iter)*[820, 220, 64, 64]);
        bbox10 = floor(1/scale(iter)*[880, 120, 64, 64]);
        I1 = rgb2gray(I);
        I1 = insertShape(I1,'FilledRectangle',bbox5,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox6,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox7,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox8,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox9,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox10,'Color','green');
        imwrite(I1,'result/Solvay_100.jpg');
    end   
    if (iter == 4)
        bbox11 = floor(1/scale(iter)*[1220, 210, 64, 64]);
        bbox12 = floor(1/scale(iter)*[1070, 200, 64, 64]);
        bbox13 = floor(1/scale(iter)*[770, 220, 64, 64]);
        bbox14 = floor(1/scale(iter)*[620, 210, 64, 64]);
        I1 = rgb2gray(I);
        I1 = insertShape(I1,'FilledRectangle',bbox11,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox12,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox13,'Color','green');
        I1 = insertShape(I1,'FilledRectangle',bbox14,'Color','green');
        imwrite(I1,'result/Solvay_150.jpg');
    end   
end
