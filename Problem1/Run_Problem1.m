clc;clear;close all
addpath('../external/export_fig/')

% Move to folder of code
if(~isdeployed)
    cd(fileparts(which(mfilename)));
end

%% Parameters
histEqFlag = true;
normFlag = true;
scaleSet = [0.5 0.75 1 1.5];
scaleNum = length(scaleSet);
% Also set neighNMS size: neighNMS = ?

%% Load data
[data , nimages ,nrows , ncolumns] = load_data();

%% Compute Eigen Face
eigenface = get_eigen_face(data, histEqFlag, normFlag , nrows , ncolumns);

%% Beatles
% Set your scoreThre for this image: scoreThre = ?
scoreThre = 0.5;
neighNMS = 64;

% load Beatles.jpg
filename = '../data/test_images/Beatles.jpg';
imgIn = imread(filename);

% perform face detection with the function "face_detect
[coorSet, scoreMapSet] = face_detect(imgIn, eigenface, scoreThre, histEqFlag, normFlag, scaleSet, neighNMS);

% Write the visualized results of scoreMap at each scale to "result"
% folder, using "imagesc" and "export_fig()". For example, the map at scale
% 0.5 should be saved as "Beatles_scoreMap_0.5.jpg"
% Write the test image with detected faces as "Beatles_result.jpg"
%% Get image with printed windows
for i = 1:1:scaleNum
    coor = coorSet{i};
    I = rgb2gray(imgIn);
    for j = 1:1:size(coor,1)
        bboxA = coor(j,:);
        I = insertShape(I,'FilledRectangle',bboxA,'Color','green');
    end
    if (i == 1)
        filename = 'result/Beatles_0.5.jpg';
        imwrite(I,filename);
    end
    if (i == 2)
        filename = 'result/Beatles_0.75.jpg';
        imwrite(I,filename);
    end
    if (i == 3)
        filename = 'result/Beatles_1.jpg';
        imwrite(I,filename);
    end
    if (i == 4)
        filename = 'result/Beatles_1.5.jpg';
        imwrite(I,filename);
    end    
end

%% Big_3
% Set your scoreThre for this image: scoreThre = ?
scoreThre = 0.75;
neighNMS = 64;

% load Big_3.jpg
filename = '../data/test_images/Big_3.jpg';
imgIn = imread(filename);

% perform face detection with the function "face_detect"
[coorSet, scoreMapSet] = face_detect(imgIn, eigenface, scoreThre, histEqFlag, normFlag, scaleSet, neighNMS);
% Write the visualized results of scoreMap at each scale to "result"
% folder, using "imagesc" and "export_fig()". For example, the map at scale
% 0.5 should be saved as "Big_3_scoreMap_0.5.jpg"
% Write the test image with detected faces as "Big_3_result.jpg"
%% Get image with printed windows
for i = 1:1:scaleNum
    coor = coorSet{i};
    I = imgIn;
    for j = 1:1:size(coor,1)
        bboxA = coor(j,:);
        I = insertShape(I,'FilledRectangle',bboxA,'Color','green');
    end
    if (i == 1)
        filename = 'result/Big3_0.5.jpg';
        imwrite(I,filename);
    end
    if (i == 2)
        filename = 'result/Big3_0.75.jpg';
        imwrite(I,filename);
    end
    if (i == 3)
        filename = 'result/Big3_1.jpg';
        imwrite(I,filename);
    end
    if (i == 4)
        filename = 'result/Big3_1.5.jpg';
        imwrite(I,filename);
    end    
end

%% G8
% Set your scoreThre for this image: scoreThre = ?
scoreThre = 0.75;
neighNMS = 64;

% load G8.jpg
filename = '../data/test_images/G8.jpg';
imgIn = imread(filename);
% perform face detection with the function "face_detect"
[coorSet, scoreMapSet] = face_detect(imgIn, eigenface, scoreThre, histEqFlag, normFlag, scaleSet, neighNMS);
% Write the visualized results of scoreMap at each scale to "result"
% folder, using "imagesc" and "export_fig()". For example, the map at scale
% 0.5 should be saved as "G8_scoreMap_0.5.jpg"
% Write the test image with detected faces as "G8_result.jpg"
%% Get image with printed windows
for i = 1:1:scaleNum
    coor = coorSet{i};
    I = rgb2gray(imgIn);
    for j = 1:1:size(coor,1)
        bboxA = coor(j,:);
        I = insertShape(I,'FilledRectangle',bboxA,'Color','green');
    end
    if (i == 1)
        filename = 'result/G8_0.5.jpg';
        imwrite(I,filename);
    end
    if (i == 2)
        filename = 'result/G8_0.75.jpg';
        imwrite(I,filename);
    end
    if (i == 3)
        filename = 'result/G8_1.jpg';
        imwrite(I,filename);
    end
    if (i == 4)
        filename = 'result/G8_1.5.jpg';
        imwrite(I,filename);
    end    
end

%% Solvay
% Set your scoreThre for this image: scoreThre = ?
scoreThre = 0.8;
neighNMS = 64;

% load Solvay.jpg
filename = '../data/test_images/Solvay.jpg';
imgIn = imread(filename);
% perform face detection with the function "face_detect"
[coorSet, scoreMapSet] = face_detect(imgIn, eigenface, scoreThre, histEqFlag, normFlag, scaleSet, neighNMS);
% Write the visualized results of scoreMap at each scale to "result"
% folder, using "imagesc" and "export_fig()". For example, the map at scale
% 0.5 should be saved as "Solvay_scoreMap_0.5.jpg"
% Write the test image with detected faces as "Solvay_result.jpg"
for i = 1:1:scaleNum
    coor = coorSet{i};
    I = rgb2gray(imgIn);
    for j = 1:1:size(coor,1)
        bboxA = coor(j,:);
        I = insertShape(I,'FilledRectangle',bboxA,'Color','green');
    end
    if (i == 1)
        filename = 'result/Solvay_0.5.jpg';
        imwrite(I,filename);
    end
    if (i == 2)
        filename = 'result/Solvay_0.75.jpg';
        imwrite(I,filename);
    end
    if (i == 3)
        filename = 'result/Solvay_1.jpg';
        imwrite(I,filename);
    end
    if (i == 4)
        filename = 'result/Solvay_1.5.jpg';
        imwrite(I,filename);
    end    
end