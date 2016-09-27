function [coorSet, scoreMapSet] = face_detect(imgIn, eigenface, scoreThre, histEqFlag, normFlag, scaleSet, neighNMS)
%% Input arguments:
% imgIn: H x W x chn matrix of input test image
% eigenFace: D x 1 eigen face vector
% scoreThre: Threshold between [-1 1] controlling the number of fired
% scanning windows
% histEqFlag: Logical flag (true/false) indicating whether histeq will be
% used for each scanned patch in the test image
% normFlag: Logical flag (true/false) indicating whether each scanned patch
% in the test image will be normalized
% scaleSet: row vector contains scales from small ones to large ones. For
% example, [0.5 0.75 1 1.5]
% neighNMS: neighborhood size within which scores are to be suppressed. In
% this HW, 64 is a good value

%% Output arguments
% imgOut: H x W x chn matrix of test image with plotted fired face windows
% scoreMapSet: cell array containing score maps at each scale

scaleNum = length(scaleSet);
scoreMapSet = cell(scaleNum,1);
dataMean = 0;
%% Preprocess test image
[heightImg, widthImg, chn] = size(imgIn);
% Write code to change imgIn into double format.
% I = rgb2gray(ImgIn);
imgIn = double(imgIn);
% If the imgIn contains 3 channels, transform it to grayscale.
imgIn = squeeze(mean(imgIn,3));
% Normalize pixel values between [0 1];
temp = imgIn./255;

%% Get scoreMapSet and scoreMapSetThre
scoreMapSetThre = cell(scaleNum,1); % The set of thresholded score maps
% Use "for loop" to iterate through different scales.
% At each scale, use the function "scoreMap = scan_face(imgScaled,
% eigenFace, histEqFlag, normFlag);" to return a score map. Assign every
% "scoreMap" to "scoreMapSet"
% Use "scoreThre" to threshold "scoreMap" and assign every thresholded map
% to "scoreMapSetThre". Note that thresholded locations should be assigned
% with a score value of -1. (-1 is the minimum possible value of cosine distance)
for i=1:1:scaleNum
    imgIn = imresize(temp,scaleSet(i));
    [scoreMap] = scan_face(imgIn, eigenface, dataMean, histEqFlag, normFlag);
    scoreMapSetIn{i} = scoreMap;
end

%% Nonlocal maximum suppression
scoreMapSetOut = nms(scoreMapSetIn, neighNMS, scaleSet, scoreThre);

%% Get fired window coordinates
coorSet = get_win_coor(scoreMapSetOut, scaleSet, heightImg, widthImg);
% "coorSet" is an M x 4 matrix where each row contains the x and y
% coordinates of top left and bottom right window corners. The format is
% [xtl ytl xbr ybr].

end