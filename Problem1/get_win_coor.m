function [coorSet] = get_win_coor(scoreMapSetOut, scaleSet, heightImg, widthImg)
coorSet = [];
scaleNum = length(scaleSet);

% Write code to scan through each scale of score maps. At each location of
% the score map where the score is larger than -1, return the corresponding
% coordinates of top-left and bottom-right corners of the scanning window
% in the non-scaled test image. That is: you need to carefully compute:
% given a certain location in the scoreMap, if there is a 64 x 64
% window centered at this cloation, what are its window corner coordinates
% in the original (non-scaled) image. This can be a little tricky as:
% 1. Score maps have shrinked sizes compared with test images at each scale
% 2. coorSet contains corner coordinates to be ploted in the non-scaled
% image but scoreMaps are scaled ones. so you need to also take scaling
% into consideration.
for i = 1:1:scaleNum
    [P , Q] = size(scoreMapSetOut{i});
    [r , c] = find(scoreMapSetOut{i} > -1);
    r1 = 64*ones(size(r));
    c1 = 64*ones(size(c));
    coorl = floor(1/scaleSet(i)*[r-31 c-31]);
    coorr = floor(1/scaleSet(i)*[r1 c1]);
    coorSet{i} = [coorl coorr];
end

end