function [scoreMap] = scan_face(imgIn, eigenFace, dataMean, histEqFlag, normFlag)

[P , Q] = size(imgIn);
scoreMap = -1*ones(P,Q);
N = 64;
M = 64;
% Write your code to get scoreMap with "for loop"
for i = 1:(P-N)
    for j = 1:(Q-M)
        patch = imgIn(i:i+N-1,j:j+M-1);
        if(histEqFlag)
            patch = double(histeq(uint8(imgIn(i:i+N-1,j:j+M-1))));
        end
        if(normFlag)
            patch = (patch-mean2(patch))/std2(patch);
        end
        scoreMap(i,j) = eigenFace(:)'*patch(:)/norm(patch(:));
    end  
end

end

