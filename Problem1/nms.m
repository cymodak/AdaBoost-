function [scoreMapSetOut] = nms(scoreMapSetIn, neighSize, scaleSet, scoreThre)
scaleNum = length(scaleSet);
threshold = scoreThre;
% Write your code to conduct non-local maximum suppression
for i = 1:1:scaleNum
    temp = scoreMapSetIn{i};
    temp = temp./max(max(temp));
    [P , Q] = size(temp);
    for p = 1:1:P
        for q = 1:1:Q
            if(temp(p,q) < threshold)
                temp(p,q) = -1;
            end
        end
    end
    for m = neighSize+1:1:P-neighSize
        for n = neighSize+1:1:Q-neighSize
            patch = temp(m-neighSize:m+neighSize,n-neighSize:n+neighSize);
            for p = 1:1:2*neighSize+1
                for q = 1:1:2*neighSize+1
                    if(patch(p,q) > temp(m,n))
                        temp(m,n) = -1;
                        break;
                    end
                end
                if (temp(m,n) == -1)
                  break;
                end
            end
        end
    end
    temp(1:neighSize,1:Q) = -1;
    temp(1:P,1:neighSize) = -1;
    temp(P-neighSize+1,1:Q) = -1;
    temp(1:P,Q-neighSize+1) = -1;
    
    scoreMapSetOut{i} = temp;
end
    
end