function [data,nimages,nrows,ncolumns] = load_data()

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

end