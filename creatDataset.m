
% the principle is the file will have .pgm
% the data structure is the same

level1list = dir('*');
level1list(~[level1list.isdir]) = [];
tf = ismember({level1list.name},{'.','..'});
level1list(tf) = [];
faceData = []
cur = 1;
for K = 1:length(level1list);
    subdir = dir(fullfile(level1list(K).name,'*.pgm'));
    if isempty(subdir);
        continue
    end
    for i = 1:length(subdir);
        I = imread(fullfile(level1list(K).name, subdir(i).name));
        I = im2double(I);
        A = zeros(112*92,1);
        A(1:end) = I(1:end);
        faceData = [faceData A];
    end
end
save faceData
