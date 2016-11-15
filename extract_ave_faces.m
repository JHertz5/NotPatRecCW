%% run initial script
clear variables

showFigures = false;
See_faces
close all

%% calculate X = abs(face-mean) for each face and for each X find avergae pixel value

%pre-allocate for efficiency
absZeroMeanFaces = zeros(56, 46, 520, 'double');
avePixVal = zeros(1, 520, 'double');

for i = 1:520
    absZeroMeanFaces(:,:,i) = abs(indvFaces(:,:,i) - aveFaces(:,:,ceil(i/10)));
    avePixVal(i) = mean(mean(absZeroMeanFaces(:,:,i)));
end

%% for each class find 2 indices corresponding to most average face
%  most average face will have average pixel value closest to zero
actual_idx = zeros(1,104);
for j = 1:2
    for i = 1:52
        [~, idx] = min(avePixVal((i-1)*10+1:10*i));
        if j == 1
            actual_idx(2*i-1) = idx + (i-1)*10;
            avePixVal(actual_idx(2*i-1)) = 1000;
        else
            actual_idx(2*i) = idx + (i-1)*10;
            avePixVal(actual_idx(2*i)) = 1000;
        end
    end
end

%% sort indices in the ascending order
actual_idx = sort(actual_idx); % these indices correspond to the 2 most average faces in each class

%% separate face data into training and testing data
testing = X(:,actual_idx);
training = X;
training(:,actual_idx)=[];
save('Separated_Data','testing','training')