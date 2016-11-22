%% run initial script
clear variables

showFigures = false;
See_faces
close all

%% calculate X = abs(face-mean) for each face and for each X find avergae pixel value

faceW = 46; faceH = 56; nFaceSets = 52; lenFaceSet = 10; nFaces = nFaceSets * lenFaceSet;

%pre-allocate for efficiency
absZeroMeanFaces = zeros(faceH, faceW, nFaces, 'double');
avePixVal = zeros(1, nFaces, 'double');

for i = 1:nFaces
    absZeroMeanFaces(:,:,i) = abs(indvFaces(:,:,i) - aveFaces(:,:,ceil(i/10)));
    avePixVal(i) = mean(mean(absZeroMeanFaces(:,:,i)));
end

%% for each class find 2 indices corresponding to most average face
%  most average face will have average pixel value closest to zero
actual_index = zeros(1,104);
for j = 1:2
    for i = 1:nFaceSets
        [~, index] = min(avePixVal((i-1)*10+1:10*i));
        if j == 1
            actual_index(2*i-1) = index + (i-1)*10;
            avePixVal(actual_index(2*i-1)) = 1000;
        else
            actual_index(2*i) = index + (i-1)*10;
            avePixVal(actual_index(2*i)) = 1000;
        end
    end
end

%% sort indices in the ascending order
actual_index = sort(actual_index); % these indices correspond to the 2 most average faces in each class

%% separate face data into training and testing data
testing = X(:,actual_index);
training = X;
training(:,actual_index)=[];
save('Separated_Data','testing','training')