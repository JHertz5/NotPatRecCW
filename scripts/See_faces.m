%% initial setup
clc
close all

if contains(pwd, 'NotPatRecCW')
    dataPath = strcat( extractBefore(pwd, 'NotPatRecCW'), 'NotPatRecCW/data');
    addpath(char(dataPath));
else
    printf('Move to NotPatRecCW directory\n');
end

load 'face.mat'

%% extract individual faces from X

faceW = 46; faceH = 56; nFaceSets = 52; lenFaceSet = 10; nFaces = nFaceSets * lenFaceSet;

indvFaces = zeros(faceH, faceW, nFaces, 'double'); %pre-allocate indvFaces for efficiency
for j = 1:nFaces
    for i = 1:faceW %extract image one line at a time
        indvFaces(1:faceH,i,j) = rot90(X((i-1)*faceH+1:i*faceH, j), 2);
    end
end

%% calculate average face for each face set
aveFaces = zeros(faceH, faceW, nFaceSets, 'double'); %pre-allocate aveFaces for efficiency
for i = 1:nFaceSets
    faceStart = (i-1)*lenFaceSet+1;
    faceEnd = i*lenFaceSet;
    aveFaces(:,:,i) = mean(indvFaces(:,:,faceStart:faceEnd),3);
end

%% show figures
if (exist('showPlots', 'var') && showPlots == true)
    figure(1)
    for j = 1:9
        subplot(3,3,j)
        h = pcolor(aveFaces(:,:,j));
        set(h,'edgecolor','none');
        colormap gray
    end
    set(findobj(gcf, 'type', 'axes'), 'Visible', 'off')

    figure(2)
    for i = 1:10
        for j = 1:10
            subplot(10,10,(i-1)*10+j)
            h = pcolor(indvFaces(:,:,(i-1)*10+j));
            set(h,'edgecolor','none');
            colormap gray
        end
    end
    set(findobj(gcf, 'type', 'axes'), 'Visible', 'off')

    figure(3)
    for i = 1:10
        for j = 1:10
            subplot(10,10,(i-1)*10+j)
            h = pcolor(abs(indvFaces(:,:,(i-1)*10+j)-aveFaces(:,:,i)));
            set(h,'edgecolor','none');
            colormap gray
        end
    end
    set(findobj(gcf, 'type', 'axes'), 'Visible', 'off')

    figure(4)
    h = pcolor(X);
    set(h,'edgecolor','none');
    colormap gray
else
    fprintf('No plots because showPlots != true\n')
end