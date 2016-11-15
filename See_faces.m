clc
clear variables %clear all deletes cached code, slows things down
close all

load 'face.mat'

%extract individual faces from X
indvFaces = zeros(56, 46, 520, 'double'); %declare indvFaces for efficiency
for j = 1:520
    for i = 1:46
        indvFaces(1:56,i,j) = rot90(X((i-1)*56+1:i*56, j), 2);
    end
end

%calculate average face for each face set
aveFaces = zeros(56, 46, 52, 'double'); %declare aveFaces for efficiency
for i = 1:52
    aveFaces(:,:,i) = mean(indvFaces(:,:,(i-1)*10+1:10*i),3);
end

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