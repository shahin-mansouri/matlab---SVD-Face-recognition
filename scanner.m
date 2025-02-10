clear; clc;
format compact;
close All;


img = imread("scanner.jpg");
img = im2gray(img);
img = im2double(img);

pointer = 1;
mid = size(img, 2)/2;
f = []

pointers = linspace(0, size(img, 2), 50);
for i=1:length(pointers)
    f = [f, findPaper(img, i)];
end
fitlm(pointers, f).Coefficients
imshow(img(1:f, :))

function pointer = findPaper(img, c)
    pointer = 1;
    for i=1:size(img, 1)-1
        prev = pointer;
        pointer = pointer+1;
        if abs(img(pointer, c) - img(prev, c)) > .3
            break
        end
    end
end

