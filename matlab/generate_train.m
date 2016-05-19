clear;close all;
%% settings
folder = '../../SRCNN_dataset/SRCNN/Train';
size_input = 42;
% size_label = 21;
size_label = 42;
scale = 3;
stride = 14;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
% padding = abs(size_input - size_label)/2;
padding = 0;
count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));
    
for i = 1 : length(filepaths)
    
    image = imread(fullfile(folder,filepaths(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));

    im_label = modcrop(image, scale);
    [hei,wid] = size(im_label);
    im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');

    for x = 1 : stride : hei-size_input+1
        for y = 1 :stride : wid-size_input+1
            
            subim_input = im_input(x : x+size_input-1, y : y+size_input-1);
            subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);

            count=count+1;
            data(:, :, 1, count) = subim_input;
            label(:, :, 1, count) = subim_label;
        end
    end
end

order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order); 

fileID = fopen('train_data.bin', 'w');
fwrite(fileID, data, 'float');
fclose(fileID);

fileID = fopen('train_label.bin', 'w');
fwrite(fileID, label, 'float');
fclose(fileID);

%%

