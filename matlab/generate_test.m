clear;close all;
%% settings
folder = '../../SRCNN_dataset/SRCNN/Test/Set14';
% size_input = 64;
size_input = 44;
% size_label = 21;
size_label = size_input;
scale = 3;
stride = size_input;

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
	if size(image,3) == 3;
    	image = rgb2ycbcr(image);
	end
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

fileID = fopen('test_data.bin', 'w');
fwrite(fileID, data, 'float');
fclose(fileID);

fileID = fopen('test_label.bin', 'w');
fwrite(fileID, label, 'float');
fclose(fileID);

