close all
clear
clc

% FORMAT:  (min_cell_num, max_cell_num, x_rate, y_rate, total_image_num, blur_type, opt)
% blur_type: 'none' | 'gaussian' | 'disk' | 'motion'

% % % Default: 
% % % save images  to  [opt.save_folder '/image' ]  folder
% % % save dotmaps to  [opt.save_folder '/dotmap']  folder
x_rate=16; 
y_rate=16;
total_image_num=1000;
blur_type =  'gaussian';
opt.dataset = 'A';
switch opt.dataset
    case 'A'
    opt.save_folder = '../data/synthesized_dataset/partA';
    case 'B'
    opt.save_folder = '../data/synthesized_dataset/partB';
    case 'Q'
    opt.save_folder = '../data/synthesized_dataset/QNRF';
end
gen_dataset(x_rate, y_rate, total_image_num, blur_type, opt);

