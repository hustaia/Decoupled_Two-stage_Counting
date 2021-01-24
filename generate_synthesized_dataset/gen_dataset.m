function gen_dataset(min_num, max_num, x_rate, y_rate, total_num, blur_type, opt)
% INPUT:
% min_num:   min number of cells in 32*32 patch
% max_num:   max number of cells in 32*32 patch
% total_num: total image num
% blur_type: the type of blur operation

% create save folder
save_path = opt.save_folder;
if ~exist(save_path)        
    mkdir(save_path);
end

% create image folder
image_path = [save_path '/image'];
opt.image_path = image_path;
if ~exist(image_path) 
    mkdir(image_path);
end

% create dotmap folder
dotmap_path = [save_path '/dotmap'];
opt.dotmap_path = dotmap_path;
if ~exist(dotmap_path) 
    mkdir(dotmap_path);
end

fprintf('\n--- total image: %d ---', total_num);

% generate image & dotmap
for i = 1:total_num
    fprintf('\nimage: %6d', i)
    gen_image(min_num, max_num, x_rate, y_rate, blur_type, i, opt);
end    

fprintf('\n---     finish     ---\n')