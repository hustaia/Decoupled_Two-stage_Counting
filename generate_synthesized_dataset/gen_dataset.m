function gen_dataset(x_rate, y_rate, total_num, blur_type, opt)
% INPUT:
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
    gen_image(x_rate, y_rate, blur_type, i, opt);
end    

fprintf('\n---     finish     ---\n')