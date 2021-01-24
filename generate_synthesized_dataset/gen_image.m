function  gen_image(min_num, max_num, x_rate, y_rate, blur_type, index, opt)

figure('visible','off');
cla;

% figure min size is 120 !!!

% set figure, make image fill whole figure
y_factor = max(y_rate, 2);
x_factor = max(x_rate, 2);    
white_image = zeros(y_factor*32, x_factor*32);
imshow(white_image, 'border', 'tight', 'initialmagnification', 'fit');
set(gcf,'Position',[100, 100, x_factor*32, y_factor*32])
axis normal;

% reverse axis
set(gca,'ydir','reverse','xaxislocation','top');

dotmap = zeros(y_rate*32, x_rate*32);

% generate patch
for i = 1:y_rate
    for j = 1:x_rate
        
        [coordinate, radius] = gen_patch(max_num, min_num, opt.dataset);
        coordinate(:, 1) = coordinate(:, 1) + (j - 1) * 32;
        coordinate(:, 2) = coordinate(:, 2) + (i - 1) * 32;
                
        %for k = 1:length(coordinate)
        for k = 1:size(coordinate, 1)
    
            x = coordinate(k, 1);
            y = coordinate(k, 2);
            
            r_x = radius(k, 1);
            r_y = radius(k, 2);

            draw_coordinate(r_x, r_y, x, y);
                        
            dotmap(y, x) = 255;

        end                  
    end
end

axis equal
axis off;
set(gcf,'color','k');
%set(f,'visible','on')

% delete redundant boundary
%set(gca,'LooseInset',get(gca,'TightInset'))
%set(gca,'looseInset',[0 0 0 0])

% get figure image data
f=getframe(gcf);
img = f.cdata;

% crop img if neccesary
if y_rate == 1
    img = img(1:32, :, :);
end

if x_rate == 1
    img = img(:, 1:32, :);
end

img_name = sprintf('/%06d.jpg', index);
img_name = [opt.image_path img_name];

% save dotmap
dot_name = sprintf('/%06d.mat', index);
dot_name = [opt.dotmap_path dot_name];
save(dot_name,'dotmap');
%imwrite(dotmap, dot_name);

% apply blur operation & save image
if strcmp(blur_type, 'none')>0 
    img = img(:,:,[1,3,2]);
    imwrite(img, img_name)
else
    switch(blur_type)
        case 'gaussian'
            PSF = fspecial('gaussian',[5 5],5); 
        case 'disk'
            PSF=fspecial('disk', 5); 
        case 'motion'
            PSF = fspecial('motion',20,15);        
    end
    img_blur = imfilter(img, PSF, 'symmetric','conv'); 
    img_blur = img_blur(:,:,[1,3,2]);
    imwrite(img_blur, img_name)

    
    
end    
        
        