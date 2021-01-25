function [coordinate, radius] = gen_patch(dataset)

% define cell radius
ratio = 0.2;
base_radius = 6;
max_radius = base_radius * (1 + ratio);
min_radius = base_radius * (1 - ratio);

switch dataset
    case 'A'
        a = csvread('A_patchnum.csv', 2, 0,[2,0,39,0]);
    case 'B'
        a = csvread('B_patchnum.csv', 2, 0,[2,0,24,0]);
    case 'Q'
        a = csvread('Q_patchnum.csv', 2, 0,[2,0,42,0]);
end
a = 1 - a / sum(a);
a = a / sum(a);
a = cumsum(a);
b = rand(1);
cur_num=0;
for i=1:length(a)
    if b > a(i)
        cur_num = i;
    end
end

% initialize mask
mask = zeros(32, 32);
coordinate = zeros(cur_num, 2);
radius = zeros(cur_num, 2);
index = 1;

%fprintf('--- cur_num: %d ---', cur_num);
if cur_num > 0
    switch dataset
    case 'A'
        lysize = 2;
    case 'B'
        lysize = 4;
    case 'Q'
        lysize = 2;
    end
    for i = 1:100000000

        x = max( randi(32),1);
        y = max( randi(32),1);

        % crop patch to judge whether (x, y) is suitable
        
        left  = uint8(max(1,  x - lysize));
        right = uint8(min(32, x + lysize));    
        top   = uint8(max(1,  y - lysize));
        down  = uint8(min(32, y + lysize));        
        %crop = mask(left:right, top:down);
        crop = mask(top:down, left:right);

        % if (x, y) is qualified, record (x, y ,radius)
        if sum(crop(:)) == 0        
            coordinate(index, 1) = x;
            coordinate(index, 2) = y;

            rand_x = rand();
            rand_y = rand();        
            radius_x = rand_x * (max_radius - min_radius) + min_radius;
            radius_y = rand_y * (max_radius - min_radius) + min_radius;
            radius(index, 1) = radius_x;
            radius(index, 2) = radius_y;

            mask(y, x) = 1;
            index = index + 1;
        end

        if index > cur_num
            break
        end

    end    
end    
