function draw_coordinate(a, b, center_x, center_y)

% a: long  axis( X )
% b: short axis( Y )
d = 0:360;
x = a * cosd(d);
y = b * sind(d);
patch(x + center_x, y + center_y, 'b', 'edgecolor', 'none', 'facealpha',0.8);   %²»Í¸Ã÷¶È
