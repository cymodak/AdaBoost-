function [imgOut] = plot_sq(imgIn, coor)
% coor: xtl, ytl, xbr, ybr
figure('Visible', 'off')
imshow(imgIn);hold on
for i = 1:size(coor,1)
    plot([coor(i,1) coor(i,2)], [coor(i,1) coor(i,4)], 'lineWidth', 2, 'color', 'green');
    plot([coor(i,3) coor(i,2)], [coor(i,3) coor(i,4)], 'lineWidth', 2, 'color', 'green');
    plot([coor(i,1) coor(i,2)], [coor(i,3) coor(i,2)], 'lineWidth', 2, 'color', 'green');
    plot([coor(i,1) coor(i,4)], [coor(i,3) coor(i,4)], 'lineWidth', 2, 'color', 'green');
end

hold off

imgOut = export_fig();

end