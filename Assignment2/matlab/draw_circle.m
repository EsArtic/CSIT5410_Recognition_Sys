function canvas = draw_circle(canvas, Xc, Yc, radius)
% On canvas draw a circle centered at [Xc, Yc] with radius specified

for x=1:1:size(canvas, 1)
    for y=1:1:size(canvas, 2)
        if abs((x-Xc)*(x-Xc)+(y-Yc)*(y-Yc)-radius*radius)<100
            canvas(x,y)=1;
        end
    end
end
end


