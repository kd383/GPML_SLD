function m = meankNN(m)
% source http://www.mathworks.com/matlabcentral/answers/86594-find-k-nearest-neighbours-for-each-element-in-a-matrix
% Get the absolute value of hte differences
% in each of the 8 neighbor directions.
nn1 = abs(conv2(m, [-1,0,0;0,1,0;0,0,0], 'valid'));
nn2 = abs(conv2(m, [0,-1,0;0,1,0;0,0,0], 'valid'));
nn3 = abs(conv2(m, [0,0,-1;0,1,0;0,0,0], 'valid'));
nn4 = abs(conv2(m, [0,0,0;-1,1,0;0,0,0], 'valid'));
nn5 = abs(conv2(m, [0,0,0;0,1,-1;0,0,0], 'valid'));
nn6 = abs(conv2(m, [0,0,0;0,1,0;-1,0,0], 'valid'));
nn7 = abs(conv2(m, [0,0,0;0,1,0;0,-1,0], 'valid'));
nn8 = abs(conv2(m, [0,0,0;0,1,0;0,0,-1], 'valid'));
% Stack these as planes (slices) in a 3D matrix.
array3D = cat(3, nn1, nn2, nn3, nn4, nn5, nn6, nn7, nn8);
m(2:(end-1),2:(end-1)) = mean(array3D,3);
end