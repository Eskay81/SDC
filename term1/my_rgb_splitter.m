## Copyright (C) 2016 ESKAY
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{retval} =} my_rgb_splitter (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: ESKAY <ESKAY@ESKAY_MACHINE>
## Created: 2016-12-01

function [retval] = my_rgb_splitter (image_file)

	my_img = imread(image_file);
	
	my_img_width = size(my_img,1);
	my_img_height = size(my_img,2);
	
	% Create red,green and blue channels
	red_chnl = my_img(:,:,1);
	green_chnl = my_img(:,:,2);
	blue_chnl = my_img(:,:,3);
	
	zero_mat = zeros(my_img_width,my_img_height,'uint8');
	
	%creating image for R,G,B seperatly
	my_red_img = cat(3,red_chnl,zero_mat,zero_mat);
	my_green_img = cat(3,zero_mat,green_chnl,zero_mat);
	my_blue_img = cat(3,zero_mat,zero_mat,blue_chnl);
	
	%displaying all in subplot form
	subplot(2,3,2); imshow(my_img);title('Original Image');
	subplot(2,3,4);imshow(my_red_img);title('Red Channel');
	subplot(2,3,5);imshow(my_green_img);title('Green Channel');
	subplot(2,3,6);imshow(my_blue_img);title('Blue Channel');
	
	% Seperating the white part (just for lanes)
	% Since RGB has to be more than 200 and almost equal for white we can just do with Red here
	[my_rows,my_cols,my_vals] = find(my_red_img > 200);
	my_red_white = zeros(my_img_width,my_img_height,'uint8');
	
	for temp_counter = 1 : length(my_rows)
		my_red_white(my_rows(temp_counter),my_cols(temp_counter)) = 255;
	end
	
	figure(2);
	imshow(my_red_white);
		
endfunction
