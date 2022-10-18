% get the appropriate video format in the folder specified 
videofiles = dir([foldername '/*qcif.yuv']);
% get the number of available videos
numvideo = numel(videofiles);
% select a video at random
filename = randi(numvideo);
filename = [foldername '\' videofiles(filename).name];
% select a random starting frame
frame_start = randi(frame_end-T_s); 
% import the video
vid = my_yuv_import(filename, [176 144], T_s,'output','mat'); 
% can use implay to see video in array format
% Renormalize to [-1, 1] and extract the signals of interest (need power of
% 2 for noiselets)
vid = 2*(double(vid(8:135, 24:151,:))/255 - 0.5);