
close all; clear; clc;

%% Initialization
% subject ID
sub_id = 1;
% trial ID
trial_id = 3;
% number of positions
numOfPosit = 2;
% path to the video files
path_vid = sprintf('C:\\Users\\user\\Documents\\MATLAB\\test_data\\sub_%d\\trial_%d\\video_audio\\', sub_id, trial_id);
% path for saving thermal images
path_thr = sprintf('C:\\Users\\user\\Documents\\MATLAB\\test_data\\sub_%d\\trial_%d\\thr_image\\', sub_id, trial_id);
if not(exist(path_thr,'dir'))
    mkdir(path_thr);
end
% path for saving rbg images
path_rgb = sprintf('C:\\Users\\user\\Documents\\MATLAB\\test_data\\sub_%d\\trial_%d\\rgb_image\\', sub_id, trial_id);
if not(exist(path_rgb,'dir'))
    mkdir(path_rgb);
end

%%
count = 1;
while count <= numOfPosit
    path_thr_full = sprintf('%s%d_%d_%d_%d.avi', path_vid, sub_id, trial_id, count, 0);
    path_rgb_full = sprintf('%s%d_%d_%d_%d.avi', path_vid, sub_id, trial_id, count, 1);
    
    if isfile(path_thr_full) && isfile(path_rgb_full)
        v_rgb = VideoReader(path_rgb_full);
        v_thr = VideoReader(path_thr_full);
        
        frame = 1;
        while hasFrame(v_rgb) && hasFrame(v_thr)
            rgb_frame = readFrame(v_rgb);
            thr_frame = readFrame(v_thr);
            
            filenameRGB = sprintf('%s%d_%d_%d_%d_%d.png', path_rgb, sub_id, trial_id, count, frame, 1);
            filenameThr = sprintf('%s%d_%d_%d_%d_%d.png', path_thr, sub_id, trial_id, count, frame, 0);
            
            imwrite(rgb_frame, filenameRGB);
            imwrite(thr_frame, filenameThr);
            frameInfo = sprintf('Position: %d, Frame: %d',count,frame);
            disp(frameInfo);
            frame = frame + 1;
        end
        clear v_rgb v_thr
        
    else
        flt = sprintf('%s and/or %s dont/doesnt exist.',path_thr_full,path_rgb_full);
        disp(flt)
    end
    
    count = count + 1;
end

clear v_rgb v_thr
