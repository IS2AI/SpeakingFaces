close all; clc; clear;

%% Initialization
% subject ID
sub_id = 1;
% trial ID
trial_id = 2;
% number of positions
numOfPosit = 2;

% path to the video and audio files
path_vid = sprintf('C:\\Users\\user\\Documents\\MATLAB\\test_data\\sub_%d\\trial_%d\\video_audio\\', sub_id, trial_id);
% path for saving thermal video commands
path_thr = sprintf('C:\\Users\\user\\Documents\\MATLAB\\test_data\\sub_%d\\trial_%d\\thr_video_cmd\\', sub_id, trial_id);
if not(exist(path_thr,'dir'))
    mkdir(path_thr);
end
% path for saving rbg video commands
path_rgb = sprintf('C:\\Users\\user\\Documents\\MATLAB\\test_data\\sub_%d\\trial_%d\\rgb_video_cmd\\', sub_id, trial_id);
if not(exist(path_rgb,'dir'))
    mkdir(path_rgb);
end
% path for saving audio commands
path_mic1 = sprintf('C:\\Users\\user\\Documents\\MATLAB\\test_data\\sub_%d\\trial_%d\\mic1_audio_cmd\\', sub_id, trial_id);
if not(exist(path_mic1,'dir'))
    mkdir(path_mic1);
end
% path for saving audio commands
path_mic2 = sprintf('C:\\Users\\user\\Documents\\MATLAB\\test_data\\sub_%d\\trial_%d\\mic2_audio_cmd\\', sub_id, trial_id);
if not(exist(path_mic2,'dir'))
    mkdir(path_mic2);
end
%%
count = 1;
while count <= numOfPosit
    path_thr_vid = sprintf('%s%d_%d_%d_%d.avi', path_vid, sub_id, trial_id, count, 0);
    path_rgb_vid = sprintf('%s%d_%d_%d_%d.avi', path_vid, sub_id, trial_id, count, 1);
    path_rgb_mic1 = sprintf('%s%d_%d_%d_%d.wav', path_vid, sub_id, trial_id, count, 1);
    path_rgb_mic2 = sprintf('%s%d_%d_%d_%d.wav', path_vid, sub_id, trial_id, count, 2);
    path_msgs = sprintf('%s%d_%d_%d.mat', path_vid, sub_id, trial_id, count);
    
    if isfile(path_thr_vid) && isfile(path_rgb_vid)
        v_rgb = VideoReader(path_rgb_vid);
        v_thr = VideoReader(path_thr_vid);
        [y1,Fs1] = audioread(path_rgb_mic1);
        [y2,Fs2] = audioread(path_rgb_mic2);
        msgs = load(path_msgs);
        msgs_len = length(msgs.msgs(:,1));
        
        
        frame = 1;
        ind = 1;
        while hasFrame(v_rgb) && hasFrame(v_thr)
            rgb_frame = readFrame(v_rgb);
            thr_frame = readFrame(v_thr);
            
            if frame <= str2double(msgs.msgs(ind,5))
                if frame == str2double(msgs.msgs(ind,4))
                    frame
                    filenameRGB = sprintf('%s%d_%d_%d_%d_%s.avi', path_rgb, sub_id, trial_id, count, 1, msgs.msgs(ind,2));
                    filenameThr = sprintf('%s%d_%d_%d_%d_%s.avi', path_thr, sub_id, trial_id, count, 0, msgs.msgs(ind,2));
                    v_thr_spl = VideoWriter(filenameThr, 'Uncompressed AVI');
                    v_thr_spl.FrameRate = v_thr.FrameRate;
                    v_rgb_spl = VideoWriter(filenameRGB, 'Uncompressed AVI');
                    v_rgb_spl.FrameRate = v_rgb.FrameRate;
                    open(v_thr_spl);
                    open(v_rgb_spl);
                    filenameMic1 = sprintf('%s%d_%d_%d_%d_%s.wav', path_mic1, sub_id, trial_id, count, 1, msgs.msgs(ind,2)); % mic1
                    filenameMic2 = sprintf('%s%d_%d_%d_%d_%s.wav', path_mic2, sub_id, trial_id, count, 2, msgs.msgs(ind,2)); % mic2
                    x_start = floor(1/v_rgb.FrameRate*Fs1*str2double(msgs.msgs(ind,4)));
                    x_end = floor(1/v_rgb.FrameRate*Fs1*(str2double(msgs.msgs(ind,5))));
                    
                    if x_end <= length(y1)
                        y1_spl = y1(x_start:x_end);
                    else
                        y1_spl = y1(x_start:length(y1));
                    end
                    
                    if x_end <= length(y2)
                        y2_spl = y2(x_start:x_end);
                    else
                        y2_spl = y2(x_start:length(y2));
                    end
                    
                    audiowrite(filenameMic1,y1_spl,Fs1);
                    audiowrite(filenameMic2,y2_spl,Fs2);
                end
                writeVideo(v_thr_spl, thr_frame);
                writeVideo(v_rgb_spl, rgb_frame);
                if frame == str2double(msgs.msgs(ind,5))
                    close(v_thr_spl);
                    close(v_rgb_spl);
                    if ind < msgs_len
                        ind = ind + 1;
                    end
                end
            end
            
            %frameInfo = sprintf('Position: %d, Frame: %d',count,frame);
            %disp(frameInfo);
            frame = frame + 1;
        end
        clear v_rgb v_thr
        
    else
        flt = sprintf('%s and/or %s dont/doesnt exist.',path_thr_vid,path_rgb_vid);
        disp(flt)
    end
    
    count = count + 1;
end

clear v_rgb v_thr
