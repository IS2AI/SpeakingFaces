close all; clc; clear;

%% Initialization

% !!!!!!! run this line if you have relaunched matlab !!!!!!!!!
% rosinit;
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

% subject ID
sub_id = 1;
% trial ID
trial_id = 3;
% total number of frames
numOfFrames = 900;
% total number of positions within one experiment
numOfPosit = 2;
% pause between positions (sec)
pauseBtwPos = 10;
% cameras fps (Hz)
fps_cam = 28;
% loop rate for image acquis.
r = rosrate(fps_cam);

% path to store video and audio files
path = sprintf('C:\\Users\\user\\Documents\\MATLAB\\test_data\\sub_%d\\trial_%d\\video_audio\\', sub_id, trial_id);
if not(exist(path,'dir'))
    mkdir(path);
end
    
%% 
% just to check the loop rate
%time = zeros(1,numOfFrames);

atPath = getenv('FLIR_Atlas_MATLAB');
atLive = strcat(atPath,'Flir.Atlas.Live.dll');
asmInfo = NET.addAssembly(atLive);
%init camera discovery
flir_init = Flir.Atlas.Live.Discovery.Discovery;

disc = flir_init.Start(10);
% for Thermal camera
thr_strm = Flir.Atlas.Live.Device.ThermalCamera(true);    
thr_strm.Connect(disc.Item(3));

%set the Iron palette
pal = thr_strm.ThermalImage.PaletteManager;
thr_strm.ThermalImage.Palette = pal.Iron;

% for RGB camera
rgb_strm = videoinput('winvideo', 1, 'MJPG_1920x1080');
set(rgb_strm, 'FramesPerTrigger', Inf);
set(rgb_strm, 'ReturnedColorspace', 'rgb');
%set(rgb_strm, 'FocusMode', 'manual');
set(rgb_strm, 'ROIPosition',[576 500 768 512]);%512 300 896 640
rgb_strm.FrameGrabInterval = 1;  % distance between captured frames 

start(rgb_strm)

count = 1;

beep();

while numOfPosit>=count
    filenameThr = sprintf('%s%d_%d_%d_%d.avi', path, sub_id, trial_id, count, 0);
    filenameRGB = sprintf('%s%d_%d_%d_%d.avi', path, sub_id, trial_id, count, 1);
    filenameMic1 = sprintf('%s%d_%d_%d_%d.wav', path, sub_id, trial_id, count, 1); % mic1
    filenameMic2 = sprintf('%s%d_%d_%d_%d.wav', path, sub_id, trial_id, count, 2); % mic2
    filenameMsgs = sprintf('%s%d_%d_%d.mat', path, sub_id, trial_id, count);
    
    v_thr = VideoWriter(filenameThr, 'Uncompressed AVI');
    v_thr.FrameRate = fps_cam;
    v_rgb = VideoWriter(filenameRGB, 'Uncompressed AVI');
    v_rgb.FrameRate = fps_cam;
    open(v_thr);
    open(v_rgb);    

    %tic
    for frame=1:numOfFrames
        %tic      
        %img_thr = thr_strm.ThermalImage.ImageArray;
        img_thr_cnv = uint8(thr_strm.ThermalImage.ImageArray);
        img_rgb = getsnapshot(rgb_strm);
                         
        writeVideo(v_thr, img_thr_cnv);
        writeVideo(v_rgb, img_rgb);
        
        waitfor(r);
        %frame
        %time(frame)=toc;
    end
     
    close(v_thr);
    close(v_rgb);
    
    clear v_thr v_rgb
    beep;
    frameInfo = sprintf('Position: %d, Total Frames: %d',count,frame);
    count = count + 1;
    disp(frameInfo);
    pause(pauseBtwPos);
    clc
end

stop(rgb_strm);

thr_strm.Disconnect();
thr_strm.Dispose();

reset(r);

% save filenameMsgs synch_time
% avgFPS = 1.0/mean(time)
% histogram(time)