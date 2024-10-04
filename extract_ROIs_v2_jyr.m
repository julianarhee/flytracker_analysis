%% Retrieve behavioral data and imaging time stamps
clearvars -except store

%load('121020_splitP1CsX_LC10LexAGCaMP6s_largeDot_Stim_f3.mat')
%fname = '121020_VT29314GC6m_splitP1CsX_MultiPanel_Stim_largeDot_f3-001.xml';
%vname = '121020_VT29314GC6m_splitP1CsX_MultiPanel_Stim_largeDot_f3-001_Cycle00001_VoltageRecording_001.csv';
%videoName = '121020_VT29314GC6m_splitP1CsX_MultiPanel_Stim_largeDot_f3-001_DN_nrAligned.tif';
basedir = '/Volumes/Juliana/2p-data';
session = '20240905';
virmen_fname = '20240905_Dmel-ChAT-GCaMP6f-tdTomato_f3v1_OL-SimpleOscillatingTarget_018';
virmenpath = fullfile(basedir, session, 'virmen', sprintf('%s.mat', virmen_fname));
load(virmenpath)

imdir = '20240905_Dmel-ChAT-sytGCaMP6f-tdTomato_f3v1-018';
fname = fullfile(basedir, session, 'raw', imdir, '20240905_Dmel-ChAT-sytGCaMP6f-tdTomato_f3v1-018.xml');
vname = fullfile(basedir, session, 'raw', imdir, '20240905_Dmel-ChAT-sytGCaMP6f-tdTomato_f3v1-018_Cycle00001_VoltageRecording_001.csv');
videoName = fullfile(basedir, session, 'processed', '20240905_Dmel-ChAT-sytGCaMP6f-tdTomato_f3v1-018_Channel01_nrAligned.tif');

%%
metaIm = xml2struct(fname);
voltage = csvread(vname,1,1);

%find start times of behavior and imaging;
imStart = strsplit(metaIm.PVScan.Sequence.Attributes.time,':');
imStart = [str2double(imStart{1}) str2double(imStart{2}) str2double(imStart{3})];
nIms = numel(metaIm.PVScan.Sequence.Frame);
for i = 1:nIms
    absFrameTimes(i) = str2double(metaIm.PVScan.Sequence.Frame{i}.Attributes.absoluteTime);
end
behStart = expr(1,8:10);
bTime = [0;expr(2:end,1)];

%find voltage times for syncing
[~,id] = findpeaks(diff(voltage(:,1)),'MinPeakHeight',1);
id = id+1;
pulseRCD = id./1000;
id_ML = find(expr(:,11) == 1);
pulseSent = expr(id_ML,1);

clear offset
for i = 1:size(pulseRCD,1)
    offset(i) = pulseSent(i) - pulseRCD(i);
end

%align behavior and imaging;
iTime = absFrameTimes + mean(offset(1));
inversion = expr(:,12);

stim.pos = expr(:,6:7); %female position in ViRMEn coords
stim.angle = atan2(stim.pos(:,2),stim.pos(:,1));
switchLeft = regionprops(diff(stim.angle)>0,'PixelList'); % switch left-to-right?
switchRight = regionprops(diff(stim.angle)<0,'PixelList'); 
for i = 1:length(switchLeft);
    stimLROnsets(i) = switchLeft(i).PixelList(1,2);
end
for i = 1:length(switchRight);
    stimRLOnsets(i) = switchRight(i).PixelList(1,2);
end

idInv = find(inversion(stimLROnsets) == 1);
idnonInv = find(inversion(stimLROnsets) == 0);
%% Get ROIs

%addpath(genpath('/Users/tom/Documents/MATLAB/ImRegCa'));
addpath(genpath('/Users/julianarhee/Documents/MATLAB/CaImAn-MATLAB-master'));

Y = read_file(videoName);
meanFLC = squeeze(mean(mean(Y,2),1));
shutterClosed = find(meanFLC < (mean(meanFLC) - std(meanFLC)));
Y(:,:,shutterClosed) = [];
iTime(shutterClosed) = [];

%discard inverted frames
lastStim = bTime(stimLROnsets(end-1)); % bTime(stimLROnsets(10));
[~,idxlastLR] = min(abs(iTime-lastStim));
Y = Y(:,:,1:idxlastLR);
iTime = iTime(1:idxlastLR);

%%
%discard image edges; 
buffer = 5; 
Y = Y(buffer:end-buffer,buffer:end-buffer,:);

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;

K = 500; %200;                                           % number of components to be found
tau = 0.75; %0.75;                                          % std of gaussian kernel (half size of neuron)
p = 2;

options = CNMFSetParms(...
    'd1',d1,'d2',d2,...                         % dimensionality of the FOV
    'p',p,...                                   % order of AR dynamics
    'gSig',tau,...                              % half size of neuron
    'merge_thr',0.80,...                        % merging threshold
    'nb',2,...                                  % number of background components
    'min_SNR',3,...                             % minimum SNR threshold
    'space_thresh',0.3,...                      % space correlation threshold
    'cnn_thr',0.2,...                            % threshold for CNN classifier
    'init_method','greedy',...
    'decay_time',0.5 ...
    );

%%
[P,Y] = preprocess_data(Y,p);
% fast initialization of spatial components using greedyROI and HALS

[Ain,Cin,bin,fin,center] = initialize_components(Y,K,tau,options,P);  % initialize

%refine
Cn =  correlation_image(Y);
refine_components = true;  % flag for manual refinement
if refine_components
    [Ain,Cin,center] = manually_refine_components(Y,Ain,Cin,center,mean(Y,3),tau,options);
end

% update spatial components
Yr = reshape(Y,d,T);
[A,b,Cin] = update_spatial_components(Yr,Cin,fin,[Ain,bin],P,options);

% update temporal components
P.p = 0;    % set AR temporarily to zero for speed
[C,f,P,S,YrA] = update_temporal_components(Yr,A,b,Cin,fin,P,options);

% classify components

rval_space = classify_comp_corr(Y,A,C,b,f,options);
ind_corr = rval_space > options.space_thresh;           % components that pass the correlation test
                                        % this test will keep processes
% further classification with cnn_classifier
try  % matlab 2017b or later is needed
    [ind_cnn,value] = cnn_classifier(A,[d1,d2],'cnn_model',options.cnn_thr);
catch
    ind_cnn = true(size(A,2),1);                        % components that pass the CNN classifier
end    

% event exceptionality

fitness = compute_event_exceptionality(C+YrA,options.N_samples_exc,options.robust_std);
ind_exc = (fitness < -30 & fitness > -Inf);

% select components

keep = (ind_corr | ind_cnn) & ind_exc;

% display kept and discarded components
A_keep = A(:,keep);
C_keep = C(keep,:);
figure;
    subplot(121); montage(extract_patch(A(:,keep),[d1,d2],[30,30]),'DisplayRange',[0,0.15]);
        title('Kept Components');
    subplot(122); montage(extract_patch(A(:,~keep),[d1,d2],[30,30]),'DisplayRange',[0,0.15])
        title('Discarded Components');
        
%% merge found components
[Am,Cm,K_m,merged_ROIs,Pm,Sm] = merge_components(Yr,A_keep,b,C_keep,f,P,S,options);

%
display_merging = 1; % flag for displaying merging example
if and(display_merging, ~isempty(merged_ROIs))
    i = 1; %randi(length(merged_ROIs));
    ln = length(merged_ROIs{i});
    figure;
        set(gcf,'Position',[300,300,(ln+2)*300,300]);
        for j = 1:ln
            subplot(1,ln+2,j); imagesc(reshape(A_keep(:,merged_ROIs{i}(j)),d1,d2)); 
                title(sprintf('Component %i',j),'fontsize',16,'fontweight','bold'); axis equal; axis tight;
        end
        subplot(1,ln+2,ln+1); imagesc(reshape(Am(:,K_m-length(merged_ROIs)+i),d1,d2));
                title('Merged Component','fontsize',16,'fontweight','bold');axis equal; axis tight; 
        subplot(1,ln+2,ln+2);
            plot(1:T,(diag(max(C_keep(merged_ROIs{i},:),[],2))\C_keep(merged_ROIs{i},:))'); 
            hold all; plot(1:T,Cm(K_m-length(merged_ROIs)+i,:)/max(Cm(K_m-length(merged_ROIs)+i,:)),'--k')
            title('Temporal Components','fontsize',16,'fontweight','bold')
        drawnow;
end

%% refine estimates excluding rejected components

Pm.p = p;    % restore AR value
[A2,b2,C2] = update_spatial_components(Yr,Cm,f,[Am,b],Pm,options);
[C2,f2,P2,S2,YrA2] = update_temporal_components(Yr,A2,b2,C2,f,Pm,options);


% do some plotting

[A_or,C_or,S_or,P_or] = order_ROIs(A_keep,C_keep,S(keep,:),P); % order components
K_m = size(C_or,1);
[C_df,~] = extract_DF_F(Yr,A_or,C_or,P_or,options); % extract DF/F values (optional)

figure;
[Coor,json_file] = plot_contours(A_or,Cn,options,1); % contour plot of spatial footprints
%savejson('jmesh',json_file,'filename');        % optional save json file with component coordinates (requires matlab json library)

%% store components, filter tc into time bins, compute direction selectivity index. 
discard = []; %[3 1 40]; %discard any obviously wrong components
keep = setdiff(1:size(A_or,2),discard);

m = 2; 
CFull = full(C_df(keep,:));
maxTime = stimLROnsets(end-1); %stimLROnsets(10);

timeBins = [-2:0.1:4];

tc = CFull;
clear meanRespLR meanRespRL
%compute response to LR sweeps
for j = 1:length(stimLROnsets)
    relTimeB = bTime - bTime(stimLROnsets(j));
    relTimeI = iTime - bTime(stimLROnsets(j));
        for i = 1:length(timeBins)-1;
            idxI = find(relTimeI > timeBins(i) & relTimeI < timeBins(i+1)); 
            meanRespLR(i,j,:) = nanmean(tc(:,idxI),2);
        end
end

%compute response to RL sweeps
for j = 1:length(stimLROnsets)
    relTimeB = bTime - bTime(stimRLOnsets(j));
    relTimeI = iTime - bTime(stimRLOnsets(j));
        for i = 1:length(timeBins)-1
            idxI = find(relTimeI > timeBins(i) & relTimeI < timeBins(i+1)); 
            meanRespRL(i,j,:) = nanmean(tc(:,idxI),2);
        end
end

EVOO_RL = squeeze(nanmean(meanRespRL(:,1:9,:),2)); 
EVOO_LR = squeeze(nanmean(meanRespLR(:,1:9,:),2)); 
selectivityIndex = (max(EVOO_RL) - max(EVOO_LR))./(max(EVOO_RL) + max(EVOO_LR));

store(m).name = videoName;
store(m).time = timeBins; 
store(m).tc = tc; 
store(m).A_or = A_or(:,keep); 
store(m).Cn = Cn; 
store(m).options = options; 
store(m).selIndex = selectivityIndex; 
store(m).evokedProg = EVOO_RL; % check the direction
store(m).evokedReg = EVOO_LR; 

figure;
imagesc([EVOO_LR EVOO_RL]')

%%
iTime_fps = 1/mean(diff(iTime));
nsec_pre = 0;
nsec_post = 3.5;

sec_between_sweeps = round(mean(diff(bTime(stimLROnsets))));

nframes_pre = round(nsec_pre * iTime_fps);
nframes_post = round(nsec_post * iTime_fps);

[n_timebins, n_trials, n_rois] = size(meanRespLR);

clear timecoursemat_lr timecoursemat_rl
%compute response to LR sweeps
for j = 1:length(stimLROnsets)
    relTimeI = iTime - bTime(stimLROnsets(j));
    [c iTime_ix] = min(abs(iTime - bTime(stimLROnsets(j))));
    start_frame = iTime_ix - nframes_pre;
    stop_frame = iTime_ix + nframes_post;
    if stop_frame > length(iTime)
        break
    end
    tstamps = iTime(start_frame:stop_frame);
    rel_tstamps = iTime(start_frame:stop_frame) - iTime(iTime_ix);
    fvalues = tc(:, iTime_ix - nframes_pre:iTime_ix + nframes_post);
    timecoursemat_lr(:, :, j) = fvalues;
end
%compute response to LR sweeps
for j = 1:length(stimRLOnsets)
    relTimeI = iTime - bTime(stimRLOnsets(j));
    [c iTime_ix] = min(abs(iTime - bTime(stimRLOnsets(j))));
    start_frame = iTime_ix - nframes_pre;
    stop_frame = iTime_ix + nframes_post;
    if stop_frame > length(iTime)
        break
    end
    tstamps = iTime(start_frame:stop_frame);
    fvalues = tc(:, iTime_ix - nframes_pre:iTime_ix + nframes_post);
    timecoursemat_rl(:, :, j) = fvalues;
end

mean_across_trials_lr = mean(timecoursemat_lr, 3);
mean_across_trials_rl = mean(timecoursemat_rl, 3);
figure;
subplot(121);
imagesc(mean_across_trials_lr);
title('Left to right');
subplot(122);
imagesc(mean_across_trials_rl);
title('Right to Left')


%%
clear ix_of_max_val
for i=1:n_rois
    tcourse = mean_across_trials_lr(i, :);
    [argvalue, argmax] = max(tcourse);
    ix_of_max_val(i, 1) = i;
    ix_of_max_val(i, 2) = argmax;
end

[~, sorted_ixs] = sort(ix_of_max_val(:,2));

figure;
subplot(121);
imagesc(mean_across_trials_lr(sorted_ixs, :));
title('Left to right');
subplot(122);
imagesc(mean_across_trials_rl);
title('Right to Left')

%%

% Get centroid of coordinates
clear CoM
for i=1:n_rois
    %meanx = mean(Coor{i}(1, :));
    %meany = mean(Coor{i}(2, :));
    %json_file(i).centroid;
    CoM(i, :) = json_file(i).centroid; %round(meanx), round(meany)];
end
%%
figure;

imagesc(Cn); %rgb2gray(Cn));
axis equal;
hold on

c = linspace(1,10,length(CoM));
scatter(CoM(sorted_ixs, 2), CoM(sorted_ixs, 1), [], c, 'filled');
colorbar
colormap jet
% 
% for i=1:n_rois
%     roi = sorted_ixs(i);
%     %plot(CoM(roi, 2), CoM(roi, 1), 'o', 
%     hold on;
% end

%%
mean_Y = mean(Y, 3);
std_Y = std(Y, 0, 3);

%%
plotdata = [];

plotdata.name = videoName;

plotdata.name = videoName;
plotdata.time_bins = timeBins; 
plotdata.neural_timecourse = tc; 
plotdata.A_or = A_or(:,keep); 
plotdata.Cn = Cn; 
plotdata.Y_mean = mean_Y;
plotdata.Y_std = std_Y;
plotdata.options = options; 
plotdata.selIndex = selectivityIndex; 
plotdata.evoked_RL = EVOO_RL; % check the direction
plotdata.evoked_LR = EVOO_LR; 


plotdata.stimLROnsets = stimLROnsets;
plotdata.stimRLOnsets = stimRLOnsets;
plotdata.iTime = iTime;
plotdata.bTime = bTime;
plotdata.mean_across_trials_lr = mean_across_trials_lr;
plotdata.mean_across_trials_rl = mean_across_trials_rl;
plotdata.timecoursemat_lr = timecoursemat_lr;
plotdata.timecoursemat_rl = timecoursemat_rl;
plotdata.roi_sorted_by_response_time = sorted_ixs;
plotdata.roi_by_argmax = ix_of_max_val;

plotdata.CoM = CoM;
plotdata.n_trials = n_trials;
plotdata.n_rois = n_rois;

plotdata.rel_tstamps = rel_tstamps;


%plot_outfile = fullfile(savedir, 'plotvars.mat');
outdir = fullfile(basedir, session, 'processed', 'matlab-files');
plot_outfile = fullfile(outdir, sprintf('%s.mat', virmen_fname));
if ~exist(outdir, 'dir')
   mkdir(outdir)
end

save(plot_outfile, '-v7.3', 'plotdata'); % easier for simple struct, can use mat73 to load dict
%save(plot_outfile, '-struct', 'plotdata'); 

fprintf('Saved to:\n %s\n', plot_outfile)

%%
nnY = quantile(Y(:),0.0005);
mmY = quantile(Y(:),0.995);
figure;
for t = 500:1:T
    imagesc(Y(:,:,t),[nnY,mmY]); xlabel('raw data','fontsize',14,'fontweight','bold'); axis equal; axis tight;
    title(sprintf('Frame %i out of %i',t,T),'fontweight','bold','fontsize',14); colormap('bone')
    set(gca,'XTick',[],'YTick',[]);
    %subplot(122);imagesc(M2(:,:,t),[nnY,mmY]); xlabel('non-rigid corrected','fontsize',14,'fontweight','bold'); axis equal; axis tight;
    %title(sprintf('Frame %i out of %i',t,T),'fontweight','bold','fontsize',14); colormap('bone')
    %set(gca,'XTick',[],'YTick',[]);
    drawnow;
    pause(0.02);
end

%%
nnY = min(Y(:)); %quantile(Y(:),0.0001);
mmY = quantile(Y(:),0.999); %max(Y(:)); %quantile(Y(:),0.995);

 [y, x] = ndgrid(1:256);
 mov_outfpath = fullfile(outdir, 'testmovie.mp4');

 figure;
 vidfile = VideoWriter(mov_outfpath,'MPEG-4');
 open(vidfile);

 clear F
 frames_to_plot = [600:1:round(T/2)];
 for i = 1:length(frames_to_plot)
    t = frames_to_plot(i);
    im = imagesc(Y(:,:,t), [nnY,mmY]);
    xlabel('raw data','fontsize',14,'fontweight','bold'); 
    axis equal; axis tight;
    colormap(gray)
    title(sprintf('Frame %i out of %i',t,T),'fontweight','bold','fontsize',14); colormap('bone')
    set(gca,'XTick',[],'YTick',[]);
    %subplot(122);imagesc(M2(:,:,t),[nnY,mmY]); xlabel('non-rigid corrected','fontsize',14,'fontweight','bold'); axis equal; axis tight;
    %title(sprintf('Frame %i out of %i',t,T),'fontweight','bold','fontsize',14); colormap('bone')
    %set(gca,'XTick',[],'YTick',[]);
    drawnow;
    %F(t) = getframe(gcf).cdata;
    thisFrame = getframe(gca);
    F(i) = thisFrame;
    pause(neural_t/2); %(0.02);
 end
 writeVideo(vidfile, F);
 % for ind = 1:256
 %    z = sin(x*2*pi/ind)+cos(y*2*pi/ind);
 %    im = sc(z, 'hot');
 %    writeVideo(vidfile, im);
 % end
close(vidfile)

%%

plotdata.CoM = CoM;
plotdata.n_trials = n_trials;
plotdata.n_rois = n_rois;

%plot_outfile = fullfile(savedir, 'plotvars.mat');
outdir = fullfile(basedir, session, 'processed', 'matlab-files');
sprintf('virft_%s', virmen_fname);

expr_outfile = fullfile(outdir, sprintf('%s.mat', virmen_fname));
if ~exist(outdir, 'dir')
   mkdir(outdir)
end

save(expr_outfile, '-v7.3', 'expr'); 
