clear

addpath(genpath('/Users/tom/Documents/MATLAB/ImRegCa'));
addpath(genpath('/Users/julianarhee/Documents/MATLAB/CVX-master'));
addpath(genpath('/Users/julianarhee/Documents/MATLAB/NoRMCorre-master'));

%%
rootdir = '/Volumes/Juliana/2p-data';
session = '20240905';

currdir = fullfile(rootdir, session, 'processed');
clear names
allFiles = dir(currdir); 
ctr = 1; 
for i = 1:length(allFiles)
    if contains(allFiles(i).name,'.tif') && ~allFiles(i).isdir...
            && ~contains(allFiles(i).name, 'STD') && ~contains(allFiles(i).name, 'Channel02')
        names{ctr} = [allFiles(i).folder '/' allFiles(i).name]; ctr = ctr+1; 
    end
end
disp('Processing Files: ')
celldisp(names)
    
%%

for i = 1:length(names)
    name = names{i};
    
    tic; Y = read_file(name); toc; % read the file 
    Y = single(Y);                 % convert to single precision
    T = size(Y,ndims(Y));
    
    % perform non-rigid motion correction
    options_nonrigid = NoRMCorreSetParms('d1',size(Y,1),'d2',size(Y,2),'grid_size',[32,32],'mot_uf',4,'bin_width',50,'max_shift',15,'max_dev',3,'us_fac',50);
    tic; [M2,shifts2,template2] = normcorre_batch(Y,options_nonrigid); toc
    
    % compute metrics
    
    nnY = quantile(Y(:),0.005);
    mmY = quantile(Y(:),0.995);
    
    [cY,mY,vY] = motion_metrics(Y,10);
    [cM2,mM2,vM2] = motion_metrics(M2,10);
    T = length(cY);
    
    % plot metrics
    figure;
    ax1 = subplot(2,2,1); imagesc(mY,[nnY,mmY]);  axis equal; axis tight; axis off; title('mean raw data','fontsize',14,'fontweight','bold')
    ax3 = subplot(2,2,2); imagesc(mM2,[nnY,mmY]); axis equal; axis tight; axis off; title('mean non-rigid corrected','fontsize',14,'fontweight','bold')
    subplot(2,2,3); plot(1:T,cY,1:T,cM2); legend('raw data','non-rigid'); title('correlation coefficients','fontsize',14,'fontweight','bold')
    subplot(2,2,4); scatter(cY,cM2); hold on; plot([0.9*min(cY),1.05*max(cM2)],[0.9*min(cY),1.05*max(cM2)],'--r'); axis square;
    xlabel('raw data','fontsize',14,'fontweight','bold'); ylabel('non-rigid corrected','fontsize',14,'fontweight','bold');
    
    % save data
    display('Saving files...');
    outFname = strrep(name,'.tif','_nrAligned.tif');
    %outPname = [pwd '/' outFname];
    outPname = [outFname];
    saveastiff(M2,outPname);
    close all; %clearvars -except names i;
end
    %% plot a movie with the results
    
    figure;
    for t = 1:1:T
        subplot(121);imagesc(Y(:,:,t),[nnY,mmY]); xlabel('raw data','fontsize',14,'fontweight','bold'); axis equal; axis tight;
        title(sprintf('Frame %i out of %i',t,T),'fontweight','bold','fontsize',14); colormap('bone')
        set(gca,'XTick',[],'YTick',[]);
        subplot(122);imagesc(M2(:,:,t),[nnY,mmY]); xlabel('non-rigid corrected','fontsize',14,'fontweight','bold'); axis equal; axis tight;
        title(sprintf('Frame %i out of %i',t,T),'fontweight','bold','fontsize',14); colormap('bone')
        set(gca,'XTick',[],'YTick',[]);
        drawnow;
        pause(0.02);
    end