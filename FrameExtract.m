function FrameExtract()

setup ;

net = load('data/imagenet-vgg-verydeep-16.mat') ;
vl_simplenn_display(net) ;

VideoName = '4.avi';
a=VideoReader(VideoName);
info=get(a);
h=info.Height;
w=info.Width;
%fr=floor(info.FrameRate)
fr = 10;
size = floor(a.NumberOfFrames/fr);
C=cell(2,size);
ind=0;
dirName=strcat(VideoName,'Frames');
mkdir (dirName);


for img = 1:fr:a.NumberOfFrames;
    ind=ind+1;
    filename=strcat(dirName,'/frame',num2str(ind),'.jpg');
    b = read(a, img);
    
    %histogram
    hist = [imhist(b(:,:,1)); imhist(b(:,:,2)); imhist(b(:,:,3))];
    hist = hist/(h*w);
    C{1,ind}=hist;
    
    imwrite(b,filename);
    im_ = single(b) ; % note: 255 range
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
    im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;
    
    % run the CNN
    res = vl_simplenn(net, im_) ;

    % show the classification result
    f34 = squeeze(gather(res(34).x));
    C{2,ind}=f34;
    
end
total=ind
ind=0;
edges = zeros(total-1,1);

%ALPHA
alpha=0.1;

for ind = 1:size;
    edgew=0;
    euch=0;
    for i=1:768;
        euch=euch+(C{1,ind}(i,1)-C{1,ind+1}(i,1))*(C{1,ind}(i,1)-C{1,ind+1}(i,1));    
    end
    euch=power(euch,0.5);
    edgew=edgew+euch*alpha;
    
    eucf=0;
    for i=1:4096;
        eucf=eucf+(C{2,ind}(i,1)-C{2,ind+1}(i,1))*(C{2,ind}(i,1)-C{2,ind+1}(i,1));
    end
    eucf=power(eucf,0.5);
    edgew=edgew+eucf*(1-alpha);
    edges(ind,1)=edgew;
end

%Clustering
noOfFrames = size;
current_value = edges(1);
clusterBoundaries = zeros(noOfFrames, 1);

clusterBoundaries(1) = 1;

clusterCount = 1;
for i = 2:noOfFrames;
    if(edges(i) - current_value > 10)
        clusterCount = clusterCount+1;
        clusterBoundaries(clusterCount) = i;
        %cluster ends at i
    end
    current_value = edges(i);
end

clusterCount = clusterCount+1;
clusterBoundaries(clusterCount) = noOfFrames;       


%keyframe selection
average=cell(1,clusterCount);
absdiff=cell(1,noOfFrames);
keycell=cell(1,clusterCount);
initialBoundary=clusterBoundaries(1,1);
nextBoundary=clusterBoundaries(2,1);
index=1;
keyframes = zeros(clusterCount,1);

%initializing averageCell
for clustind = 1:clusterCount
    average{1,clustind} = zeros(768,1);
    average{2,clustind} = zeros(4096,1);
end


for clustind=1:clusterCount
    minMean=[10000 10000];
    
    for i = 1:2
        for k=initialBoundary:nextBoundary
            average{i,index}=average{i,index}+C{i,k};
            if k==nextBoundary
                average{i,index} = average{i,index}/(nextBoundary-initialBoundary+1);
            end
        end
    end
    
    tempMean=[0 0];
    for k=initialBoundary:nextBoundary
        for i = 1:2
            absdiff{i,k}=abs(average{i,index}-C{i,k});
            tempMean(i) = mean(absdiff{i,k});
        end
        if((tempMean(1)*alpha + tempMean(2)*(1-alpha)) < minMean) 
                minMean = tempMean;
                keyframes(index,1) = k;
        end
    end
    
    index=index+1;
    initialBoundary=nextBoundary+1;
    nextBoundary=clusterBoundaries(clustind+2,1);
end

%displaying frames
displayRows = 4;
%displayRows = floor(clusterCount/displayCols) + 1;
displayCols = floor(clusterCount/displayRows) + 1;

% for i=1:displayRows
%     for j=1:displayCols
%        filename=strcat(dirName,'/frame',num2str(keyframes(clustind,1)),'.jpg');
%        if exist(filename, 'file') == 2
%            %subplot(displayRows,displayCols,clustind);
%            [ha, pos] = tight_subplot(displayRows,displayCols,0.01,0.01,0.01);
%            for ii = 1:clusterCount; 
%                axes(ha(ii)); 
%                axesimshow(filename);
%            end
%            set(ha(1:4),'XTickLabel',''); set(ha,'YTickLabel','');
%            imshow(filename);
%            title(num2str(keyframes(clustind,1)));
%            clustind = clustind+1;
%        end
%     end
% end
[ha, pos] = tight_subplot(displayRows,displayCols,0.01,0.01,0.01);
           for ii = 1:clusterCount;
               filename=strcat(dirName,'/frame',num2str(keyframes(ii,1)),'.jpg');
               if exist(filename, 'file') == 2
                    axes(ha(ii)); 
                    imshow(filename);
               end
           end
saveas(gcf,strcat(dirName,'/Output.jpg'));
end