load 'faceData.mat'
%http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
%% Initializations
ri=round(400*rand(1,1));            % Randomly pick an index.
test_img=faceData(:,ri);                          %  contains the image we later on will use to test the algorithm
train_img=double(faceData(:,[1:ri-1 ri+1:end]));           %  contains the rest of the 399 images. 

%% Get mean face
mean_face = reshape(mean(train_img,2),112,92) ;
mean_column = reshape(mean_face,[10304,1]);
figure(1)
imagesc(mean_face) ;
colormap(gray) ;   

%% Calculating the deviation of each image from mean image
A = [];  
for i = 1 : size(train_img,2)
    temp = double(train_img(:,i)) - mean_column; % Computing the difference image for each image in the training set Ai = Ti - m
    A = [A temp]; % Merging all centered images A:centered image vectors
end

%% PCA
x = mean(train_img,2);
C = A*A'/size(train_img,2);
L = A'*A;
[V,D] = eig(L);
[sv si] = sort(diag(D),'descend');
Vs = V(:,si);
%%  plot the percentage of total variance versus number of eigenvalues to get an idea as how PCA can "squeeze" the variance into the first few eigenvalues
 [rowDim, colDim]=size(faceData);
 meanFace=mean(double(cat(3, faceData)), 3);
 fprintf(' ===> %.2f sec\n', toc);
 fprintf('Perform PCA... '); tic
 [A2, eigVec, eigValue]=pca(A);
 fprintf(' ===> %.2f sec\n', toc);
 cumVar=cumsum(eigValue);
 cumVarPercent=cumVar/cumVar(end)*100;
 plot(cumVarPercent, '.-');
 xlabel('No. of eigenvalues');
 ylabel('Cumulated variance percentage (%)');
 title('Variance percentage vs. no. of eigenvalues');
 fprintf('Saving results into eigenFaceResult.mat...\n');
 save eigenFaceResult A2 eigVec cumVarPercent rowDim colDim
 DS.input=A2;
 DS.outputName=unique({faceData.parentDir});
 DS.output=zeros(1, size(DS.input,2));
 for i=1:length(DS.output)
 	DS.output(i)=find(strcmp(DS.outputName, faceData(i).parentDir));
  	DS.annotation{i}=faceData(i).path;
 end
myTic=tic;
maxDim=100;
rr=pcaPerfViaKnncLoo(DS, maxDim, 1);
plot(1:maxDim, cumVarPercent(1:maxDim), '.-', 1:maxDim, rr*100, '.-'); grid on
xlabel('No. of eigenfaces');
ylabel('LOO recog. rate & cumulated variance percentage');
[maxValue, maxIndex]=max(rr);
line(maxIndex, maxValue*100, 'marker', 'o', 'color', 'r');
legend('Cumulated variance percentage', 'LOO recog. rate', 'location', 'southeast');
fprintf('Optimum number of eigenvectors = %d, with recog. rate = %.2f%%\n', maxIndex, maxValue*100);
 toc(myTic)

%% Sorting and eliminating eigenvalues
L_eig_vec = [];
for i = 1 : size(V,2) 
    if( D(i,i)>1 )
        L_eig_vec = [L_eig_vec V(:,i)];
    end
end
Eigenfaces = A*L_eig_vec; 
%% Projecting centered image vectors into facespace
ProjectedImages = [];
Train_Number = size(Eigenfaces,2);
for i = 1 : Train_Number
    temp = Eigenfaces'*A(:,i); % Projection of centered images into facespace
    ProjectedImages = [ProjectedImages temp]; 
end
%% Extracting the PCA features from test image
temp = test_img(:,:,1);
[irow, icol] = size(temp);
InImage = reshape(temp',irow*icol,1);
Difference = double(InImage)-mean_column; % Centered test image
ProjectedTestImage = Eigenfaces'*Difference; % Test image feature vector
%% Calculating Euclidean distances 
Euc_dist = [];
for i = 1 : Train_Number
    q = ProjectedImages(:,i);
    temp = ( norm( ProjectedTestImage - q ) )^2;
    Euc_dist = [Euc_dist temp];
end

[Euc_dist_min , Recognized_index] = min(Euc_dist);
OutputName = strcat(int2str(Recognized_index),'.pgm');
%% display
figure(4)
subplot(1,2,1)
imshow(reshape(test_img,112,92));
title('Looking for ...','FontWeight','bold','Fontsize',16,'color','red');
subplot(122);
subplot(122);
imshow(reshape(faceData(:,Recognized_index),112,92));
title('Found!','FontWeight','bold','Fontsize',16,'color','red');

%% Error rate
% number_of_faces_recognized = 0;
% number_of_faces_presented = 0;
% recognition_rate = number_of_faces_recognized / number_of_faces_presented;
% 
%  for K = 1 : size(faceData)
%    %face_image = imread(reshape(faceData(K,1),92,112)); %of the K'th face image
%     face_image = faceData(K,1);
%    %was_it_recognized = try_to_recognize_face(faceData);
%    was_it_recognized = double(number_of_faces_presented(faceData));
%    number_of_faces_presented = number_of_faces_presented + 1;
%    number_of_faces_recognized = number_of_faces_recognized + was_it_recognized;
%  end
%  recognition_rate = number_of_faces_recognized / number_of_faces_presented;