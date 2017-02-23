%处理数据，使得符合网络数据需要
isRotated = true;
valRate = 0.2;

alldata = load('allimgs-new-2w.mat');
labels = single(alldata.tImages(2,:)) + 1;
data = alldata.xImg_matrix;

% 随机打乱数据
randIndex = single(randperm(size(alldata.xImg_matrix,4)));
data = data(:,:,:,randIndex);
labels = labels(:,randIndex);


if isRotated 
    index1111 = single(find(labels==1));
    index2222 = single(find(labels==2));

    rotatedImgCount = size(find(labels==2),2)*3;
%调整数据的标签，labels
    labels = [labels 2*ones(1,rotatedImgCount)];    
    labels = [labels ones(1,rotatedImgCount)];   
end

maxIndex = size(data, 4);
foldSize = floor(maxIndex / 10);
setCell = {};


for k = 1 : 10
	% train - 1, test - 2, validation - 3
	% test
    disp(strcat('building fold ',num2str(k),'....'));
	set = uint8(ones(1,size(data, 4)));
	if (k <10) 
		testIndex = [(k-1)*foldSize + 1 : k*foldSize];
	else 
		testIndex = [(k-1)*foldSize + 1 : maxIndex];
	end
	
	set(1,testIndex) = 2;
	
	%train and val
    trainIndex = find(set==1);
	
	% the first valRate of train data is used for evaluation
	valIndex = trainIndex(1,1:floor(valRate*size(trainIndex,2)));
	set(1,valIndex) = 3;
	
    setCell{k} = set;
end

if isRotated
    for i = 1 : size(index2222,2)         
        tempImg = rot90(data(:,:,:,index2222(1,i)));
        data(:,:,:,end+1) = tempImg;
        tempImg = rot90(tempImg);
        data(:,:,:,end+1) = tempImg;
        tempImg = rot90(tempImg);
        data(:,:,:,end+1) = tempImg;
    end

    for i = 1 : size(index1111,2)
        tempImg = rot90(data(:,:,:,index1111(1,i)));
        data(:,:,:,end+1) = tempImg;
        tempImg = rot90(tempImg);
        data(:,:,:,end+1) = tempImg;
        tempImg = rot90(tempImg);
        data(:,:,:,end+1) = tempImg;
    end

    for k = 1 : 10 
        curSet = setCell{k};
        for i = 1 : size(index2222,2)
            setTag = curSet(1, index2222(1,i));
            % add three tags to the current set
            curSet = [curSet setTag*uint8(ones(1,3))];
        end
        for i = 1 : size(index1111,2)
            setTag = curSet(1, index1111(1,i));
            % add three tags to the current set
            curSet = [curSet setTag*uint8(ones(1,3))];
        end
        setCell{k} = curSet;
    end
end

% building imdb
imdb = struct();

%Normalization
z = reshape(data,[],size(data,4)) ;
z = bsxfun(@minus, z, mean(z,1)) ;
n = std(z,0,1) ;
z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
data = reshape(z, 44, 44, 3, []) ;

%白化
z = reshape(data,[],size(data,4)) ;
W = z*z'/size(data,4) ;
[V,D] = eig(W) ;

% the scale is selected to approximately preserve the norm of W
d2 = diag(D) ;
en = sqrt(mean(d2)) ;
z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
data = reshape(z, 44, 44, 3, []) ;

randIndex = randperm(size(data,4));
data = data(:,:,:,randIndex);
labels = labels(:,randIndex);

for i = 1 : 10
    setCell{i} = setCell{i}(1,randIndex);
end

imdb.images.set = setCell;
imdb.images.data = data;
imdb.images.labels = labels;
imdb.meta.sets = {'train', 'test', 'val'} ;
imdb.meta.classes = {'Uninfected','Parasitemic'}';
if isRotated 
    save('C:\Users\Jianlang\Documents\dl data\data\imdb-new-11w-rotated.mat','-struct', 'imdb','-v7.3') ;
else
    save('C:\Users\Jianlang\Documents\dl data\data\imdb-new-2w-unrotated.mat','-struct', 'imdb','-v7.3') ;
end


