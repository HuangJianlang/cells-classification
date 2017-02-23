disp('loading imdb...');
imdb = load('C:\Users\Jianlang\Documents\dl data\data\imdb-new-11w-rotated.mat');
% load trained model
disp('loading pretrained model...');

models = load ('C:\Users\Jianlang\Documents\dl data\data\Cells-14182-lenet\fold-1\net-epoch-31.mat');

batchSize = 500;

allscores = [];

for fold = 1 : 1
    for k = 1 : size(models,2)
        model = load(fullfile('data','models',models{k}));
        
        model.net = vl_simplenn_tidy(model.net);
        testSet = imdb.images.set{fold};
        testImgs = imdb.images.data(:,:,:,testSet == 2);
        testLabels = imdb.images.labels(:,testSet == 2);

        model.net.layers{end}.type = 'softmax';
        totalbatch = floor(size(testImgs,4) / batchSize);
        
        scores = [];
        testLabels_cur = [];
        for i = 1 : totalbatch
            disp(strcat('model-',num2str(k),',Evaluating batch-',num2str(i),'/',num2str(totalbatch),'.'));
            indexStart = (i - 1) * batchSize+1;
            indexEnd = i * batchSize;
            if (i == totalbatch)
                indexEnd = size(testImgs, 4);
            end
            res = vl_simplenn(model.net, testImgs(:,:,:, indexStart : indexEnd));
            testLabels_cur = [testLabels_cur testLabels(:,indexStart : indexEnd)];
            ss = squeeze(gather(res(end).x)) ;
            scores = [scores ss];
        end
        allscores(:,:,k) = scores;
    end
    finalscores = sum(allscores, 3) / size(models,2);
    dummyLabels = zeros(2, size(finalscores, 2));
    dummyLabels(1,find(testLabels_cur == 1)) = 1;
    dummyLabels(2,find(testLabels_cur == 2)) = 1;
    plotconfusion(dummyLabels, finalscores);
end


