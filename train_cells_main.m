function [net, info] = train_cells_main(varargins)
varargins = {};
opts.cudnnWorkspaceLimit = 1024*1024*1204;
opts.weightInitMethod = 'gaussian' ;
opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.batchNormalization = false;

opts.expDir = sprintf('.\\data\\Cells-14182-%s', opts.modelType) ;
    
%data path 
opts.imdbPath = 'C:\Users\Jianlang\Documents\dl data\data\imdb-new-11w-rotated.mat';
opts.networkType = 'simplenn' ;

opts.train = struct() ;
opts = vl_argparse(opts, varargins) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%  准备数据
% -------------------------------------------------------------------------

net = init_cells

imdb = load(opts.imdbPath) ;

net.meta.classes.name = imdb.meta.classes(:)';

% -------------------------------------------------------------------------
% training
% -------------------------------------------------------------------------

trainfn = @cnn_train ;


baseDir = opts.expDir;

setCells = imdb.images.set;


for fold = 1 : 1

	opts.expDir = strcat(baseDir,'\\fold-',num2str(fold));

	if ~exist(opts.expDir) 
        mkdir(opts.expDir);
    end
    
    imdb.images.set = setCells{fold};

	[net, info] = trainfn(net, imdb, getBatch(opts), ...
	  'expDir', opts.expDir, ...
	  net.meta.trainOpts, ...
	  opts.train, ...
	  'val', find(imdb.images.set == 3)) ;
end

imdb.images.set = setCells;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------

fn = @(x,y) getSimpleNNBatch(x,y) ;

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
disp(strcat('batch',num2str(batch)));

images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
