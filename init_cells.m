function net = init_cells(varargin)

%网络参数设置
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;
disp('Calling init_cells');
lr = [.1 2] ;

%设计CNN
net.layers = {} ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.01*randn(5,5,3,16, 'single'), zeros(1, 16, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1) ;  % 40*40*16
						   
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [13 13], ...
                           'stride', 2) ;  %(40 - 13 + 1) / 2 = 14,  14*14*16
						   
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(5,5,16,32, 'single'), zeros(1,32,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1) ;   % 14-5+1 = 10,  10*10*32
						   
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [5 5], ...
                           'stride', 2) ; % (10-5+1) / 2 = 3,  3*3*32
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(3,3,32,64, 'single'), zeros(1,64,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1) ;   
                       
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(1,1,64,2, 'single'), zeros(1,2,'single')}}, ...
                           'learningRate', .1*lr, ...
                           'stride', 1) ;
                       
net.layers{end+1} = struct('type', 'relu') ;

% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

% hyperparameters 参数
net.meta.inputSize = [44 44 3] ;
net.meta.trainOpts.learningRate = [0.05*ones(1,22) 0.03*ones(1,5) 0.02*ones(1,4)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 100 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% Fill in default values
net = vl_simplenn_tidy(net) ;

end

