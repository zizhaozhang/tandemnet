local c = require 'trepl.colorize'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')

-- Model Setting
cmd:option('-load_model_from', '', 'path to a model checkpoint to initialize model weights from.')
cmd:option('-rnn_size', 256, 'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size', 128, 'the encoding size of each token in the vocabulary and image fc7.')

-- Optimization: General
-- cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size', 64, 'what is the batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-grad_clip', 0.1, 'clip gradients at this value (note should be lower than usual 5 because we normalize grads by both batch and seq_length)')
cmd:option('-drop_prob_lm', 0.4, 'strength of dropout in the Language Model RNN')
cmd:option('-max_epochs', 20, 'maxinum number of epochs')

-- Optimization: for the RNN
cmd:option('-rnn_optim', 'adam', 'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-rnn_learning_rate', 1e-4, 'learning rate')
cmd:option('-rnn_optim_alpha', 0.8, 'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-rnn_optim_beta', 0.999, 'beta used for adam')
cmd:option('-rnn_optim_epsilon', 1e-8, 'epsilon that goes into denominator for smoothing')
cmd:option('-lr_decay',0.9,'learning rate decay, if you dont want to decay learning rate, set 1')
cmd:option('-start_lr_decay', 1, 'learning rate decay interval in epoch')
cmd:option('-lr_decay_interval', 1, 'learning rate decay interval in epoch')

-- Optimization: for the CNN
cmd:option('-cnn_optim', 'sgdmom', 'optimization to use for CNN')
cmd:option('-cnn_optim_alpha', 0.9, 'alpha for momentum of CNN')
cmd:option('-cnn_optim_beta', 0.999, 'alpha for momentum of CNN') -- using sgdmom for cnn
cmd:option('-cnn_learning_rate', 1e-2, 'learning rate for the CNN')
cmd:option('-cnn_weight_decay', 0.0005, 'L2 weight decay just for the CNN')
cmd:option('-conv_feat_size', 256, 'convolutional feature map channels')
cmd:option('-conv_feat_num', 196, 'convolutional feature length') -- 14*14 feature map
cmd:option('-depth', 16, 'resnet depth') -- 6x6 feature map
cmd:option('-widen_factor', 4, 'resnet depth') -- 4 -> 256 conv_feat_size
cmd:option('-transfer_from', '', 'transfer from pretrained resnet') 
cmd:option('-conv_map_width', 14, 'convolutional feature length') -- 14*14 feature map
cmd:option('-cls_lr_scale', 1, 'convolutional feature length') -- 14*14 feature map

-- For multi model
cmd:option('-deathRate', 0.5, 'deathRate of language part') -- 14*14 feature map
cmd:option('-drop_prob_mm', 0.5, 'strength of dropout in the Language Model RNN')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-suffix', '', 'suffix that is appended to the model file paths')
cmd:option('-seed', 1234, 'random number generator seed to use')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-split', 'val', 'split to use: val|test|train')
cmd:option('-fold', 0, 'which train/val fold are working on')
cmd:option('-free_interval', 10, 'which train/val fold are working on')
cmd:option('-save_dir', 'checkpoints/first_exp', 'which train/val fold are working on')
cmd:option('-use_history_opt', 1, 'if use history opt (learning rate mostly)')
cmd:option('-only_eval', 0, 'if use history opt (learning rate mostly)')


cmd:text('VISUALIZATION')
cmd:option('-test_interval', 1, 'interval of testing in epoch')
cmd:option('-denseloss_saveinterval', 50, 'interval for saving dense training loss in iteration')
cmd:option('-visatt', 'false', 'whether visualize attention or not')

cmd:text('DISPLAY')
cmd:option('-display', 'true', 'display result while training')
cmd:option('-display_id', 10, 'display window id')
cmd:option('-display_host', 'localhost', 'display hostname 0.0.0.0')
cmd:option('-display_port', 8888, 'display port number')
cmd:option('-display_interval', 10, 'display interval')

-- for test net
cmd:option('-test_small_set', 0, 'best test model')
cmd:option('-test_batch_size', 1, 'test batch size') -- must equal to one
cmd:option('-remove_text_feats', 0, ' remove text features when doing evaluation (set to zero)')



opt = cmd:parse(arg)
opt = xlua.envparams(opt) -- merge from sh file

data_opt = {
    dataset_dir = '',
    batch_size = 32
    -- TODO add more based on your needs
    
}

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
opt.gpu_mode = opt.gpuid >= 0
if opt.gpu_mode then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then 
    require 'cudnn' 
    cudnn.fastest = true
	cudnn.benchmark = true
  end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid)
end
     
-- constant options
opt.save_dir = string.format('%s', opt.save_dir) 
opt.log_dir = 'training_log'
opt.snapshot_dir = 'snapshot'
opt.results_dir = 'results'
opt.graph_dir = 'graphs'
opt.figure_dir = 'figures'

if string.len(opt.save_dir) > 0 then
    print (c.red '--> '..string.format('mkdir -p %s/%s', opt.save_dir, opt.results_dir))
    -- create directory and log file
    print (c.red '--> ' .. 'create save folder '..opt.save_dir)
    os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.log_dir))
    os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.snapshot_dir))
    os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.results_dir))
    os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.graph_dir))
    os.execute(string.format('mkdir -p %s/%s', opt.save_dir, opt.figure_dir))
end

if opt.split == 'test' then opt.test_batch_size = 1 end

return {opt, data_opt}