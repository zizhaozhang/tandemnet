require 'torch'
require 'nn'
require 'optim'
require 'gnuplot'

local tablex = require 'pl.tablex'
local json = require 'cjson'
local c = require 'trepl.colorize'
local net_utils = require 'utils.net_utils'
require 'utils.optim_updates'
require 'utils/DataLoader'
require 'modules.ReportModel'
require 'modules.MultiModel'
torch.setnumthreads(2)
opt, data_opt = unpack(dofile('opts.lua'))
if opt.only_eval == 0 then
    cmd:log(string.format('%s/%s/log_cmdline', opt.save_dir, opt.log_dir), opt)
end
-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
print(c.blue '==> '..'configuring data loader')
loader = DataLoader(data_opt)
print (c.blue '==> ' .. 'number training data loaded: ', loader:total_amount('train'))
-------------------------------------------------------------------------------
-- Load initial network
-------------------------------------------------------------------------------
modules = {}
local rnn_opt = net_utils.gen_rnn_state(opt, loader)
modules.rnn = nn.ReportModel(rnn_opt)

-- last, might copy weight or create clones
if string.len(opt.load_model_from) > 0 then
    snapshot = net_utils.load_checkpoint(opt.load_model_from)
    print (c.red '--> ' .. 'copy cnn weights from checkpoint')
    modules.cnn = snapshot.model.cnn:clone()
    modules.classifier = snapshot.model.cls:clone()
    if opt.use_history_opt then
        print (c.red '--> ' .. 'use history opt')
        opt.rnn_learning_rate = snapshot.optimState.rnn_lr
        opt.cnn_learning_rate = snapshot.optimState.cnn_lr
    end
    if opt.gpu_mode then for k,v in pairs(modules) do v:cuda() end end
    rnn_params, rnn_grad_params = modules.rnn:getParameters()
    print (c.red '--> ' .. 'copy rnn weights from checkpoint')
    rnn_params:copy(snapshot.model.rnn_params)
    cnn_params, cnn_grad_params = modules.cnn:getParameters()
    cls_params, cls_grad_params = modules.classifier:getParameters()
else
    if string.len(opt.transfer_from) > 0 then
        modules.cnn = net_utils.transfer_from('experiment/pretrained_cnn/'..opt.transfer_from..'.t7', 5)
    else
        modules.cnn = require('modules.wide-resnet')({depth=opt.depth, widen_factor=opt.widen_factor, dropout=0.3})
    end
    modules.classifier = nn.MultiModel(opt, data_opt.num_classes)
    -- first, ship everything to GPU
    if opt.gpu_mode then for k,v in pairs(modules) do v:cuda() end end
    -- second, coolect params space
    rnn_params, rnn_grad_params = modules.rnn:getParameters()
    cnn_params, cnn_grad_params = modules.cnn:getParameters()
    cls_params, cls_grad_params = modules.classifier:getParameters()
    print(c.blue 'total number of parameters in language model: ', rnn_params:nElement())
    print(c.blue 'total number of parameters in CNN: ', cnn_params:nElement())
    print(c.blue 'total number of parameters in multimodel classifier: ', cls_params:nElement())
end

-- build criterion
criterion = nn.CrossEntropyCriterion():cuda()
-------------------------------------------------------------------------------
-- Define evaluation module 
-------------------------------------------------------------------------------

local function cuda_batch(batch)
    --   What a batch contains
    --     .images [B, 3, Width, Height]: Training images
    --     .labels [V, B]: Text labels. Each entry is a value between [0, vocabulary-size] wehre 0 is END_TOKEN
    --     .binary_label: [V] Image labels need to predict.
    --     .vocab.word_to_idx['.']: the label of the period character (.) in the range of [0, vocublary_size]. 
    --                              Since in the paper, we assume N types of feature sentences. So there will be N '.' in each column 
    --                              of .labels. 
    --     B is batch size V is vocabulary size
  batch.images = batch.images:cuda()
  batch.cls_label = batch.binary_label:cuda()
  batch.seq_len = torch.gt(batch.labels,0):sum(1):view(-1):cuda()
  local pid = loader.vocab.word_to_idx['.']
  batch.text_feat_loc = {}
  for k = 1, batch.labels:size(2) do
    local tmp = batch.labels[{{},k}]
    local p = tmp:eq(pid):view(-1)
    local pos = torch.nonzero(p):view(-1)
    batch.text_feat_loc[k] = torch.totable(pos)
  end
  batch.labels = batch.labels:cuda()
end

function split_attention(attprob, batch)
     -- image attention
    local height = math.sqrt(opt.conv_feat_num)
     imgattprob = attprob[{{},{1,opt.conv_feat_num}}]:contiguous():view(attprob:size(1),height,height)
     local iter_print =  string.format('visual att ratio = %f, language att ratio = %f',imgattprob:sum()/attprob:size(1),  attprob[{{},{opt.conv_feat_num+1, attprob:size(2)}}]:sum()/attprob:size(1) )
     qattprob = attprob[{{},{opt.conv_feat_num+1,attprob:size(2)}}]
     -- test attention
     local textatt = {}
     for i, v in pairs(batch.texts) do
        textatt[i] = {[1]=v, [2]=torch.totable(qattprob[i]:squeeze())}
     end
     return imgattprob, textatt, iter_print
end
function eval()

     for k,v in pairs(modules) do v:evaluate() end
     local batch = loader:getBatch{split=opt.split, test_batch_size=opt.test_batch_size}
     cuda_batch(batch)
     -- forward
     local conv_feats = modules.cnn:forward(batch.images)
     local text_feats = modules.rnn:forward({torch.Tensor():cuda(), batch.labels, batch.seq_len, tablex.deepcopy(batch.text_feat_loc)})
     if opt.remove_text_feats > 0 then
        text_feats:zero()
        print(c.red '--> set text_feats to zero')
     end
     local prob, attprob = unpack(modules.classifier:forward({conv_feats, text_feats}))
     local loss = criterion:forward(prob, batch.cls_label)

     imgattprob, textatt, iter_print = split_attention(attprob, batch)
    
     return prob, imgattprob, textatt, batch.binary_label, batch.names, batch.images:float():clone(), iter_print
end
-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local function lossFun()
    for k,v in pairs(modules) do v:training() end
    rnn_grad_params:zero()
    cnn_grad_params:zero()
    cls_grad_params:zero()
    ---------------------------------------------------------------------------
    -- Forward pass
    -----------------------------------------------------------------------------
    local batch = loader:getBatch{split='train'}
    cuda_batch(batch)
    local conv_feats = modules.cnn:forward(batch.images)
    local text_feats = modules.rnn:forward({torch.Tensor():cuda(), batch.labels, batch.seq_len, tablex.deepcopy(batch.text_feat_loc)})
    local prob, attprob = unpack(modules.classifier:forward({conv_feats, text_feats}))
    local loss = criterion:forward(prob, batch.cls_label)
    -----------------------------------------------------------------------------
    -- Backward pass
    -----------------------------------------------------------------------------
    local dprob = criterion:backward(prob, batch.cls_label)
    local dconv_feats, dtext_feats = unpack(modules.classifier:backward({conv_feats, text_feats}, {dprob, attprob:clone():zero()}))
    modules.rnn:backward({torch.Tensor():cuda(), batch.labels, batch.seq_len,tablex.deepcopy(batch.text_feat_loc)}, dtext_feats)
    modules.cnn:backward(batch.images, dconv_feats)
    ------------------ gradient clipping ---------------
    local iter_print = ''
    local grad_norm
    grad_norm = rnn_grad_params:norm()
    if grad_norm > opt.grad_clip then
        rnn_grad_params:mul(opt.grad_clip / grad_norm)
        iter_print = iter_print .. string.format('\t - rnn grad clipped norm: [%f -> %f]\n', grad_norm, opt.grad_clip)
    else
        iter_print = iter_print .. string.format('\t - rnn grad is not clipped norm: %f\n', grad_norm)
    end
    grad_norm = cls_grad_params:norm()
    if grad_norm > opt.grad_clip then
        cls_grad_params:mul(opt.grad_clip / grad_norm)
        iter_print = iter_print .. string.format('\t - cls grad clipped norm: [%f -> %f]\n', grad_norm, opt.grad_clip)
    else
        iter_print = iter_print .. string.format('\t - cls grad is not clipped norm: %f\n', grad_norm)
    end
    iter_print = iter_print .. string.format('\t - cnn grad is not clipped norm: %f\n', cnn_grad_params:norm())

    -- apply L2 regularization
    if opt.cnn_weight_decay > 0 then
        cnn_grad_params:add(opt.cnn_weight_decay, cnn_params)
    end

    -- update parameters
    sgdmom(cnn_params, cnn_grad_params, optimState.cnn_lr, opt.cnn_optim_alpha, {})
    adam(rnn_params, rnn_grad_params, optimState.rnn_lr, opt.rnn_optim_alpha, opt.rnn_optim_beta, opt.rnn_optim_epsilon, {})
    adam(cls_params, cls_grad_params, optimState.rnn_lr*opt.cls_lr_scale, opt.rnn_optim_alpha, opt.rnn_optim_beta, opt.rnn_optim_epsilon, {})

    -- calculate accuracy
    local maxv, preds = torch.max(prob, 2)
    local acc = preds:view(-1):cuda():eq(batch.cls_label):sum()

    evaluator.tab_train_acc:add(acc)
    evaluator.tab_train_num_data:add(torch.Tensor{preds:size(1)})
    
    return loss, attprob, iter_print
end
collectgarbage() -- free some memory

--------------------------------------------------------------
-- Main
--------------------------------------------------------------
evaluator = dofile'evaluator.lua'
-- if do evaluation
if opt.only_eval > 0 then
    epoch = snapshot.epoch
    print (c.red ('--> perform evaluation ... on '.. opt.split))
    evaluator:copy(snapshot.evaluator)
    -- do some dummy forward
    print (c.red '--> do some dummy forward ...')
    for iter = 1, 0 do 
        print (iter..' ')
        for k,v in pairs(modules) do v:training() end
        local batch = loader:getBatch{split='train'}
        cuda_batch(batch)
        local conv_feats = modules.cnn:forward(batch.images)
        local prob = modules.classifier:forward(conv_feats)
    end

    evaluator:update(0, epoch, 0, 0, {})
    os.exit()
end
-- otherwise, do training
-- do some initialization
optimState = {}
optimState.rnn_lr = opt.rnn_learning_rate
optimState.cnn_lr = opt.cnn_learning_rate

iter_epoch = math.floor(loader:total_amount('train') / opt.batch_size) -- iter per epoch
total_iter = opt.max_epochs * iter_epoch 
local best_acc = 0

for iter = 1, total_iter do
    local epoch = iter / iter_epoch
    local time = torch.Timer()
    local loss, attprob, iter_print = lossFun(iter)
    
    time = time:time().real

    local evl_acc = evaluator:update(iter, epoch, iter_epoch, loss, optimState)
    if iter % opt.display_interval == 0 or iter == 1 then
        print (c.blue 'Train => '..string.format('[%.2fs] iter=%05d(epoch %.3f), cnn_lr=%.5f, rnn_lr=%.5f/%.5f, loss=%.4f', 
                                                    time, iter, epoch, optimState.cnn_lr, optimState.rnn_lr, optimState.rnn_lr*opt.cls_lr_scale, loss))
        print (iter_print)                                            
    end
    -- schedule learning rate, stop answer unit
    -- exponential learning rate decay
    if iter % iter_epoch == 0 and opt.lr_decay < 1 and epoch >= opt.start_lr_decay then
        if epoch % opt.lr_decay_interval == 0 then
            optimState.rnn_lr = optimState.rnn_lr * 0.9 --opt.lr_decay -- decay it
            optimState.cnn_lr = optimState.cnn_lr * opt.lr_decay
            print('decayed learning rate by a factor ' .. opt.lr_decay)
        end
    end
    -- save checkpoint
    if iter % iter_epoch == 0 then
        local isBest = ''
        if best_acc < evl_acc then 
            best_acc = evl_acc 
            isBest = 'Best'
        end
        local savefile = string.format('%s/%s/snapshot_epoch%.2f%s.t7', opt.save_dir, opt.snapshot_dir, epoch, isBest)
        -- need to save whole cnn model because it has BN, do not wanna to lose mean and std
        local model = {cls=modules.classifier:clearState(), cnn=modules.cnn:clearState(), rnn_params=rnn_params}
        net_utils.save_checkpoint(iter, opt, epoch, optimState, model, evaluator, savefile)
        
    end
    if iter % opt.free_interval == 0 then collectgarbage() end
end
