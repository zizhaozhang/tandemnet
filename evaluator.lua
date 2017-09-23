local c = require 'trepl.colorize'
require 'image'
require 'optim'
require 'gnuplot'
require 'utils.ConfusionMatrix'
local tablex = require 'pl.tablex'
local matio = require 'matio'
local utils = require 'utils.utils'

local Evaluate = {}
Evaluate.__index = Evaluate
function Evaluate.new()
    local sl = {}
    setmetatable(sl, Evaluate)
    sl.tab_avgloss = 0
    sl.dense_trainloss = {}
    sl.tab_accum_trainloss = 0
    sl.tab_trainacc_history = {}
    sl.tab_trainloss_history = {}
    sl.tab_testacc_history = {}
    sl.tab_testacc_notext_history = {}

    sl.rnn_lr_history = {}
    sl.cnn_lr_history = {}
    sl.dense_epoch_history = {} 
    sl.epoch_history = {}
    if opt.display == 'true' then
        disp = require 'display'
        disp.url = string.format('http://%s:%d/events', opt.display_host, opt.display_port)
    end
    sl.tab_train_acc = torch.zeros(1)
    sl.tab_train_num_data = torch.zeros(1)
    return sl
end
function Evaluate:copy(evl)
    local cp = tablex.deepcopy
    self.tab_avgloss = cp(evl.tab_avgloss)
    self.dense_trainloss = cp(evl.dense_trainloss)
    self.tab_accum_trainloss = cp(evl.tab_accum_trainloss)
    self.tab_trainacc_history = cp(evl.tab_trainacc_history)
    self.tab_trainloss_history = cp(evl.tab_trainloss_history)
    self.tab_testacc_history = cp(evl.tab_testacc_history)
    self.tab_testacc_notext_history = cp(evl.tab_testacc_notext_history)

    self.rnn_lr_history = cp(evl.rnn_lr_history)
    self.cnn_lr_history = cp(evl.cnn_lr_history) 

    self.dense_epoch_history = cp(evl.dense_epoch_history)
    self.epoch_history = cp(evl.epoch_history)

    self.it = evl.it
    self.epoch = evl.epoch
    self.tab_train_loss = evl.tab_train_loss
    self.iter_epoch = evl.iter_epoch
    self.cnn_lr = evl.cnn_lr
    self.rnn_lr = evl.rnn_lr


    if opt.display == 'true' then
        disp = require 'display'
        disp.url = string.format('http://%s:%d/events', opt.display_host, opt.display_port)
    end
    print (c.red '--> '.. 'copy evaluator history')
end
function Evaluate:update(it, epoch, iter_epoch, tab_train_loss, optimState) 
    -- -- save tmp variables
    -- local it = self.it
    -- local epoch = self.epoch
    -- local tab_train_loss = self.tab_train_loss
    -- local iter_epoch = self.iter_epoch
    -- local optimState = {}
    -- optimState.cnn_lr = self.cnn_lr
    -- optimState.rnn_lr = self.rnn_lr

    -- if it ~= nil and iter_epoch ~= nil and tab_train_loss ~= nil and optimState~= nil then
    self.it = it
    self.epoch = epoch
    self.tab_train_loss = tab_train_loss
    self.iter_epoch = iter_epoch
    self.cnn_lr = optimState.cnn_lr
    self.rnn_lr = optimState.rnn_lr
    -- end

    self.tab_avgloss = (self.tab_avgloss==0 and tab_train_loss) or self.tab_avgloss
    self.tab_avgloss = self.tab_avgloss * 0.9 + tab_train_loss * 0.1
    table.insert(self.dense_trainloss, self.tab_avgloss)
    table.insert(self.dense_epoch_history, epoch)

    if opt.display == 'true' then
        local base_display_id = opt.display_id + 100
        if it % opt.denseloss_saveinterval == 1 then
            local line_dense_epoch = torch.Tensor(self.dense_epoch_history)
            local tab_label = {'epoch', 'loss'}
            local line_dense_trainloss = torch.Tensor(self.dense_trainloss)
            line_dense_epoch = line_dense_epoch:cat(line_dense_trainloss, 2)
            disp.plot(line_dense_epoch,
                    {title='training loss',
                    labels=tab_label,
                    ylabel='loss', win=base_display_id})
            base_display_id = base_display_id + 1
        end
    end
    local base_display_id = opt.display_id + 200

    self.tab_accum_trainloss = self.tab_accum_trainloss + tab_train_loss
    local test_accuracy = 0
    -- doing testing
    if (it % iter_epoch == 0 and epoch % opt.test_interval == 0) or it == total_iter or opt.only_eval > 0  -- or it%10 == 0 
    then
        opt.full_model = opt.full_model and opt.only_eval == 0
        local sufix = (opt.remove_text_feats == 0 and 'text') or 'notext'
        local base_path = string.format('%s/%s/%s_epoch_%03d%s/', opt.save_dir, opt.figure_dir,  opt.split, epoch, sufix)
        confusion = ConfusionMatrix(data_opt.num_classes, {'normal','low grade','high grade','insufficient'})
        if opt.full_model then  confusion2 = optim.ConfusionMatrix(data_opt.num_classes) end

        --if opt.visatt == 'true' then
            os.execute(string.format('rm -r %s/', base_path))
            os.execute(string.format('mkdir -p %s', base_path))
        --end
        print(c.red '--> ' .. 'start evaluation ...')
        local count = 0
        local all_pred = torch.Tensor(1, data_opt.num_classes):zero()
        local all_label = torch.Tensor(1):zero()
        -- save features for visualization before mlp
        local feat_before_mlp = torch.Tensor(1,256):zero()
        -- local feat_module = modules.classifier.model:findModules('nn.CAddTable')
        -- assert(#feat_module>1)
        -- feat_module = feat_module[#feat_module] -- last one is the one we waht
        local aii_idx = 2
        local all_img_ids = {}
        while true do
            local time = torch.Timer()
            local pred, imgatt_pred, textatt_prob, label, imgids, images, iter_print = eval()
            all_pred = all_pred:cat(pred:float(),1)
            all_label = all_label:cat(label:float(),1)
            -- feat_before_mlp = feat_before_mlp:cat(feat_module.output:float(),1)
            all_img_ids[imgids[1]] = torch.range(aii_idx,aii_idx+4)
            aii_idx = aii_idx + 5
            local pred_notext, attprob
            if opt.full_model then -- text feture is set to zero and do a forword again
                print(c.red '--> ' .. 'forward without text')
                pred_notext, attprob = unpack(modules.classifier:forward({modules.cnn.output, modules.rnn.output:clone():zero()}))
            end
            count = count + pred:size(1)
            local maxv, idx = pred:max(2)
            idx = idx:view(-1)
            local ifcorrect = ''
            assert(pred:size(1) == 5,' batch size must be 5 ')
            if idx[1] ~=  label[1] then
                ifcorrect = label[1].. ' -> '..idx[1] .. ' ' .. imgids[1]
            end
            print(c.blue 'TEST => ' .. (string.format('[%.2fs] [%d/%d] %s %s', time:time().real, count, loader:total_amount(opt.split), iter_print, ifcorrect) ))

            confusion:batchAdd(pred:float(), label)
            if opt.full_model then confusion2:batchAdd(pred_notext:float(), label) end
            if opt.visatt == 'true' then -- only ues for full model (cnn+rnn)
                local rec = {}
                local ori_img = loader:deprocess(images)
                
                local num_show = (opt.remove_text_feats > 0 and 1) or #imgids
                for b = 1, num_show do
                    rec[imgids] = (rec[imgids] or 0)
                    rec[imgids] = rec[imgids] + 1
                    local att_result = imgatt_pred[b]:float()
                    local score = att_result:sum()
                    local img = ori_img[b]
                    att_result = att_result:div(att_result:max())
                    att_result = image.scale(att_result, data_opt.imageSize, data_opt.imageSize)
                    att_result = att_result:view(1,data_opt.imageSize, data_opt.imageSize):expandAs(img)
                    local sattname = string.format('%s_%d(%.2f).png', imgids[b], rec[imgids], score)
                    sattname = ifcorrect .. sattname
                    local sattpath = paths.concat(base_path, sattname)
                    image.save(sattpath, 0.6*att_result + 0.4*(img/255))
                    -- extra outputs when evaluation
                    if opt.only_eval > 0 then
                        local sattpath = paths.concat(base_path, string.format('%s_%d(%.3f).json', imgids[b], rec[imgids], torch.Tensor(textatt_prob[b][2]):sum()))
                        utils.write_json(sattpath, textatt_prob[b])
                        matio.save(paths.concat(base_path, string.format('%s_%d.mat', imgids[b], rec[imgids])), imgatt_pred[b]:float())
                    end
                end
            end            
            if count >= loader:total_amount(opt.split) then     
                loader:clearData(opt.split) -- clear up the memory
                break 
            end 
        end -- end of testing iter
        print (c.red '--> '..(string.format('%d of data are evaluated', count)))
        print (c.red '--> ', confusion)
        -- print (all_img_ids)
        local result_pre_path = string.format('%s/%s/%s_epoch_%03d%s_all_prediction.mat', opt.save_dir, opt.results_dir, opt.split, epoch, sufix)
        matio.save(result_pre_path, {pred=all_pred, label=all_label, feat_before_mlp=feat_before_mlp,all_img_ids=all_img_ids})
        local result_cm_path = string.format('%s/%s/%s_epoch_%03d%s.png', opt.save_dir, opt.results_dir, opt.split, epoch, sufix)
        image.save(result_cm_path, confusion:render())
        if opt.only_eval > 0 then return end
        -- get training accuacy
        self.tab_train_acc:cdiv(self.tab_train_num_data)
        self.tab_accum_trainloss = self.tab_accum_trainloss / iter_epoch 
        table.insert(self.tab_trainacc_history, self.tab_train_acc[1])
        table.insert(self.tab_trainloss_history, self.tab_accum_trainloss)
        -- get test accuracy
        confusion:updateValids()
        test_accuracy = confusion.totalValid
        table.insert(self.tab_testacc_history, test_accuracy)
        if opt.full_model then
            confusion2:updateValids()
            table.insert(self.tab_testacc_notext_history, confusion2.totalValid)
        end
        
        table.insert(self.epoch_history, epoch)
        --- get learning rate history
        table.insert(self.rnn_lr_history, optimState.rnn_lr)
        table.insert(self.cnn_lr_history, optimState.cnn_lr)

        -- serializing data for plot
        local line_epoch = torch.Tensor(self.epoch_history)                                               
        local line_trainacc = torch.Tensor(self.tab_trainacc_history)
        local line_testacc = torch.Tensor(self.tab_testacc_history)     
        -- plot to disk
        local fname_accplot = paths.concat(string.format('%s/%s',opt.save_dir,opt.graph_dir), 'train_accuracy_curve.png')
        gnuplot.pngfigure(fname_accplot)
        gnuplot.plot({'train', line_epoch, line_trainacc},
                     {'test', line_epoch, line_testacc})
        gnuplot.xlabel('epoch')
        gnuplot.ylabel('accuracy')
        gnuplot.movelegend('right','bottom')
        gnuplot.title('training & test accuracy')
        gnuplot.plotflush()
        -- plot to web
        if opt.display == 'true' then
            disp.plot(line_epoch:cat(line_trainacc,2):cat(line_testacc,2),
                        {title='train / test accuracy',
                        labels={'epoch', 'train', 'test'},
                        ylabel='accuracy', win=base_display_id})
            base_display_id = base_display_id + 1
        end

        -- clear up variables
        self.tab_train_num_data:zero()
        self.tab_train_acc:zero()
        self.tab_accum_trainloss = 0

        collectgarbage()
    end -- end of testing 
    return test_accuracy
end

return Evaluate.new()