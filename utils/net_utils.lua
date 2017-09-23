local utils = require 'utils.utils'
local net_utils = {}
local c = require 'trepl.colorize'

function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if word ~= '' then
        if j >= 2 then txt = txt .. ' ' end
        txt = txt .. word
      end
    end
    table.insert(out, txt)
  end
  return out
end

-- hiding this piece of code on the bottom of the file, in hopes that
-- noone will ever find it. Lets just pretend it doesn't exist
function net_utils.language_eval(predictions, id, path)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local path = path or 'evaluation/'
  local out_struct = predictions 
  utils.write_json(path .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  -- print (c.red 'save evaluation results at '..path .. id .. '.json')
  os.execute('./misc/call_python_caption_eval.sh ' .. '../' .. path ..id .. '.json') -- i'm dying over here
  -- print (c.red 'save evaluation results at '.. path ..id .. '.json')
  -- print (c.red 'read from '..path .. id .. '.json_out.json')
  local result_struct = utils.read_json(path .. id .. '.json_out.json') -- god forgive me
  return result_struct
end
function net_utils.gen_rnn_state(opt, loader)
  local rnn_opt = {}
  rnn_opt.vocab_size = loader:getVocabSize()
  rnn_opt.seq_length = loader:getSeqLength()
  rnn_opt.input_encoding_size = opt.input_encoding_size -- !!attention this have to be smaller than vocab size
  -- assert(rnn_opt.input_encoding_size <= rnn_opt.vocab_size)
  rnn_opt.rnn_size = opt.rnn_size
  rnn_opt.num_layers = 1
  rnn_opt.dropout = opt.drop_prob_lm
  -- rnn_opt.batch_size = opt.batch_size
  rnn_opt.linear_feat_length = opt.linear_feat_length
  rnn_opt.num_classes = opt.num_classes
  rnn_opt.conv_feat_layer_id = opt.conv_feat_layer_id
  rnn_opt.conv_feat_size = opt.conv_feat_size 
  rnn_opt.conv_feat_num = opt.conv_feat_num
  return rnn_opt
end

function net_utils.save_checkpoint(iter, opt, epoch, optimState, model, evaluator, savefile)
  print(c.red '--> saving checkpoint to ' .. savefile)
  local checkpoint = {}
  checkpoint.iter = iter
  checkpoint.opt = opt
  checkpoint.epoch = epoch
  checkpoint.optimState = optimState
  checkpoint.modules = {}

  checkpoint.model = model 
  checkpoint.evaluator = evaluator
  torch.save(savefile, checkpoint)
  model = nil
  
end

function net_utils.load_checkpoint(init_from)
  print(c.red '--> load checkpoint at '..init_from)
  local checkpoint = torch.load(init_from)
  return checkpoint
end

function net_utils.transfer_from(transfer_from, freeze)
  print (c.red '--> '..'transfer pretrained model from '..transfer_from)
  local pretrained
  if string.find(transfer_from,'resnet') then
    pretrained = torch.load(transfer_from)
    assert(torch.type(pretrained:get(#pretrained.modules)) == 'nn.Linear')
    pretrained:remove(#pretrained.modules)
    assert(torch.type(pretrained:get(#pretrained.modules)) == 'nn.View', torch.type(pretrained:get(#pretrained.modules)))
    pretrained:remove(#pretrained.modules)
    assert(torch.type(pretrained:get(#pretrained.modules)) == 'cudnn.SpatialAveragePooling', torch.type(pretrained:get(#pretrained.modules)))
    pretrained:remove(#pretrained.modules)
    print ('--> remove classification module (averagepool, view, linear)')

    if freeze > 0 then
      local modules = pretrained.modules
      for k = 1, freeze do
        modules[k].parameters = function() return nil end 
        modules[k].accGradParameters = function() end -- overwrite this to reduce computations
      end
      print (c.red '--> '.. 'freeze modules from 1 to '..freeze)
    end

  elseif string.find(transfer_from,'vgg') or string.find(transfer_from,'alexnet') then
    pretrained = torch.load(transfer_from)
    pretrained:remove(#pretrained.modules)
    pretrained:remove(#pretrained.modules)
    print ('--> remove classification module (softmax, linear)')
  else 
    local snapshot = torch.load(transfer_from)
    pretrained = snapshot.model.cnn:clone()
    print ('--> remove nothing')
  end
  return pretrained

end


return net_utils
