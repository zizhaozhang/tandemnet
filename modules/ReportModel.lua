local utils = require 'utils.utils'
local net_utils = require 'utils.net_utils'
local LSTM = require 'modules.LSTM'

-------------------------------------------------------------------------------
-- Spatial Attention Report Model
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.ReportModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size')
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.text_feat_num = utils.getopt(opt, 'text_feat_num', 5)

  -- create the core lstm network. note +1 for both the START and END tokens
  local dropout = utils.getopt(opt, 'drop_prob_lm', 0)
  self.lstm_input_size = self.input_encoding_size 
  self.core = LSTM.lstm(self.lstm_input_size, self.vocab_size+1, self.rnn_size, self.num_layers, dropout)
  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)

  -- create attention network
  local att_opt = {}
  att_opt.rnn_size = self.rnn_size

  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

function layer:createClones()
  -- construct the net clones
  print('--> constructing clones inside the model')
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table}
  for t=2,self.seq_length+2 do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
  end
end

function layer:getModulesList()
  return {self.core, self.lookup_table}
end

function layer:parameters()
  -- flatten model parameters and gradients into single vectors
  local params, grad_params = {}, {}
  for k, m in pairs(self:getModulesList()) do
    local p, g = m:parameters()
    for _, v in pairs(p) do table.insert(params, v) end
    for _, v in pairs(g) do table.insert(grad_params, v) end
  end
  -- invalidate clones as weight sharing breaks
  self.clones = nil
  -- return all parameters and gradients
  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.lookup_tables) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
end

--[[
takes an image and question, run the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence length and N is batch (so columns are sequences)
--]]
function layer:sample(fc_feats, opt)
  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  local batch_size = fc_feats:size(1)

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs -- logprobs predicted in last time step
  local fullLogProbs = torch.FloatTensor(self.seq_length, batch_size, self.vocab_size+1)


  -- initialize state
  self:_createInitState(batch_size)
  local state = self.init_state

  for t=1,self.seq_length+2 do

    -- print ('current time step', t)
    local xt, it, sampleLogprobs
    if t == 1 then
      -- feed in the images
      xt = fc_feats
    elseif t == 2 then
      -- feed in the start tokens
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      self.lookup_tables_inputs[t] = it
      xt = self.lookup_table:forward(it)
    else
      -- take predictions from previous time step and feed them in
      if sample_max == 1 then
        -- use argmax "sampling"
        sampleLogprobs, it = torch.max(logprobs, 2)
        it = it:view(-1):long()
      else
        -- sample from the distribution of previous predictions
        local prob_prev
        if temperature == 1.0 then
          prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
        else
          -- scale logprobs by temperature
          prob_prev = torch.exp(torch.div(logprobs, temperature))
        end
        it = torch.multinomial(prob_prev, 1)
        sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
        it = it:view(-1):long() -- and flatten indices for downstream processing
      end
      xt = self.lookup_table:forward(it)
    end

    if t >= 3 then
      seq[t-2] = it -- record the samples
      seqLogprobs[t-2] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
      fullLogProbs[t-2]:copy(logprobs:float()) -- and also full LogProbs
    end

   
    local inputs = {xt,unpack(state)}
    local out = self.core:forward(inputs)
    logprobs = out[self.num_state+1] -- last element is the output vector
    state = {}
    for j=1,self.num_state do table.insert(state, out[j]) end

  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs, fullLogProbs
end
function layer:sample_retrieval(fc_feats, seq_ref, opt)
  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  -- local ref_seq_num = utils.getopt(opt, 'ref_seq_num')

  local batch_size = fc_feats:size(1)

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size):zero()
  local logprobs -- logprobs predicted in last time step
  local fullLogProbs = torch.FloatTensor(self.seq_length, batch_size, self.vocab_size+1):zero()
  -- save attention
  if self.save_step_attention then 
    self.step_attention = torch.FloatTensor(self.seq_length+1, batch_size, self.conv_feat_num):zero() 
  end

  -- initialize state
  self:_createInitState(batch_size)
  local state = self.init_state

  for t=1,self.seq_length+2 do

    -- print ('current time step', t)
    local xt, it, sampleLogprobs
    if t == 1 then
      -- feed in the images
      xt = fc_feats
    elseif t == 2 then
      -- feed in the start tokens
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      self.lookup_tables_inputs[t] = it
      xt = self.lookup_table:forward(it)

    else
    --   print (string.format('t=%d, use reference', t))
      it = seq_ref[t-2]
      self.lookup_tables_inputs[t] = it
      xt = self.lookup_table:forward(it) -- using reference sentence all the way down

      sampleLogprobs, it = torch.max(logprobs, 2)
      it = it:view(-1):long()
    end

    if t >= 3 then
      seq[t-2] = it:squeeze() -- record the samples
      seqLogprobs[t-2] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
      fullLogProbs[t-2]:copy(logprobs:float()) -- and also full LogProbs
    end

   
    local inputs = {xt,unpack(state)}
    local out = self.core:forward(inputs)
    logprobs = out[self.num_state+1] -- last element is the output vector
    state = {}
    for j=1,self.num_state do table.insert(state, out[j]) end

  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs, fullLogProbs
end

--[[
input is a tuple of:
1. torch.Tensor of size NxK (K is dim of image code)
2. torch.LongTensor of size DxN, elements 1..M
   where M = opt.vocab_size and D = opt.seq_length

returns a (D+2)xNx(M+1) Tensor giving (normalized) log probabilities for the 
next token at every iteration of the LSTM (+2 because +1 for first dummy 
img forward, and another +1 because of START/END tokens shift)
--]]
function layer:updateOutput(input)
  local fc_feats = input[1]
  local seq = input[2]
  local batch_size = seq:size(2)
  local seq_len = input[3]
  local min_len, max_len = seq_len:min(), seq_len:max()
  local feat_len_loc = input[4] -- must be a 2-dimension table
  
  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass
  self:_createInitState(batch_size)

  self.output:resize(batch_size, self.text_feat_num, self.rnn_size):zero() -- only extract the last one

  self.state = {[0] = self.init_state}
  self.inputs = {}
  self.lookup_tables_inputs = {}
  self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency
  for t=1,self.seq_length+2 do
    
    local can_skip = false
    local xt
    if t == 1 then
      -- feed in the images
      if fc_feats:nElement() == 0 then
        -- feed in the start tokens
        local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
        self.lookup_tables_inputs[t] = it
        xt = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors
      else
        xt = fc_feats -- NxK sized input
      end
    elseif t == 2 then
      -- feed in the start tokens
      local it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      self.lookup_tables_inputs[t] = it
      xt = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors
    else
      -- feed in the rest of the sequence...
      local it = seq[t-2]:clone()
      if torch.sum(it) == 0 then
        -- computational shortcut for efficiency. All sequences have already terminated and only
        -- contain null tokens from here on. We can skip the rest of the forward pass and save time
        can_skip = true 
      end
      --[[
        seq may contain zeros as null tokens, make sure we take them out to any arbitrary token
        that won't make lookup_table crash with an error.
        token #1 will do, arbitrarily. This will be ignored anyway
        because we will carefully set the loss to zero at these places
        in the criterion, so computation based on this value will be noop for the optimization.
      --]]  
      it[torch.eq(it,0)] = 1
      if not can_skip then
        self.lookup_tables_inputs[t] = it
        xt = self.lookup_tables[t]:forward(it)
      end
    end

    if not can_skip then
      local h_state = self.state[t-1][self.num_state]

      self.inputs[t] = {xt,unpack(self.state[t-1])}
      -- forward the network
      -- print (self.inputs[t])
      local out = self.clones[t]:forward(self.inputs[t])

      -- process the outputs
      self.state[t] = {}
      for i=1,self.num_state do table.insert(self.state[t], out[i]) end
      self.tmax = t
      -- extract output
      local for_pred = out[self.num_state+1] -- the rest is state
      if t > 2 then
        for k = 1, seq_len:size(1) do
          local tab_len = feat_len_loc[k]
          if tab_len[1] == t-2 then 
            local p = self.text_feat_num+1 - #tab_len -- compute the id of text feat
            self.output[k][p]:copy(for_pred[k])   -- mean pooling
            table.remove(feat_len_loc[k], 1)  -- remove the already added one
          end
        end
      end
    end 
  end
  
  -- for exmerimental to average output so as to 
  -- for k = 1, #feat_len_loc do
  --   assert(#feat_len_loc[k] == 0)
  -- end
  self.output = self.output:transpose(2,3)

  return self.output
end

-- compute backprop gradients 
-- gradOutput is an (D+2)xNx(M+1) Tensor.
function layer:updateGradInput(input, gradOutput)
  
  local seq = input[2]
  local batch_size = seq:size(2)
  local seq_len = input[3]
  local min_len, max_len = seq_len:min(), seq_len:max()
  local feat_len_loc = input[4] -- must be a 2-dimension table

  -- for exmerimental to average output so as to 
  -- local gradOutput = gradOutput:clone()
  -- gradOutput = gradOutput:view(gradOutput:size(1), 1, gradOutput:size(2)):expandAs(self.output)
  local gradOutput = gradOutput:transpose(2,3)

  -- go backwards and lets compute gradients
  local dstate = {[self.tmax] = self.init_state} -- this works when init_state is all zeros
  self.__gradOutput = self.__gradOutput or gradOutput.new():resize(gradOutput:size(1),gradOutput:size(3))

  for t=self.tmax,1,-1 do
    self.__gradOutput:zero()
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    -- extracting gradOutput
    if t-2 <= max_len and t > 2 then
      for k = 1, seq_len:size(1) do
        local tab_len = feat_len_loc[k]
        if tab_len[#tab_len] == t-2 then
          local p = #tab_len -- compute the id of text feat
          -- print (string.format('add hidden gradient $d at time %d for data %d ', p, t-2, k ))  
          self.__gradOutput[k] = gradOutput[k][p] 
          table.remove(feat_len_loc[k], #feat_len_loc[k])  -- remove the already added one
        end
      end
    end
    table.insert(dout, self.__gradOutput)

    local dinputs = self.clones[t]:backward(self.inputs[t], dout)
    -- split the gradient to xt and to state
    local dxt = dinputs[1] -- first element is the input vector
    dstate[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do table.insert(dstate[t-1], dinputs[k]) end

    -- continue backprop of xt
    if t == 1 then
      dimgs = dxt
    else
      local it = self.lookup_tables_inputs[t]
      self.lookup_tables[t]:backward(it, dxt) -- backprop into lookup table
    end
  end

  -- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
  self.gradInput = {dimgs, torch.Tensor()}
  return self.gradInput
end