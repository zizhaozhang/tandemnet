require 'nngraph'

local utils = require 'utils.utils'

local layer, parent = torch.class('nn.MultiModel', 'nn.Module')

function layer:__init(opt, nLabel)
    parent.__init(self)

    local parms = {}
    parms.conv_feat_size = utils.getopt(opt, 'conv_feat_size', 256)
    parms.multfeat_dim = utils.getopt(opt, 'multfeat_dim', 256)
    parms.attfeat_dim = utils.getopt(opt, 'attfeat_dim', 256)
    parms.conv_feat_num = utils.getopt(opt, 'conv_feat_num', 196)
    parms.height = math.sqrt(parms.conv_feat_num)
    parms.drop_ratio = utils.getopt(opt, 'drop_prob_mm', 0.5)
    parms.text_feat_num = utils.getopt(opt, 'text_feat_num', 5)
    self.deathRate = utils.getopt(opt, 'deathRate', 0.5)

    if self.deathRate == 0 then print ('--> the death Rate is set to zero') end
    --  build model
    local SpatialConvolution = cudnn.SpatialConvolution
    local Avg = cudnn.SpatialAveragePooling
    local ReLU = cudnn.ReLU
    local Max = nn.SpatialMaxPooling
    local SoftMax = cudnn.SoftMax
    local Tanh = cudnn.Tanh

    local ifeat = nn.Identity()()   -- image convolutional feat BxDxhxw
    local sfeat = nn.Identity()()   -- language feat    BxDx5
        local in_qfeat = nn.Dropout(parms.drop_ratio)(sfeat)
        local in_ifeat = nn.Dropout(parms.drop_ratio)(ifeat)
        -- do a image feature embedding to Tanh range
        in_ifeat = Tanh()(SpatialConvolution(parms.conv_feat_size, parms.multfeat_dim, 1,1,1,1,0,0)(in_ifeat))
        -- do a text feature embedding to Tanh range
        in_qfeat = nn.View(parms.multfeat_dim, parms.text_feat_num, 1)(in_qfeat)
        in_qfeat = Tanh()(SpatialConvolution(parms.multfeat_dim, parms.multfeat_dim, 1,1,1,1,0,0)(in_qfeat))
        in_qfeat = nn.View(parms.multfeat_dim, parms.text_feat_num)(in_qfeat)
        -- image attention
        local globqfeat = nn.Mean(2,2)(in_qfeat)
        local qfeatatt = nn.Replicate(parms.conv_feat_num, 3)(nn.Linear(parms.multfeat_dim, parms.attfeat_dim)(globqfeat))
        local ifeatproj = SpatialConvolution(parms.multfeat_dim, parms.attfeat_dim, 1,1,1,1,0,0)(in_ifeat)
        local ifeatatt = nn.View(parms.attfeat_dim, parms.conv_feat_num):setNumInputDims(3)(ifeatproj)
        local addfeat = nn.View(parms.attfeat_dim,parms.conv_feat_num,1):setNumInputDims(2)(Tanh()(nn.CAddTable()({ifeatatt, qfeatatt})))

        -- local iattscore = nn.View(parms.conv_feat_num):setNumInputDims(3)(SpatialConvolution(parms.attfeat_dim,1,1,1,1,1,0,0)(addfeat))
        -- text attention
        local globifeat = nn.View(parms.multfeat_dim):setNumInputDims(3)(Avg(parms.height,parms.height,1,1)(in_ifeat))
        local ifeatatt = nn.Replicate(parms.text_feat_num, 3)(nn.Linear(parms.multfeat_dim, parms.attfeat_dim)(globifeat))
        local tfeatproj = SpatialConvolution(parms.multfeat_dim, parms.attfeat_dim, 1,1,1,1,0,0)(nn.View(parms.multfeat_dim,parms.text_feat_num,1):setNumInputDims(2)(in_qfeat))
        tfeatproj = nn.View(parms.multfeat_dim, parms.text_feat_num):setNumInputDims(3)(tfeatproj)
        local addfeat2 = nn.View(parms.attfeat_dim,parms.text_feat_num,1):setNumInputDims(2)(Tanh()(nn.CAddTable()({ifeatatt, tfeatproj})))

        -- local qattscore = nn.View(parms.text_feat_num):setNumInputDims(3)(SpatialConvolution(parms.attfeat_dim,1,1,1,1,1,0,0)(addfeat2))
       
        -- global attention
        local attfeat_join = nn.JoinTable(2, 3)({addfeat, addfeat2})
        att = nn.View(parms.text_feat_num+parms.conv_feat_num):setNumInputDims(3)(SpatialConvolution(parms.attfeat_dim,1,1,1,1,1,0,0)(attfeat_join))
        att = SoftMax()(att)

        local flat_ifeat = nn.View(parms.multfeat_dim, parms.conv_feat_num):setNumInputDims(3)(in_ifeat)
        local joint_feat = nn.JoinTable(2,2)({flat_ifeat, in_qfeat})
        local att_feat = nn.MV()({joint_feat, att})
        -- concat context vector and image feat
        local output_feat =  nn.CAddTable()({att_feat, globifeat})

        local out = nn.Linear(parms.multfeat_dim, nLabel)(nn.Dropout(0.5)(output_feat))

    self.model = nn.gModule({ifeat, sfeat}, {out, att})
    -- stochastic
    self.gate = true
    self.train = true
end

function layer:sampleGates()
    self.gate = true
    if torch.rand(1)[1] < self.deathRate then 
        self.gate = false
    end
end

function layer:parameters()
    return self.model:parameters()
end
function layer:training()
    self.model:training()
    self.train = true
end
function layer:evaluate()
    self.model:evaluate()
    self.train = false
end
--[[
input: 1) conv_image_feats 2) text_feats
---]]
function layer:updateOutput(input)

    self:sampleGates()

    local ifeat = input[1]
    local tfeat = input[2]
    -- assert(ifeat:size(2) == tfeat:size(2), string.format('unmatch feat size %d ~= %d',ifeat:size(2), tfeat:size(2)))
    if self.train then
        if self.gate == false then
            tfeat:zero()
        end
    else -- evaluate mode
        tfeat:mul(1-self.deathRate)
    end

    self.model:forward({ifeat, tfeat})
    self.output = self.model.output
    return self.output
end

function layer:updateGradInput(input, gradOutput)
    local ifeat = input[1]
    local tfeat = input[2]

    -- if self.train then
    --     if self.gate == false then
    --         assert(tfeat:sum() == 0)
    --     end
    -- end
    self.model:backward(input, gradOutput)
    self.gradInput = self.model.gradInput
    if self.train then
        if self.gate == false then
            self.gradInput[2]:zero()
        end
    end
    return self.gradInput
end