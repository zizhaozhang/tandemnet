-- Necessary functions you need to deploy training
--     1. getBatch(): See train.lua/cuda_batch() to know what should be returned.
--     2. total_amount(): The total number of training data.

require 'image'

local DataLoader = torch.class('DataLoader')

function DataLoader:__init(opt)
    self.batch_size = opt.batch_size
    self.img_width = 224
    self.img_height = 224
    self.vocab_size = opt.vocab_size
end

function DataLoader:getBatch(which)
    -- Input:
        -- which: {train, test, val}
    -- initialization
    local images = torch.Tensor(self.batch_size, 3,self.img_height,self.img_width)
    -- Each entry of labels is a value between [0, vocabulary_size] wehre 0 is END_TOKEN
    local labels = torch.LongTensor(self.vocab_size, self.batch_size)
    local binary_labels = torch.LongTensor(self.batch_size)
    
    -- TODO: fill out those matrics. See 
    return images, labels, binary_labels
end

function DataLoader::total_amount(which)
    local data_size
    -- TODO
    return data_size
end