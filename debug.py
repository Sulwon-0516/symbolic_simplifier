from Models.train_utils import *
from Data.halide_utils import HalideVocab
import json
from Optimizer.transform_rule import RuleSet
from Models.EncoderDecoder import EncoderDecoder
from Data.Batcher import PipelineBatcher

if __name__ == '__main__':
    # to debug dataloader
    pipeline_batcher = PipelineBatcher()

    #pp_b_org = Batcher_org.PipelineBatcher()
    config = "/home/inhee/symbolic_simplifier/pretrain_log_first/D4_16_6_7/config.json"
    with open(config) as fp:
        hps = json.load(fp)
        
    encoder_name = hps["encoder_name"]
    decoder_name = hps["decoder_name"]
    #vocab_org = pp_b_org.vocab
    vocab = pipeline_batcher.vocab
    calibrate_hps(hps, vocab, decoder_name)
    rules = RuleSet(vocab)
 
    
    e2d = EncoderDecoder(encoder_name, decoder_name, hps, device=get_device())
    
    rules = pretrain_loader(e2d,rules, vocab)
    print("all ok")