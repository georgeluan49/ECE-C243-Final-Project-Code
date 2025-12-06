
modelName = 'speechBaseline4adamwsmalleps'

data_dir = "/home/qitongluan/neural_seq_decoder/competitionData"

args = {}
args['outputDir'] = data_dir + '/logs/speech_logs/' + modelName
args['datasetPath'] = data_dir + '/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 0.02 # 0.01
args['lrEnd'] = 0.02 # 0.01
args['nUnits'] = 1024
args['nBatch'] = 5000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5

import os
print(os.getcwd())
from src.neural_decoder.neural_decoder_trainer_AdamW import trainModel

trainModel(args)