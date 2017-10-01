import subprocess
from .jason_helpers import *
import os
from uuid import uuid4 as gen_uuid
import re

class CRFSharp:
    def __init__(self, 
                 base_dir,
                 template,
                 model_dir=None,
                 result_dir=None,
                 data_dir=None,
                 modelfile=None,
                 thread=6, 
                 maxiter=20,
                 minfeafreq=1,
                 mindiff=0.0001,
                 debug=2,
                 vq=1,
                 slotrate=0.95,
                 nbest=3,
                 prob=2,
                 maxword=200
                 ):
        self.bin_dir = os.path.dirname(os.path.abspath(__file__)) + '/bin'
        assert os.path.isfile(self.bin_dir + '/CRFSharpConsole.exe')
        self.template = template
        assert os.path.isfile(self.template)
        self.base_dir = base_dir
        if not data_dir:
            self.data_dir = base_dir + '/data'
        else:
            self.data_dir = data_dir
        if not model_dir:
            self.model_dir = base_dir + '/model'
        if not result_dir:
            self.result_dir = base_dir + '/result'
        else:
            self.result_dir
        for dir_path in [self.base_dir, self.model_dir,\
                         self.result_dir, self.data_dir]:
            check_and_create_dir(dir_path)
        self.maxiter = maxiter
        self.minfeafreq = minfeafreq
        self.mindiff = mindiff
        self.thread = thread
        self.debug = debug
        self.vq = vq
        self.slotrate = slotrate
        self.nbest = nbest
        self.prob = prob
        self.maxword = maxword

    def run_encode_cmd(self, 
            trainfile,
            modelfile,
            retrainmodel=None):
        args = ['mono', 
                '{0}/CRFSharpConsole.exe'.format(self.bin_dir),
                '-encode',
                '-template', self.template,
                '-trainfile', trainfile,
                '-modelfile',  modelfile,
                '-maxiter',  str(self.maxiter),
                '-minfeafreq',  str(self.minfeafreq),
                '-mindiff',  str(self.mindiff),
                '-thread',  str(self.thread),
                '-debug',  str(self.debug),
                '-vq',  str(self.vq),
                '-slotrate', str(self.slotrate)]
        if retrainmodel:
            args += ['-retrainmode', retrainmodel]
        res = subprocess.check_output(args)
        return res

    def encode(self, sentences, labels):
        """
        args:
            sentences: list of list of tokens in strings
            labels: list of list of labels
        """
        assert len(sentences) == len(labels)
        for sentence, label in zip(sentences, labels):
            assert len(sentence) == len(label)
        temp_input_file = self.format_train_file(sentences, labels)
        self.modelfile = self.model_dir + '/' + str(gen_uuid()) + '.model'
        res = self.run_encode_cmd(temp_input_file, self.modelfile)
        os.remove(temp_input_file)

        return self.modelfile
    
    def format_train_file(self, sentences, labels):
        temp_input_file = self.data_dir + '/' + str(gen_uuid()) + '.test'
        with open(temp_input_file, 'w') as fp:
            for sentence, label in zip(sentences, labels):
                assert len(sentence) == len(label)
                for word, tag in zip(sentence, label):
                    fp.write(self.encode_token(word) + '\t' + tag + '\n')
                fp.write('\n')
        return temp_input_file

    def format_test_file(self, sentences):
        temp_input_file = self.data_dir + '/' + str(gen_uuid()) + '.train'
        with open(temp_input_file, 'w') as fp:
            for sentence in sentences:
                for word in sentence:
                    fp.write(self.encode_token(word) + '\n')
                fp.write('\n')
        return temp_input_file

    def encode_token(self, t):
        if t == ' ':
            t = 'SPACE'
        elif t == '\n':
            t = 'NEWLINE'
        elif re.match('[a-zA-Z]', t):
            pass
        elif re.match('[0-9]', t):
            t = 'NUM'
        else:
            t = 'SPECIAL'
        return t

    def run_decode_cmd(self, 
            testfile,
            modelfile,
            ):
        outputfile = self.result_dir + '/' + str(gen_uuid()) + '.result'
        segoutputfile = self.result_dir +'/'+ str(gen_uuid()) + '.segresult'
        args = ['mono', 
                '{0}/CRFSharpConsole.exe'.format(self.bin_dir),
                '-decode',
                '-modelfile', modelfile,
                '-inputfile', testfile,
                '-outputfile', outputfile,
                '-outputsegfile', segoutputfile,
                '-thread', str(self.thread),
                '-nbest', str(self.nbest),
                '-prob', str(self.prob),
                '-maxword', str(self.maxword),
                ]
        subprocess.check_output(args)
        return outputfile, segoutputfile

    def make_phrases(self, tags):
        phrases = []
        prev_tag = tags[0]
        for tag in tags:
            tag = self._get_tag(tag)
            if prev_tag != tag:
                phrases.append(prev_tag)
                prev_tag = tag
        phrases.append(prev_tag)
        return phrases

    def parse_outputfile(self, outputfile, srcids):
        with open(outputfile, 'r') as fp:
            lines = fp.readlines()

        # init looping
        res = dict()
        cand_cnt = -1
        srcid_idx = 0
        srcid = srcids[srcid_idx]
        one_res = dict()
        one_res['cands'] = []
        one_res['sentence'] = []
        res[srcid] = one_res

        # Parse the file
        for line in lines:
            if line[0] == '#':
                cand_cnt += 1
                if cand_cnt == self.nbest:
                    srcid_idx += 1
                    srcid = srcids[srcid_idx]
                    one_res = dict()
                    one_res['cands'] = []
                    res[srcid] = one_res
                    cand_cnt = 0
                    one_res['sentence'] = []
                cand = dict()
                one_res['cands'].append(cand)
                p = float(line[1:-1])
                cand['prop'] = p
                srcid = srcids[srcid_idx]
                cand['token_predict'] = []
            elif line != '\n':
                tokens = line.split()
                if cand_cnt == 0:
                    one_res['sentence'].append(tokens[0])
                cand['token_predict'].append(tokens[1])

        for srcid, one_res in res.items():
            for cand in one_res['cands']:
                cand['phrase_predict'] = self.make_phrases(
                                            cand['token_predict'])
        return res

    def _get_tag(self, raw_tag):
        if raw_tag != 'O':
            raw_tag = raw_tag[2:]
        return raw_tag


    def decode(self, sentences, srcids=None, modelfile=None):
        if not srcids:
            srcids = range(0, len(sentences))
        if not modelfile:
            modelfile = self.modelfile
        temp_input_file = self.format_test_file(sentences)
        outputfile, segoutputfile = self.run_decode_cmd(temp_input_file, \
                                                        modelfile)
        os.remove(temp_input_file)
        return self.parse_outputfile(outputfile, srcids)
