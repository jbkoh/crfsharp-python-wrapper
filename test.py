from crfsharp import CRFSharp
import json
import arrow


with open('test_config.json', 'r') as fp:
    config = json.load(fp)

if __name__ == '__main__':
    model = CRFSharp(base_dir='./temp', 
                     template='data/template.NE', 
                     thread=30,
                     nbest=10)
    # Learn from file

    """ # learning from file
    model.run_encode_cmd(
            trainfile='data/crfsharp.train', 
            modelfile='data/crfsharp.model')
    """
    
    """ # predicting from file
    model.run_decode_cmd(
            testfile='data/crfsharp.test', 
            modelfile='data/crfsharp.model')
    """

    """ # Testing parsing
    outputfile = 'temp/data/42bce14b-9269-4746-a01a-5e6521088544.result'
    srcids = range(0, 27)
    model.parse_outputfile(outputfile, srcids)
    """

    # Testing in python
    with open('data/bml_char_label_dict.json', 'r') as fp:
        label_dict = json.load(fp)
    srcids = list(label_dict.keys())[0:20]
    sentences = list()
    labels = list()
    for srcid in srcids:
        sentence = [comp[0] for comp in label_dict[srcid]]
        label = [comp[1] for comp in label_dict[srcid]]
        sentences.append(sentence)
        labels.append(label)
    begin_time = arrow.get()
    modelfile = model.encode(sentences, labels)
    res = model.decode(sentences, srcids)
    end_time = arrow.get()
    print('took: ', end_time - begin_time)





