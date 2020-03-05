import sys
import perceptron_pos_tagger as ppt
import data_structures


def read_in_gold_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [[tup.split('_') for tup in line.split()] for line in lines]
        sents = [data_structures.Sentence(line) for line in lines]

    return sents 


def read_in_plain_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        sents = [data_structures.Sentence(line) for line in lines]

    return sents 


def output_auto_data(outfile, auto_data, untagged_data):
    ''' According to the data structure you used for "auto_data",
        write code here to output your auto tagged data into a file,
        using the same format as the provided gold data (i.e. word_pos word_pos ...). 
    '''
    with open(outfile, "w") as outf:
        for sent_index in range(len(auto_data)):
            predicted_tags = auto_data[sent_index]
            sent = untagged_data[sent_index].snt
            output_str = ""
            for i in range(len(predicted_tags)):
                output_str += sent[i] + "_" + predicted_tags[i] + " "
            outf.write(output_str.strip() + "\n")


if __name__ == '__main__':

    # Run python train_test_tagger.py train/ptb_02-21.tagged dev/ptb_22.tagged dev/ptb_22.snt test/ptb_23.snt to train & test your tagger
    # train_file = sys.argv[1]
    # gold_dev_file = sys.argv[2]
    # plain_dev_file = sys.argv[3]
    # test_file = sys.argv[4]

    train_file = "./train/ptb_02-21.tagged"
    gold_dev_file = "./dev/ptb_22.tagged"
    plain_dev_file = "./dev/ptb_22.snt"
    test_file = "./test/ptb_23.snt"

    # Read in data
    train_data = read_in_gold_data(train_file)
    gold_dev_data = read_in_gold_data(gold_dev_file)
    plain_dev_data = read_in_plain_data(plain_dev_file)
    test_data = read_in_plain_data(test_file)


    # regular Perceptron
    pos_tagger = ppt.Perceptron_POS_Tagger()
    preds = pos_tagger.run(train_data, gold_dev_data, plain_dev_data, epoch=6)
    # Output your auto tagged data
    output_auto_data("dev_output.tagged", preds, plain_dev_data)

    # averaged Perceptron
    avg_pos_tagger = ppt.Perceptron_POS_Tagger("avg")
    preds_avg = avg_pos_tagger.run(train_data, gold_dev_data, plain_dev_data, epoch=6, n=50)
    # Output your auto tagged data
    output_auto_data("avg_dev_output.tagged", preds, plain_dev_data)


