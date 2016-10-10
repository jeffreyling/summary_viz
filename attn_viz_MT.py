import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sys
import re
import os
import argparse

OLD_TORCH = False

def calculate_sentence_entropy(attn, doc):
    words = doc.strip().split()
    idxs = [-1]
    for i,w in enumerate(words):
        if w == '</s>':
            idxs.append(i)
    idxs.append(len(words))

    ent = 0
    for cur_attn in attn:
        sent_attn = []
        for j in range(len(idxs)):
            if j == 0: continue
            sent_attn.append(sum(cur_attn[idxs[j-1]+1:idxs[j]]))

        # print sent_attn
        # raw_input()
        cur_ent = sum([-p*np.log(p) for p in sent_attn if p != 0])
        ent += cur_ent
    return ent / len(attn)


def load_file(fn, num_docs, hop=False):
    pred_sents = []
    src_sents = []
    raw = open(fn, 'r').read().split('\n')[:-1]
    attn = []
    attn_i = []
    for i, line in enumerate(raw):      
        if len(pred_sents) > num_docs: break
        if i < 4: continue
        if OLD_TORCH:
            col = line[4:8]
        else:
            col = line[:4]
        if col=="SENT":
            if i > 4:
                # print 'gold:', src_sents[-1], len(src_sents[-1].split())
                # print 'pred:', pred_sents[-1], len(pred_sents[-1].split())
                # print len(attn_i)
                # raw_input()
                attn.append(np.array(attn_i))
                attn_i = []

            line = line[line.find(':')+1:]
            if OLD_TORCH:
                line = line[:-5].decode("utf-8").strip()
            else:
                line = line.decode("utf-8").strip()

            # print line
            # print len(line.split())
            # raw_input()
            src_sents.append(line)
        elif col=="PRED":
            if OLD_TORCH:
                if line[9] == 'S': continue #SCORE
            else:
                if line[5] == 'S': continue #SCORE
            line = line[line.find(':')+1:]
            if OLD_TORCH:
                line = line[:-5].decode("utf-8").strip()
            else:
                line = line.decode("utf-8").strip()
            pred_sents.append(line)
        elif col=="GOLD":
            continue
        elif col=="attn" or col=="gold":
            continue
        else:
            if OLD_TORCH:
                line = line[:-9]
            if line.strip() == '':
                continue
            cur_attn = map(float, line.strip().split())
            attn_i.append(cur_attn)

    return pred_sents, src_sents, attn

def make_source(gold, repeat_words=0, sent_attn_only=False, sent_length=20):
    if sent_attn_only:
        idx = 0
        source = []
        gold = gold.strip().split()
        while idx*sent_length < len(gold):
            sent = gold[idx*sent_length:(idx+1)*sent_length]    
            source.append(' '.join(sent))
            idx += 1
    else:
        if repeat_words > 0:
            source = []
            idx = 0
            for word in gold.split():
                source.append(word)
                idx += 1
                if idx >= sent_length:
                    source = source + source[-repeat_words:]
                    idx = repeat_words

            source = source[:200]
        else:
            source = gold.split()

        # remove blanks
        while source[-1] == '<blank>':
            source.pop()

    return source


def plot_hm(source, targets, data, fn):
    source = source.split()
    if data.shape[0] != len(source):
        print 'source'
        print source
        print data.shape, len(source)
        raw_input()
    if data.shape[1] != len(targets) + 1:
        print 'target'
        print targets
        print data.shape, len(targets) + 1
        raw_input()
    ax = sns.heatmap(data, xticklabels = targets, yticklabels = source, cmap="Blues", robust=True, cbar=False, linewidth=.5,
                     vmin=0.0, vmax=1.0)
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.figure.set_size_inches(8*float(len(targets))/len(source),8)
    plt.xticks(rotation=40, ha="left")
    plt.yticks(rotation=0)
    plt.savefig(fn, dpi=300, bbox_inches='tight')


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    # parser.add_argument('--sent_length', help="Sentence lengths", type=int, default=20)
    # parser.add_argument('--repeat_words', help="Repeat words for no_pad doc", type=int, default=0)
    parser.add_argument('--num_docs', help="How many docs to generate", type=int, default=15)
    args = parser.parse_args(arguments)
    print 'args:', args

    filename = args.filename
    preds, srcs, attns = load_file(filename, args.num_docs)
    sns.set(font='sans-serif', font_scale=0.8)

    basename = filename.split('.')[0]
    basename = os.path.join('output', basename)

    if not os.path.isdir(basename):
        os.mkdir(basename)

    for i in range(args.num_docs):
        source = srcs[i] # make_source(srcs[i], repeat_words=args.repeat_words, sent_length=args.sent_length, sent_attn_only=False)
        # source_sent = make_source(srcs[i], repeat_words=args.repeat_words, sent_length=args.sent_length, sent_attn_only=True)

        # pred
        targets = preds[i]
        data = attns[i].T
        outfilename = os.path.join(basename, str(i)+"pred.png")
        plot_hm(source, targets.split(), data, outfilename)

        # pred sent
        # data = attns_sent[i].T
        # outfilename = os.path.join(basename, str(i)+"predsent.png")
        # plot_hm(source_sent, targets.split(), data, outfilename, sent_attn_only=True)

        # gold
        # targets = golds[i]
        # data = attns_gold[i].T
        # outfilename = os.path.join(basename, str(i)+"gold.png")
        # plot_hm(source, targets.split(), data, outfilename, sent_attn_only=False)

        # gold sent
        # data = attns_gold_sent[i].T
        # outfilename = os.path.join(basename, str(i)+"goldsent.png")
        # plot_hm(source_sent, targets.split(), data, outfilename, sent_attn_only=True)

        if i % 5 == 0:
            print 'finished', i

if __name__ == '__main__':
    main(sys.argv[1:])
