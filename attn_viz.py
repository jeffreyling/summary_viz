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


def load_file(fn, num_docs, hop=False, soft_baseline=False, no_pad=False, sent_length=58, num_sents=10):
    pred_sents = []
    src_sents = []
    gold_sents = []
    raw = open(fn, 'r').read().split('\n')[:-1]
    attn = []
    attn_gold = []
    attn_sent = []
    attn_gold_sent = []
    attn_i = []
    attn_i_sent = []
    attn_i_gold = []
    attn_i_gold_sent = []
    decoder = 0
    sent_level = False
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
                # print len(attn_i), len(attn_i[0])
                # raw_input()
                attn.append(np.array(attn_i))
                attn_gold.append(np.array(attn_i_gold))
                attn_sent.append(np.array(attn_i_sent))
                attn_gold_sent.append(np.array(attn_i_gold_sent))
                attn_i = []
                attn_i_gold = []
                attn_i_sent = []
                attn_i_gold_sent = []

            line = line[line.find(':')+1:]
            if OLD_TORCH:
                line = line[:-5].decode("utf-8").strip()
            else:
                line = line.decode("utf-8").strip()
            if soft_baseline or no_pad:
                line = ' '.join(line.split()[:num_sents*sent_length]) # for soft baseline
            else:
                # hierarchical
                sents = line.strip('</s>').split('</s>')
                sents = [' '.join(sent.strip().split()[:sent_length]) for sent in sents]
                line = ' '.join(sents)
                # print line
                # print sents
                # for sent in sents: print(len(sent.split(' ')))
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
            if OLD_TORCH:
                if line[9] == 'S': continue #SCORE
            else:
                if line[5] == 'S': continue #SCORE
            line = line[line.find(':')+1:]
            if OLD_TORCH:
                line = line[:-5].decode("utf-8").strip()
            else:
                line = line.decode("utf-8").strip()
            # cut off start and end tag
            line = ' '.join(line.strip().split()[1:-1])
            gold_sents.append(line)
        elif col=="ATTN":
            if OLD_TORCH:
                if line[9:-5] == "GOLD":
                    decoder = 0 # gold
                    sent_level = False
                elif line[9:-5] == "LEVEL GOLD":
                    decoder = 0
                    sent_level = True
                elif line[9:-5] == "PRED":
                    decoder = 1
                    sent_level = False
                else:
                    decoder = 1 # pred      
                    sent_level = True
            else:
                line = line.strip()
                if line[5:] == "GOLD":
                    decoder = 0 # gold
                    sent_level = False
                elif line[5:] == "LEVEL GOLD":
                    decoder = 0
                    sent_level = True
                elif line[5:] == "PRED":
                    decoder = 1
                    sent_level = False
                else:
                    decoder = 1 # pred      
                    sent_level = True
        elif col=="attn" or col=="gold":
            continue
        else:
            if OLD_TORCH:
                line = line[:-9]
            if line.strip() == '':
                continue
            cur_attn = map(float, line.strip().split())
            if decoder == 1:
                if sent_level:
                    attn_i_sent.append(cur_attn)
                else:
                    attn_i.append(cur_attn)
            else:
                if sent_level:
                    attn_i_gold_sent.append(cur_attn)
                else:
                    attn_i_gold.append(cur_attn)
    return pred_sents, src_sents, gold_sents, attn, attn_gold, attn_sent, attn_gold_sent

def make_source(gold, sent_attn_only=False, sent_length=20, num_sents=10):
    if sent_attn_only:
        idx = 0
        source = []
        gold = gold.strip().split()
        while idx*sent_length < len(gold):
            sent = gold[idx*sent_length:(idx+1)*sent_length]    
            source.append(' '.join(sent))
            idx += 1
    else:
        # if repeat_words > 0:
            # source = []
            # idx = 0
            # for word in gold.split():
                # source.append(word)
                # idx += 1
                # if idx >= sent_length:
                    # source = source + source[-repeat_words:]
                    # idx = repeat_words

            # source = source[:num_sents*sent_length]
        # else:
        source = gold.split()

        # remove blanks
        while source[-1] == '<blank>':
            source.pop()

    return source


def plot_hm(source, targets, data, fn, sent_attn_only=False):
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
    if sent_attn_only:
        ax.figure.set_size_inches(float(len(targets))/len(source),5)
    else:
        ax.figure.set_size_inches(20*float(len(targets))/len(source),20)
    plt.xticks(rotation=90, ha="left")
    plt.yticks(rotation=0)
    plt.savefig(fn, dpi=300, bbox_inches='tight')

def topk_attn(attn, multisampling):
    n = attn.shape[1]
    for row in attn:
        idx = np.argsort(row)
        row[idx[:n-multisampling]] = 0

    attn = attn / attn.sum(axis=1, keepdims=True)
    return attn

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument('--soft_baseline', help="Visualizing soft baseline", type=bool, default=False)
    parser.add_argument('--sent_length', help="Sentence lengths", type=int, default=40)
    parser.add_argument('--num_sents', help="Num sents in doc for no pad", type=int, default=10)
    parser.add_argument('--no_pad', help="No pad doc", type=bool, default=True)
    # parser.add_argument('--repeat_words', help="Repeat words for no_pad doc", type=int, default=0) # DO NOT SET: beam.lua does it already
    parser.add_argument('--num_docs', help="How many docs to generate", type=int, default=15)
    parser.add_argument('--multisampling', help="Multisampling: take topk attn", type=int, default=0)
    args = parser.parse_args(arguments)
    print 'args:', args

    filename = args.filename
    preds, srcs, golds, attns, attns_gold, attns_sent, attns_gold_sent = load_file(filename, args.num_docs, soft_baseline=args.soft_baseline, sent_length=args.sent_length, no_pad=args.no_pad, num_sents=args.num_sents)
    sns.set(font='sans-serif', font_scale=0.3)

    basename = filename.split('.')[0]
    basename = os.path.join('output', basename)

    if not os.path.isdir(basename):
        os.mkdir(basename)

    for i in range(args.num_docs):
        source = make_source(srcs[i], sent_length=args.sent_length, sent_attn_only=False)
        source_sent = make_source(srcs[i], sent_length=args.sent_length, sent_attn_only=True)

        # pred
        targets = preds[i]
        # data = attns[i].T
        # outfilename = os.path.join(basename, str(i)+"pred.png")
        # plot_hm(source, targets.split(), data, outfilename, sent_attn_only=False)

        # pred sent
        data = topk_attn(attns_sent[i], args.multisampling).T
        outfilename = os.path.join(basename, str(i)+"predsent.png")
        plot_hm(source_sent, targets.split(), data, outfilename, sent_attn_only=True)

        # gold
        targets = golds[i]
        # data = attns_gold[i].T
        # outfilename = os.path.join(basename, str(i)+"gold.png")
        # plot_hm(source, targets.split(), data, outfilename, sent_attn_only=False)

        # gold sent
        data = topk_attn(attns_gold_sent[i], args.multisampling).T
        outfilename = os.path.join(basename, str(i)+"goldsent.png")
        plot_hm(source_sent, targets.split(), data, outfilename, sent_attn_only=True)

        if i % 5 == 0:
            print 'finished', i

if __name__ == '__main__':
    main(sys.argv[1:])
