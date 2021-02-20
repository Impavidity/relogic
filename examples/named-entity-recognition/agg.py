import os, glob, itertools, sys

def generate_collection(path, tag):
    results = itertools.chain.from_iterable(glob.iglob(os.path.join(root,'*.gold_conll'))
                                               for root, dirs, files in os.walk(path + tag))

    text = ""
    for cur_file in results: 
        with open(cur_file, 'r') as f:
            print(cur_file)
            flag = None
            for line in f.readlines():
                l = line.strip()
                l = ' '.join(l.split())
                ls = l.split(" ")
                if len(ls) >= 11:
                    word = ls[3]
                    pos = ls[4]
                    cons = ls[5]
                    ori_ner = ls[10]
                    ner = ori_ner
                    # print(word, pos, cons, ner)
                    if ori_ner == "*":
                        if flag==None:
                            ner = "O"
                        else:
                            ner = "I-" + flag
                    elif ori_ner == "*)":
                        ner = "I-" + flag
                        flag = None
                    elif ori_ner.startswith("(") and ori_ner.endswith("*") and len(ori_ner)>2:
                        flag = ori_ner[1:-1]
                        ner = "B-" + flag
                    elif ori_ner.startswith("(") and ori_ner.endswith(")") and len(ori_ner)>2 and flag == None:
                        ner = "B-" + ori_ner[1:-1]

                    text += "\t".join([word, pos, cons, ner]) + '\n'
                else:
                    text += '\n'
            text += '\n'
            # break

    with open("onto."+tag+".ner", 'w') as f:
        f.write(text)


generate_collection(sys.argv[1], "train")
generate_collection(sys.argv[1], "test")
generate_collection(sys.argv[1], "development")