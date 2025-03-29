import string
from collections import defaultdict

'''
    This function constructs folds that have a balanced category distribution.
    Folds are stacked up together to give the order of docs in the main data.
    
    idx_order defines the order of documents in the data. Each sequence of (docs_per_fold) documents in idx_order can be treated as a single fold, containing documents balanced across each category.
'''
def prepare_folds(args):
    n_docs = args.dataset_size
    docs_per_fold = n_docs // args.num_folds
    
    folds = [[] for _ in range(args.num_folds)]
    
    for i in range(1, n_docs):
        folds[i % args.num_folds].append(i)  # Assign documents sequentially to folds

    idx_order = sum(folds, [])  # Flatten the list
    return idx_order


'''
    This file prepares the numericalized data in the form of lists, to be used in training mode.
    idx_order is the order of documents in the dataset.

        x:  list[num_docs, sentences_per_doc, words_per_sentence]       if pretrained = False
            list[num_docs, sentences_per_doc, sentence_embedding_dim]   if pretrained = True
        y:  list[num_docs, sentences_per_doc]
'''

def prepare_data(idx_order, args):
    x, y = [], []

    word2idx = defaultdict(lambda: len(word2idx))
    tag2idx = defaultdict(lambda: len(tag2idx))

    # Initialize special tokens
    word2idx['<pad>'], word2idx['<unk>'] = 0, 1
    tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>'] = 0, 1, 2

    # Iterate over documents
    for doc in idx_order:
        doc_x, doc_y = [], []

        doc_path = f"{args.data_path}\\file_{str(doc)}.txt"  # Ensure doc is a string

        try:
            with open(doc_path, 'r', encoding='utf-8') as fp:
                # Iterate over sentences
                for sent in fp:
                    sent = sent.strip()
                    if not sent:
                        continue  # Skip empty lines
                    
                    try:
                        sent_x, sent_y = sent.strip().split('\t')
                    except ValueError:
                        print(f"Skipping malformed line: {sent}")
                        continue

                    # Process words
                    if not args.pretrained:
                        sent_x = sent_x.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                        sent_x = [word2idx[word] for word in sent_x.split()]
                    else:
                        sent_x = list(map(float, sent_x.split()[:args.emb_dim]))

                    sent_y = tag2idx[sent_y.strip()]

                    if sent_x:
                        doc_x.append(sent_x)
                        doc_y.append(sent_y)

        except FileNotFoundError:
            print(f"Warning: File not found - {doc_path}")
            continue

        x.append(doc_x)
        y.append(doc_y)

    return x, y, word2idx, tag2idx  # Convert defaultdict to dict before returning


'''
    This file prepares the numericalized data in the form of lists, to be used in inference mode.
    idx_order is the order of documents in the dataset.

        x:  list[num_docs, sentences_per_doc, words_per_sentence]       if pretrained = False
            list[num_docs, sentences_per_doc, sentence_embedding_dim]   if pretrained = True
'''
def prepare_data_inference(idx_order, args, sent2vec_model):
    x = []

    # iterate over documents
    for doc in idx_order:
        doc_x = []
        doc_path = fr"{args.data_path}\\{str(doc)}.txt"
        with open(doc_path, 'r', encoding='utf-8') as fp:
            
            # iterate over sentences
            for sent in fp:
                sent_x = sent.strip()

                # cleanse text, map words and tags
                if not args.pretrained:
                    sent_x = sent_x.lower().translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
                    sent_x = list(map(lambda x: args.word2idx[x] if x in args.word2idx else args.word2idx['<unk>'], sent_x.split()))
                else:
                    sent_x = sent2vec_model.embed_sentence(sent_x).flatten().tolist()[:args.emb_dim]
                
                if sent_x != []:
                    doc_x.append(sent_x)
                    
        x.append(doc_x)

    return x
    