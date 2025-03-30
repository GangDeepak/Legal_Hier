import argparse
import sent2vec
import os
import json
import torch
import pickle
import sys



# Import the required model and functions
from Hier_BiLSTM_CRF.model.Hier_BiLSTM_CRF import Hier_LSTM_CRF_Classifier
from Hier_BiLSTM_CRF.prepare_data import prepare_data_inference
from Hier_BiLSTM_CRF.train import infer_step

def main():
    parser = argparse.ArgumentParser(description="Infer tags for unannotated files")

    # Define arguments
    parser.add_argument('--pretrained', default=False, type=bool, help="Whether to use pretrained sentence embeddings")
    parser.add_argument('--data_path', default=r'C:\Users\Deepak\OneDrive\Desktop\New folder\data\Hier_BiLSTM_CRF\test', type=str, help="Folder containing text files")
    parser.add_argument('--model_path', default=r'C:\Users\Deepak\OneDrive\Desktop\New folder\saved_models\Hier_BiLSTM_CRF\model_state4.tar', type=str, help="Path to trained model")
    parser.add_argument('--sent2vec_model_path', default='infer/sent2vec.bin', type=str, help="Path to trained sent2vec model (if pretrained=True)")
    parser.add_argument('--save_path', default=r'C:\Users\Deepak\OneDrive\Desktop\New folder\outputs\Hier_BiLSTM_CRF\predictions.txt', type=str, help="Path to save predictions")
    parser.add_argument('--word2idx_path', default=r'C:\Users\Deepak\OneDrive\Desktop\New folder\saved_models\Hier_BiLSTM_CRFword2idx.json', type=str, help="Path to word2idx dictionary")
    parser.add_argument('--tag2idx_path', default=r'C:\Users\Deepak\OneDrive\Desktop\New folder\saved_models\Hier_BiLSTM_CRFtag2idx.json', type=str, help="Path to tag2idx dictionary")
    parser.add_argument('--emb_dim', default=200, type=int, help="Sentence embedding dimension")
    parser.add_argument('--word_emb_dim', default=100, type=int, help="Word embedding dimension (if pretrained=False)")
    parser.add_argument('--device', default='cuda', type=str, help="Device: cuda / cpu")

    args = parser.parse_args()

    # Load word2idx and tag2idx
    with open(args.word2idx_path) as fp:
        args.word2idx = json.load(fp)
        
    with open(args.tag2idx_path) as fp:
        args.tag2idx = json.load(fp)
    
    # Load pretrained sent2vec model if needed
    if args.pretrained:
        print("Loading pretrained sent2vec model...", end=" ", flush=True)
        sent2vec_model = sent2vec.Sent2vecModel()
        sent2vec_model.load_model(args.sent2vec_model_path)
        print("Done")
    else:
        sent2vec_model = None

    # Prepare data
    print("\nPreparing data...", end=" ", flush=True)
    idx_order = list(map(lambda x: os.fsdecode(x)[:-4], os.listdir(os.fsencode(args.data_path))))
    x = prepare_data_inference(idx_order, args, sent2vec_model)
    print("Done")

    # Load model
    print("\nLoading model...", end=" ", flush=True)

    ckpt = torch.load(
        r"C:\Users\Deepak\OneDrive\Desktop\New folder\saved_models\Hier_BiLSTM_CRFmodel_state4.tar",
        map_location=torch.device(args.device),
        pickle_module=pickle 
    )
    # Define model before loading state_dict
    model = Hier_LSTM_CRF_Classifier(
        n_tags=len(args.tag2idx),
        sent_emb_dim=args.emb_dim,
        sos_tag_idx=args.tag2idx['<start>'],
        eos_tag_idx=args.tag2idx['<end>'],
        pad_tag_idx=args.tag2idx['<pad>'],
        vocab_size=len(args.word2idx),
        word_emb_dim=args.word_emb_dim,
        pretrained=args.pretrained,
        device=args.device
    ).to(args.device)

    # Load state dictionary
    # Remove incompatible keys
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt["state_dict"].items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    print("Done")

    # Run inference
    pred = infer_step(model, x)

    # Convert predictions to tags
    idx2tag = {v: k for (k, v) in args.tag2idx.items()}

    # Save predictions
    print("Saving predictions...", end=" ", flush=True)
    with open(args.save_path, 'w') as fp:
        for idx, doc in enumerate(idx_order):
            print(doc, end='\t', file=fp)
            p = list(map(lambda x: idx2tag[x], pred[idx]))
            print(*p, sep=',', file=fp)
    print("Done")

if __name__ == '__main__':
    main()
