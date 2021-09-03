import os
import pickle
from utils.preprocess import *
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

class Vocabulary(object):
    """
    Build custom vocabulary from scratch
    """
    def __init__(self,
        max_size=None,
        min_freq=None,
        start_word="<bos>",
        end_word="<eos>",
        unk_word="<unk>",
        pad_word="<pad>"):
        
        self.min_freq = min_freq
        self.max_size = max_size
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_word = pad_word
        self._init_vocab()

    def special_tokens_map_extended(self):
        """
        Return special map tokens
        """
        return self._special_tokens

    def _init_vocab(self):
        """
        Initialize the dictionaries for converting tokens to integers (and vice-versa).
        """
        self._word2idx = {}
        self._idx2word = {}
        self.freqs = {}
        self.vocab_size = 0

        self._add_word(self.pad_word)
        self._add_word(self.start_word)
        self._add_word(self.end_word)
        self._add_word(self.unk_word)

        self.start_word_idx = self.stoi(self.start_word)
        self.end_word_idx = self.stoi(self.end_word)
        self.unk_word_idx = self.stoi(self.unk_word)
        self.pad_word_idx = self.stoi(self.pad_word)

        self._special_tokens = {
            'bos_token': self.start_word,
            'cls_token': self.start_word,
            'eos_token': self.end_word,
            'sep_token': self.end_word,
            'pad_token': self.pad_word,
            'unk_token': self.unk_word,
        }

        self._special_ids = {
            'bos_token_id': self.start_word_idx,
            'cls_token_id': self.start_word_idx,
            'eos_token_id': self.end_word_idx,
            'sep_token_id': self.end_word_idx,
            'pad_token_id': self.pad_word_idx,
            'unk_token_id': self.unk_word_idx,
        }

        self.cls_token_id = self.bos_token_id = self.start_word_idx
        self.eos_token_id = self.sep_token_id = self.end_word_idx
        self.pad_token_id = self.pad_word_idx
        self.unk_token_id = self.unk_word_idx

        self.cls_token = self.bos_token = self.start_word
        self.eos_token = self.sep_token = self.end_word
        self.pad_token = self.pad_word
        self.unk_token = self.unk_word


    def _add_word(self, word):
        """
        Add a token to the vocabulary.
        """
        if not word in self._word2idx.keys():
            self._word2idx[word] = self.vocab_size
            self.freqs[word] = 0
            self._idx2word[self.vocab_size] = word
            self.vocab_size += 1
        self.freqs[word] += 1

    @staticmethod
    def from_pickle(pkl):
        """
        Load the vocabulary from pickle file
        """
        assert os.path.exists(pkl), f"{pkl} not exists"
        with open(pkl, 'rb') as f:
            vocab = pickle.load(f)
        
        return vocab

    @staticmethod
    def from_coco_json(json_file, max_size=None, min_freq=None, tokenizer=None):
        """
        Build vocabulary from JSON file in COCO format 
        """

        assert os.path.exists(json_file), f"{json_file} not exists"
        coco = COCO(json_file)

        image_ids = coco.getImgIds()

        texts = []
        for image_id in image_ids:
            ann_ids = coco.getAnnIds(imgIds=image_id)
            anns = coco.loadAnns(ann_ids)
            texts += [i['caption'] for i in anns]

        return Vocabulary.from_list(
            texts, max_size=max_size, 
            min_freq=min_freq, tokenizer=tokenizer)

    @staticmethod
    def from_list(texts, max_size=None, min_freq=None, tokenizer=None):
        """
        Vocabulary from list of texts
        """
        
        if tokenizer is None:
            tokenizer = Preprocess([
                Consecutive(),
                RemoveEmoji(),
                WordTokenizer()
            ])

        vocab = Vocabulary(
            max_size=max_size,
            min_freq=min_freq)

        token_lst = tokenizer(texts)
        vocab.build_vocab(token_lst)

        return vocab

    def build_vocab(self, lst_tokens):
        """
        Build vocab from list of tokens, limit to min_freq and max_size
        """
        freqs = {}

        # Calculate frequency of words
        for tokens in lst_tokens:
            for word in tokens:
                if word not in self._special_tokens.values() and word not in freqs.keys():
                    freqs[word] = 0
                freqs[word] += 1

        # Sort words by frequency
        sorted_freqs = {k: v for k, v in sorted(freqs.items(), key=lambda item: item[1], reverse=True)}

        # Max size
        if self.max_size is not None:
            sorted_freqs =  {k: v for i, (k, v) in enumerate(sorted_freqs.items()) if i < self.max_size}

        # Filter low frequency words
        if self.min_freq is not None:
            sorted_freqs =  {k: v for k, v in sorted_freqs.items() if v >= self.min_freq}

        for word, freq in sorted_freqs.items():
            self._word2idx[word] = self.vocab_size
            self._idx2word[self.vocab_size] = word
            self.vocab_size += 1
            self.freqs[word] = freq


    def convert_ids_to_tokens(self, tok_ids):
        """
        Convert token ids to texts
        """
        result = []
        for tok in tok_ids:
            word = self.itos(tok)
            result.append(word)
        return result

    def itos(self, idx):
        if not idx in self._idx2word.keys():
            return self._idx2word[self.unk_word_idx]
        return self._idx2word[idx]

    def stoi(self, word):
        if not word in self._word2idx.keys():
            return self._word2idx[self.unk_word]
        return self._word2idx[word]

    def get_vocab(self):
        return self._word2idx

    def get_freqs(self):
        return self.freqs

    def save_pickle(self, path):
        """
        Save vocab to pickle
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def convert_tokens_to_ids(self, tokens, max_len=None):
        """
        Convert list of tokens to ids
        """
        if max_len is not None:
            token_length = len(tokens)
            if max_len < token_length:
                tokens = tokens[:max_len]
            else:
                for _ in range(max_len - token_length):
                    tokens.append(self.pad_token())
        return [self.stoi(tok) for tok in tokens]
    
    def tokenize(self, texts, max_len=None):
        result = []
        for text in texts:
            tokens = word_tokenize(text)
            ids = self.convert_tokens_to_ids(tokens, max_len)
            result.append(ids)

        return result

    def __call__(self, texts, max_len=None):
        return self.tokenize(texts, max_len=max_len)
        
    def __len__(self):
        return len(self._word2idx)

    def __str__(self) -> str:
        s = f"Vocabulary size: {self.vocab_size}"
        return s

    def most_common(self, topk = None, ngrams = None):
        """
        Return a dict of most common words
        
        Args:
            topk: Top K words
            ngrams: string
                '1grams': unigram
                '2grams': bigrams
                '3grams': trigrams
                
        """
        
        if topk is None:
            topk = self.max_size
        idx = 0
        common_dict = {}
        
        if ngrams is None:
            for token, freq in self.freqs.items():
                if idx >= topk:
                    break
                common_dict[token] = freq
                idx += 1
        else:
            if ngrams == "1gram":
                for token, freq in self.freqs.items():
                    if idx >= topk:
                        break
                    if len(token.split()) == 1:
                        common_dict[token] = freq
                        idx += 1
            if ngrams == "2grams":
                for token, freq in self.freqs.items():
                    if idx >= topk:
                        break
                    if len(token.split()) == 2:
                        common_dict[token] = freq
                        idx += 1
            if ngrams == "3grams":
                for token, freq in self.freqs.items():
                    if idx >= topk:
                        break
                    if len(token.split()) == 3:
                        common_dict[token] = freq
                        idx += 1
                
            
        return common_dict

    def plot(self, types = None, topk = 100, figsize = (8,8) ):
        """
        Plot distribution of tokens:
            types: list
                "freqs": Tokens distribution
                "allgrams": Plot every grams
                "1gram - 2grams - 3grams" : Plot n-grams
        """
        ax = plt.figure(figsize = figsize)
        if types is None:
            types = ["freqs", "allgrams"]
        
        if "freqs" in types:
            if "allgrams" in types:
                plt.title("Top " + str(topk) + " highest frequency tokens")
                plt.xlabel("Unique tokens")
                plt.ylabel("Frequencies")
                cnt_dict = self.most_common(topk)
                bar1 = plt.barh(list(cnt_dict.keys()),
                                list(cnt_dict.values()),
                                color="blue")
            else:
                if "1gram" in types:
                    plt.title("Top " + str(topk) + " highest frequency unigram tokens")
                    plt.xlabel("Unique tokens")
                    plt.ylabel("Frequencies")
                    cnt_dict = self.most_common(topk, "1gram")
                    bar1 = plt.barh(list(cnt_dict.keys()),
                                    list(cnt_dict.values()),
                                    color="blue",
                                    label = "Unigrams")

                if "2grams" in types:
      
                    plt.title("Top " + str(topk) + " highest frequency bigrams tokens")
                    plt.xlabel("Unique tokens")
                    plt.ylabel("Frequencies")
                    cnt_dict = self.most_common(topk, "2grams")
                    bar1 = plt.barh(list(cnt_dict.keys()),
                                    list(cnt_dict.values()),
                                    color="gray",
                                    label = "Bigrams")

                if "3grams" in types:
                    plt.title("Top " + str(topk) + " highest frequency trigrams tokens")
                    plt.xlabel("Unique tokens")
                    plt.ylabel("Frequencies")
                    cnt_dict = self.most_common(topk, "3grams")
                    bar1 = plt.barh(list(cnt_dict.keys()),
                                    list(cnt_dict.values()),
                                    color="green",
                                    label = "Trigrams") 
            
        plt.legend()
        plt.show()

if __name__ == '__main__':
    vocab = Vocabulary.from_coco_json('./val.json', max_size=5000)
    vocab.save_pickle('vocab.pkl')
    texts = ["Various glass items are displayed on a glass shelf in a window .",
    "A man trying to shoo away a mongoose with a broom , while a man watches from behind ."]

    outputs = vocab(texts, max_len=32)
    print(outputs[1])
    
    