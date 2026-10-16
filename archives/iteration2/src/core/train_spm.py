# tokenizer trainer 

import sentencepiece as spm
from pathlib import Path

class TokenizerTrainer:
    # trains a SP tokenizer on your corpus 
    def __init__(self, vocab_size = 16000):
        self.vocab_size = vocab_size

    def train(self, corpus_file, output_dir, model_name="tokenizer"):

        # create op direc
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # full path for model 
        model_prefix = output_path / model_name

        print("trining.........")
        print("-"*69)
        print(f"corpus :{corpus_file}")
        print(f"Vocab_size: {self.vocab_size}")
        print(f"model op:{model_prefix}.model")


        spm.SentencePieceTrainer.Train(
            input = str(corpus_file),
            model_prefix = str(model_prefix),

            # setting 
            model_type = "bpe", # tying bpe at first might change latter
            vocab_size = self.vocab_size,
            character_coverage  = 0.9995, #chtgpt said to put this 

            # special toksn (fixed)
            unk_id=0,  # Unknown token
            bos_id=1,  # Beginning of sequence
            eos_id=2,  # End of sequence  
            pad_id=3,  # Padding token

            # performance (specially for large data)
            shuffle_input_sentence=True,
            input_sentence_size=2000000,
            train_extremely_large_corpus=True,  

        )

        print("-"*69)
        print("Done traininng")
        print("Model Saved")
        print("vocab saved")

            




# tokenizer*( for using the trained model)

class NepaliTokenizer:

    def __init__(self, model_path):
        # load a train tokenizer 

        # checking if file exists 
        if not Path(model_path).exists():
            raise FileExistsError(f"FIle xina babu:{model_path}")
        
        # load spm model 
        self.sp = spm.SentencePieceProcessor()
        success = self.sp.Load(str(model_path))

        if not success:
            raise RuntimeError(f"Failed to laod model")
        
        # get special token IDS 
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

        print("TOkeinzer loaded")
        print(f"Vocab size: {self.vocab_size}")

    @property
    def vocab_size(self):
        return self.sp.GetPieceSize()
    
    def encode(self, text, add_bos= False, add_eos = False):
        # convert text to token ids 
        # text  Input : नमस्ते
        # add bos : add begnign token at stat 
        # add_eos : add end token at end
        # returns  token idss

        # get token ids from sp 
        ids = self.sp.EncodeAsIds(text)
        
        # Build result
        result = []
        
        # Add beginning token if requested
        if add_bos:
            result.append(self.bos_id)
        
        # Add all token IDs
        for token_id in ids:
            result.append(token_id)
        
        # Add end token if requested
        if add_eos:
            result.append(self.eos_id)
        
        return result
    

    def decode(self, ids, remove_special = True):
        # vice versa of encodder 
        # filter specal tokens if requurested 
        if remove_special:
            filtered = []
            for token_id in ids:
                # skipping special tokens 
                if token_id == self.bos_id:
                    continue
                if token_id == self.eos_id:
                    continue
                if token_id == self.pad_id:
                    continue

                # keeping regular tokens
                filtered.append(token_id)
            ids = filtered

        # decode to text 
        return self.sp.DecodeIds(ids)
    
    def encode_batch(self, texts, add_bos=False, add_eos=False):
        
        # Encode multiple texts at once.
        
        results = []
        for text in texts:
            encoded = self.encode(text, add_bos=add_bos, add_eos=add_eos)
            results.append(encoded)
        return results
    
    def decode_batch(self, ids_list, remove_special=True):
       
        # Decode multiple token sequences at once.
        results = []
        for ids in ids_list:
            decoded = self.decode(ids, remove_special=remove_special)
            results.append(decoded)
        return results
              
    


