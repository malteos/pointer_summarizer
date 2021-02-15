# Most of this file is copied form https://github.com/abisee/pointer-generator/blob/master/batcher.py
import json
import queue
import time
from random import shuffle
from threading import Thread
from typing import List

import numpy as np
import tensorflow as tf

import data_util.config as config
import data_util.data as data

import random

random.seed(1234)


class XingExample(object):

    # def __init__(self, article, abstract_sentences, vocab):
    def __init__(self, citing_context_tokens, cited_abstract_tokens, citation_text_tokens, vocab):

        # Get ids of special tokens
        start_decoding = vocab.word2id(data.START_DECODING)
        stop_decoding = vocab.word2id(data.STOP_DECODING)

        # Process the article
        # article_words = article.split()
        if len(citing_context_tokens) > config.max_enc_steps:
            citing_context_tokens = citing_context_tokens[:config.max_enc_steps]

        if len(cited_abstract_tokens) > config.max_enc_steps:
            cited_abstract_tokens = cited_abstract_tokens[:config.max_enc_steps]

        # store the length after truncation but before padding
        self.citing_context_len = len(citing_context_tokens)
        self.cited_abstract_len = len(cited_abstract_tokens)

        # list of word ids; OOVs are represented by the id for UNK token
        self.citing_context_token_ids = [vocab.word2id(w) for w in citing_context_tokens]
        self.cited_abstract_token_ids = [vocab.word2id(w) for w in cited_abstract_tokens]

        # Process the abstract
        # list of word ids; OOVs are represented by the id for UNK token
        citation_text_tokens_ids = [vocab.word2id(w) for w in citation_text_tokens]

        # Get the decoder input sequence and target sequence
        self.dec_input, self.target = self.get_dec_inp_targ_seqs(citation_text_tokens_ids, config.max_dec_steps, start_decoding,
                                                                 stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        if config.pointer_gen:
            # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
            self.citing_context_token_ids_extend_vocab, self.article_oovs = data.article2ids(citing_context_tokens, vocab, None)

            self.cited_abstract_token_ids_extend_vocab, self.article_oovs = data.article2ids(cited_abstract_tokens, vocab, self.article_oovs)

            # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
            citation_text_token_ids_extend_vocab = data.abstract2ids(citation_text_tokens, vocab, self.article_oovs)

            # Overwrite decoder target sequence so it uses the temp article OOV ids
            _, self.target = self.get_dec_inp_targ_seqs(citation_text_token_ids_extend_vocab, config.max_dec_steps, start_decoding,
                                                        stop_decoding)

        # Store the original strings
        # self.original_article = article
        # self.original_abstract = abstract
        # self.original_abstract_sents = abstract_sentences
        self.citing_context_tokens = citing_context_tokens
        self.cited_abstract_tokens = cited_abstract_tokens
        self.citation_text_tokens = citation_text_tokens


    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    # def pad_encoder_input(self, max_len, pad_id):
    #     while len(self.enc_input) < max_len:
    #         self.enc_input.append(pad_id)
    #     if config.pointer_gen:
    #         while len(self.enc_input_extend_vocab) < max_len:
    #             self.enc_input_extend_vocab.append(pad_id)

    def pad_citing_context_token_ids(self, max_len, pad_id):
        while len(self.citing_context_token_ids) < max_len:
            self.citing_context_token_ids.append(pad_id)
        if config.pointer_gen:
            while len(self.citing_context_token_ids_extend_vocab) < max_len:
                self.citing_context_token_ids_extend_vocab.append(pad_id)

    def pad_cited_abstract_token_ids(self, max_len, pad_id):
        while len(self.cited_abstract_token_ids) < max_len:
            self.cited_abstract_token_ids.append(pad_id)
        if config.pointer_gen:
            while len(self.cited_abstract_token_ids_extend_vocab) < max_len:
                self.cited_abstract_token_ids_extend_vocab.append(pad_id)


class XingBatch(object):
    def __init__(self, example_list, vocab, batch_size):
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(data.PAD_TOKEN)  # id of the PAD token used to pad sequences
        # self.init_encoder_seq(example_list)  # initialize the input to the encoder
        self.init_cited_abstract_seq(example_list)  # initialize the input to the encoder
        self.init_citing_context_seq(example_list)  # initialize the input to the encoder

        self.init_decoder_seq(example_list)  # initialize the input and targets for the decoder
        # self.store_orig_strings(example_list)  # store the original strings

    # def init_encoder_seq(self, example_list):
    #     # Determine the maximum length of the encoder input sequence in this batch
    #     max_enc_seq_len = max([ex.enc_len for ex in example_list])
    #
    #     # Pad the encoder input sequences up to the length of the longest sequence
    #     for ex in example_list:
    #         ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
    #
    #     # Initialize the numpy arrays
    #     # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
    #     self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
    #     self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
    #     self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)
    #
    #     # Fill in the numpy arrays
    #     for i, ex in enumerate(example_list):
    #         self.enc_batch[i, :] = ex.enc_input[:]
    #         self.enc_lens[i] = ex.enc_len
    #         for j in range(ex.enc_len):
    #             self.enc_padding_mask[i][j] = 1
    #
    #     # For pointer-generator mode, need to store some extra info
    #     if config.pointer_gen:
    #         # Determine the max number of in-article OOVs in this batch
    #         self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
    #         # Store the in-article OOVs themselves
    #         self.art_oovs = [ex.article_oovs for ex in example_list]
    #         # Store the version of the enc_batch that uses the article OOV ids
    #         self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
    #         for i, ex in enumerate(example_list):
    #             self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]
    def init_cited_abstract_seq(self, example_list: List[XingExample]):
        # Determine the maximum length of the encoder input sequence in this batch
        max_cited_abstract_len = max([ex.cited_abstract_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_cited_abstract_token_ids(max_cited_abstract_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.cited_abstract_batch = np.zeros((self.batch_size, max_cited_abstract_len), dtype=np.int32)
        self.cited_abstract_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.cited_abstract_padding_mask = np.zeros((self.batch_size, max_cited_abstract_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.cited_abstract_batch[i, :] = ex.cited_abstract_token_ids[:]
            self.cited_abstract_lens[i] = ex.cited_abstract_len
            for j in range(ex.cited_abstract_len):
                self.cited_abstract_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if config.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.cited_abstract_batch_extend_vocab = np.zeros((self.batch_size, max_cited_abstract_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.cited_abstract_batch_extend_vocab[i, :] = ex.cited_abstract_token_ids_extend_vocab[:]

    def init_citing_context_seq(self, example_list: List[XingExample]):
        # Determine the maximum length of the encoder input sequence in this batch
        max_citing_context_len = max([ex.citing_context_len for ex in example_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for ex in example_list:
            ex.pad_citing_context_token_ids(max_citing_context_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.citing_context_batch = np.zeros((self.batch_size, max_citing_context_len), dtype=np.int32)
        self.citing_context_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.citing_context_padding_mask = np.zeros((self.batch_size, max_citing_context_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.citing_context_batch[i, :] = ex.citing_context_token_ids[:]
            self.citing_context_lens[i] = ex.citing_context_len
            for j in range(ex.citing_context_len):
                self.citing_context_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        if config.pointer_gen:
            # Determine the max number of in-article OOVs in this batch
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            # Store the in-article OOVs themselves
            self.art_oovs = [ex.article_oovs for ex in example_list]
            # Store the version of the enc_batch that uses the article OOV ids
            self.citing_context_batch_extend_vocab = np.zeros((self.batch_size, max_citing_context_len), dtype=np.int32)
            for i, ex in enumerate(example_list):
                self.citing_context_batch_extend_vocab[i, :] = ex.citing_context_token_ids_extend_vocab[:]

    def init_decoder_seq(self, example_list):
        # Pad the inputs and targets
        for ex in example_list:
            ex.pad_decoder_inp_targ(config.max_dec_steps, self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.int32)
        self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros((self.batch_size), dtype=np.int32)

        # Fill in the numpy arrays
        for i, ex in enumerate(example_list):
            self.dec_batch[i, :] = ex.dec_input[:]
            self.target_batch[i, :] = ex.target[:]
            self.dec_lens[i] = ex.dec_len
            for j in range(ex.dec_len):
                self.dec_padding_mask[i][j] = 1

    def store_orig_strings(self, example_list):
        self.original_articles = [ex.original_article for ex in example_list]  # list of lists
        self.original_abstracts = [ex.original_abstract for ex in example_list]  # list of lists
        self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list]  # list of list of lists


class XingBatcher(object):
    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold

    def __init__(self, data_path, vocab, mode, batch_size, single_pass):
        self._data_path = data_path
        self._vocab = vocab
        self._single_pass = single_pass
        self.mode = mode
        self.batch_size = batch_size
        # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
        self._batch_queue = queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = queue.Queue(self.BATCH_QUEUE_MAX * self.batch_size)

        # Different settings depending on whether we're in single_pass mode or not
        if single_pass:
            self._num_example_q_threads = 1  # just one thread, so we read through the dataset just once
            self._num_batch_q_threads = 1  # just one thread to batch examples
            self._bucketing_cache_size = 1  # only load one batch's worth of examples before bucketing; this essentially means no bucketing
            self._finished_reading = False  # this will tell us when we're finished reading the dataset
        else:
            self._num_example_q_threads = 1  # 16 # num threads to fill example queue
            self._num_batch_q_threads = 1  # 4  # num threads to fill batch queue
            self._bucketing_cache_size = 1  # 100 # how many batches-worth of examples to load into cache before bucketing

        # Start the threads that load the queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        # Start a thread that watches the other threads and restarts them if they're dead
        if not single_pass:  # We don't want a watcher in single_pass mode because the threads shouldn't run forever
            self._watch_thread = Thread(target=self.watch_threads)
            self._watch_thread.daemon = True
            self._watch_thread.start()

    def next_batch(self):
        # If the batch queue is empty, print a warning
        if self._batch_queue.qsize() == 0:
            tf.logging.warning(
                'Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())
            if self._single_pass and self._finished_reading:
                tf.logging.info("Finished reading dataset in single_pass mode.")
                return None

        batch = self._batch_queue.get()  # get the next Batch
        return batch

    def fill_example_queue(self):
        # input_gen = self.text_generator(data.example_generator(self._data_path, self._single_pass))
        # TODO single pass?
        with open(self._data_path) as file_handler:
            for line in file_handler:
                sample = json.loads(line)

                example = XingExample(sample['citing_context_tokens'], sample['cited_abstract_tokens'], sample['citation_text_tokens'], self._vocab)
                self._example_queue.put(example)  # place the Example in the example queue.


        # while True:
        #     try:
        #         (article, abstract) = next(
        #             input_gen)  # read the next example from file. article and abstract are both strings.
        #     except StopIteration:  # if there are no more examples:
        #         tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
        #         if self._single_pass:
        #             tf.logging.info(
        #                 "single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
        #             self._finished_reading = True
        #             break
        #         else:
        #             raise Exception("single_pass mode is off but the example generator is out of data; error.")
        #
        #     # print(abstract)
        #     abstract_sentences = [sent.strip() for sent in data.abstract2sents(
        #         abstract)]  # Use the <s> and </s> tags in abstract to get a list of sentences.
        #
        #     # print(abstract_sentences)
        #     example = XingExample(article, abstract_sentences, self._vocab)  # Process into an Example.
        #     self._example_queue.put(example)  # place the Example in the example queue.


    def fill_batch_queue(self):
        while True:
            if self.mode == 'decode':
                # beam search decode mode single example repeated in the batch
                ex = self._example_queue.get()
                b = [ex for _ in range(self.batch_size)]
                self._batch_queue.put(XingBatch(b, self._vocab, self.batch_size))
            else:
                # Get bucketing_cache_size-many batches of Examples into a list, then sort
                inputs = []  # Type: XingExample
                for _ in range(self.batch_size * self._bucketing_cache_size):
                    inputs.append(self._example_queue.get())
                inputs = sorted(inputs, key=lambda inp: inp.citing_context_len, reverse=True)  # sort by length of encoder sequence TODO how to sort two sequences?

                # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
                batches = []
                for i in range(0, len(inputs), self.batch_size):
                    batches.append(inputs[i:i + self.batch_size])
                if not self._single_pass:
                    shuffle(batches)
                for b in batches:  # each b is a list of Example objects
                    self._batch_queue.put(XingBatch(b, self._vocab, self.batch_size))

    def watch_threads(self):
        while True:
            tf.logging.info(
                'Bucket queue size: %i, Input queue size: %i',
                self._batch_queue.qsize(), self._example_queue.qsize())

            time.sleep(60)
            for idx, t in enumerate(self._example_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found example queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()
            for idx, t in enumerate(self._batch_q_threads):
                if not t.is_alive():  # if the thread is dead
                    tf.logging.error('Found batch queue thread dead. Restarting.')
                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def text_generator(self, example_generator):
        while True:
            try:
                e = next(example_generator)  # e is a tf.Example
                try:
                    article_text = e.features.feature['article'].bytes_list.value[
                        0]  # the article text was saved under the key 'article' in the data files
                    abstract_text = e.features.feature['abstract'].bytes_list.value[
                        0]  # the abstract text was saved under the key 'abstract' in the data files
                except ValueError:
                    tf.logging.error('Failed to get article or abstract from example')
                    continue
                if len(article_text) == 0:  # See https://github.com/abisee/pointer-generator/issues/1
                    # tf.logging.warning('Found an example with empty article text. Skipping it.')
                    continue
                else:
                    article_text = article_text.decode('utf-8')
                    abstract_text = abstract_text.decode('utf-8')

                    yield (article_text, abstract_text)
            except StopIteration:
                print('stop iteration...')
                break
