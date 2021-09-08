import glob
import os
import re

import nltk
from nltk import sent_tokenize, word_tokenize, ne_chunk, Tree
from nltk.corpus import stopwords

import pandas as pd


def csv_generate(report_data):
    """
    This method generates the statistic report for nltk analysis
    """

    spec_chars = ["!", "#", "%", "&", "'", "(", ")",
                  "*", "+", ",", "-", ".", "/", ":", ";", "<",
                  "=", ">", "?", "@", "[", "\\", "]", "^", "_",
                  "`", "{", "|", "}", "~", "–"]

    my_file = pd.DataFrame(report_data)

    for char in spec_chars:
        my_file['sentence_list'] = my_file['sentence_list'].str.replace(char, ' ')
        my_file['sentence_list'] = my_file['sentence_list'].str.split().str.join(" ")

    return my_file


def extract_noun_phrase(filtered_post):
    """
    This method extracts noun phrases from given tokens

    filtered_post: word tokens with its post
    """
    pattern = 'NP: {<DT>?<JJ>*<NN>}'  # Chunking pattern
    chunk_result = nltk.RegexpParser(pattern)
    result = chunk_result.parse(filtered_post)

    try:
        parse_tree = Tree.fromstring(str(result))

        final_result = [" ".join([leaf.split('/')[0] for leaf in subtree.leaves()]) for subtree in parse_tree if
                        type(subtree) == Tree and subtree.label() == "NP"]

        return final_result

    except Exception as e:
        print(e)


def extract_verb_phrase(all_tokens):
    """
    This method extracts verb phrases from given tokens

    all_tokens: unfiltered tokens with its post
    """
    v_pattern = "VP: {<VB>?<VBD>?<VBG>?<VBN>?<VBP>?<VBZ>?<VBG>*<VBN>*}"  # Chunking pattern
    chunk_result = nltk.RegexpParser(v_pattern)
    result = chunk_result.parse(all_tokens)

    try:
        parse_tree = Tree.fromstring(str(result))

        final_result = [" ".join([leaf.split('/')[0] for leaf in subtree.leaves()]) for subtree in parse_tree if
                        type(subtree) == Tree and subtree.label() == "VP"]

        return final_result

    except Exception as e:
        print(e)


def sentence_extraction(filename, data, method_call_cnt):
    """
    This method extracts sentence from the given text file.

    Keyword arguments:
    output_path -- output path for the file
    filename -- name of the file
    data -- file content

    """

    sent_list, noun_phrases_list, verb_phrases_list, label_list, ner_rich_list = [], [], [], [], []
    new_df = pd.DataFrame()

    output_file_name = filename + '_process.txt'
    # output_file_path = output_path + output_file_name

    stop_words = set(stopwords.words('english'))

    pattern = '\se.g.\s'

    data = data.replace('\n\n', ' ')  # Replacing extra new lines.
    data = data.replace('\n', ' ')
    data = re.sub(pattern, ' e.g.- ', data)  # Replacing whitespace and newline.

    res = sent_tokenize(data)  # Sentence tokenization

    total_tokens, total_filtered_tokens = 0, 0
    total_np, total_vp = 0, 0

    try:
        sentence_cnt = 0
        for s in res:
            # Words tokenization
            tokens = word_tokenize(s)
            total_tokens += len(tokens)

            # Filtering tokens
            filtered_result = [w for w in tokens if not w.lower() in stop_words]
            total_filtered_tokens += len(filtered_result)

            # POS tagging for filtered tokens
            filtered_post = nltk.pos_tag(filtered_result)

            # Noun phrase chunks
            np = extract_noun_phrase(filtered_post)
            total_np += len(np)

            if np is not None:
                nps = set(np)
                nps_len = len(nps)
                total_vp += nps_len
            else:
                nps_len = 0

            # Verb phrase chunks
            vp = extract_verb_phrase(filtered_post)

            if vp is not None:
                vps = set(vp)
                vps_len = len(vps)
                total_vp += vps_len
            else:
                vps_len = 0

            sentence_cnt += 1

            sent_list.append(s)
            noun_phrases_list.append(nps_len)
            verb_phrases_list.append(vps_len)

            if method_call_cnt <= 10:
                label_list.append('TRAIN')
            else:
                label_list.append('TEST')

            if nps_len and vps_len >= 1:
                ner_rich_list.append(1)
            else:
                ner_rich_list.append(0)

        report_dict = {
                       'sentence_list': sent_list,
                       'np_list': noun_phrases_list,
                       'vp_list': verb_phrases_list,
                       'ner_rich': ner_rich_list,
                       'label': label_list
                    }

        new_df = csv_generate(report_dict)

        print(f'\n\n {output_file_name} Processing Ends ==> {sentence_cnt}')
        print(f'\n {"--------" * 20}')
        return new_df

    except Exception as e:
        print(e.args)

    return new_df


if __name__ == '__main__':
    final_ds = pd.DataFrame()

    method_call_cnt = 0

    i_path = input('Enter The Input Path: ')  # Get user's input files directory path

    i_path_exist = os.path.exists(i_path)  # Check the user input path

    if i_path_exist:
        input_path = i_path + '*'

        for file in glob.glob(input_path):
            method_call_cnt += 1
            fname = os.path.basename(file)
            filename, ext = os.path.splitext(fname)
            new_i_path = i_path + fname

            with open(new_i_path, "r") as fr:
                lines = fr.read()

            new_df = sentence_extraction(filename, lines, method_call_cnt)

            final_ds = pd.concat([final_ds, new_df], ignore_index=True)

            final_ds.to_csv('output2/final_csv_6.csv', index=True, index_label='SR_NO')
    else:
        print("\n \n INVALID PATHS ")

