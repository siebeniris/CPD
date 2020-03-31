# https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
import argparse
from pprint import pprint
from typing import List, Dict, Any, Tuple

import regex
import numpy as np
import pandas as pd
import seaborn as sns

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import lemmatize, simple_preprocess
from gensim.models import CoherenceModel
# plot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from wordcloud import WordCloud
# NLTK Stop words
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know',
                   'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather',
                   'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line',
                   'even', 'also', 'may', 'take', 'come'])

%matplotlib inline
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


def parse_args() -> Dict[str, Any]:
    """
    Parse command line arguments
    :return: Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description="LDA Model...")

    parser.add_argument("--input", type=str, help='input filename to process')
    parser.add_argument("--output", type=str, help="output filename")

    parser.add_argument("--process_corpus", action="store_true", help="To process corpus")
    parser.add_argument("--inspect_corpus", action="store_true", help="To inspect corpus")
    parser.add_argument("--inspect_file", action="store_true", help="To inspect file")

    return vars(parser.parse_args())


def sent_to_words(sentences: List[str]) -> str:
    """
    Replace in string.
    :param sentences: a list of sentences
    :return: generate sentence one by one
    """
    for sent in sentences:
        sent = regex.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = regex.sub('\s+', ' ', sent)  # remove newline chars
        sent = regex.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True)
        yield (sent)


def process_data_words(data_words: List[List[str]]) -> List[List[str]]:
    """
    Process documents, using bigrams and trigrams.
    :param data_words: a list of documents
    :return: a processed list of documents
    """

    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=30)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=30)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']

    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in data_words]
    texts = [bigram_mod[doc] for doc in texts]

    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []

    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]
    return texts_out


def build_model(data_ready: List[List[str]]) -> Any:
    """
    Build LDA Model
    :param data_ready: a list of documents
    :return: corpus, lda_model
    """
    id2word = corpora.Dictionary(data_ready)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_ready]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=4,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)

    pprint(lda_model.print_topics())
    return corpus, lda_model


def format_topics_sentences(ldamodel, corpus, texts):
    """
    In LDA models, each document is composed of multiple topics. But, typically only one of the topics is dominant.
    Extract this dominant topic for each sentence and shows the weight of the topic and the keywords in a nicely
    formatted output.
    Format dominant topics and its percentage distribution in each document.
    :param ldamodel:
    :param corpus:
    :param texts:
    :return:
    """
    # init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)


def representative_sentence_for_each_topic(df_topic_sents_keywords: pd.DataFrame):
    """
    Get samples of sentences that most represent a given topic.
    :param df_topic_sents_keywords:
    :return: None
    """
    pd.options.display.max_colwidth = 100

    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Perc_Contribution'], ascending=False).head(1)],
                                                axis=0)

    # Reset Index
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

    # Show
    print(sent_topics_sorteddf_mallet.head(10))


def frequency_distribution_word_counts_in_documents(df_dominant_topic):
    """
    Get frequency distribution of word counts in documents.
    :param df_dominant_topic: dataframe with domainant topics.
    :return: None
    """
    doc_lens = [len(d) for d in df_dominant_topic.Text]

    # Plot
    plt.figure(figsize=(16, 15), dpi=160)
    plt.hist(doc_lens, bins=1000, color='navy')
    plt.text(750, 50, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(750, 45, "Median : " + str(round(np.median(doc_lens))))
    plt.text(750, 40, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(750, 35, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(750, 30, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

    plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0, 1000, 9))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
    plt.show()


def distribution_of_document_word_counts_by_dominant_topic(df_dominant_topic: pd.DataFrame):
    """
    Show the distribution of document word counts by dominant topic.
    :param df_dominant_topic: dataframe for dominant topic
    :return: None
    """
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=160, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
        doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
        ax.hist(doc_lens, bins=1000, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
        ax.set(xlim=(0, 1000), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color=cols[i])
        ax.set_title('Topic: ' + str(i), fontdict=dict(size=16, color=cols[i]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0, 1000, 9))
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
    plt.show()


def word_clouds_top_n_keywords_by_topic(lda_model: gensim.models.LdaModel) -> None:
    """
    Show the word cloud with the size of the words proportional to the weight.

    :param lda_model: LDA Model
    :return: None
    """
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(stopwords=stop_words,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = lda_model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    plt.show()


def word_counts_of_topic_keywords(lda_model) -> None:
    """
    When it comes to the keywords in the topics, the importance of the keywords matters. Along with that, how
    frequently the words have appeared in the documents.
    Plot the word counts and the weights of each
    :param lda_model: LDA model
    :return: None
    """
    from collections import Counter
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in data_ready for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i, weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
               label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                    label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.030)
        ax.set_ylim(0, 3500)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
        ax.legend(loc='upper left')
        ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
    plt.show()


def sentence_chart_colored_by_topic(lda_model, corpus, start=0, end=13):
    corp = corpus[start:end]
    mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    fig, axes = plt.subplots(end - start, 1, figsize=(20, (end - start) * 0.95), dpi=160)
    axes[0].axis('off')
    for i, ax in enumerate(axes):
        if i > 0:
            corp_cur = corp[i - 1]
            topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
            word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]
            ax.text(0.01, 0.5, "Doc " + str(i - 1) + ": ", verticalalignment='center',
                    fontsize=16, color='black', transform=ax.transAxes, fontweight=700)

            # Draw Rectange
            topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
            ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1,
                                   color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

            word_pos = 0.06
            for j, (word, topics) in enumerate(word_dominanttopic):
                if j < 14:
                    ax.text(word_pos, 0.5, word,
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=16, color=mycolors[topics],
                            transform=ax.transAxes, fontweight=700)
                    word_pos += .009 * len(word)  # to move the word for the next iter
                    ax.axis('off')
            ax.text(word_pos, 0.5, '. . .',
                    horizontalalignment='left',
                    verticalalignment='center',
                    fontsize=16, color='black',
                    transform=ax.transAxes)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end - 2), fontsize=22, y=0.95,
                 fontweight=700)
    plt.tight_layout()
    plt.show()


# Sentence Coloring of N Sentences
def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)


def total_number_documents_per_topic(lda_model):
    dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)

    # Distribution of Dominant Topics in Each Document
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
    df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

    # Total Topic Distribution by actual weight
    topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

    # Top 3 Keywords for each Topic
    topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False)
                                     for j, (topic, wt) in enumerate(topics) if j < 3]

    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
    df_top3words.reset_index(level=0,inplace=True)
    return df_dominant_topic_in_each_doc, df_top3words, df_topic_weightage_by_doc


def plots_number_of_documents_topic_weightage(lda_model):

    df_dominant_topic_in_each_doc, df_top3words, df_topic_weightage_by_doc = total_number_documents_per_topic(lda_model)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)

    # Topic Distribution by Dominant Topics
    ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
    ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
    tick_formatter = FuncFormatter(
        lambda x, pos: 'Topic ' + str(x) + '\n' + df_top3words.loc[df_top3words.topic_id == x, 'words'].values[0])
    ax1.xaxis.set_major_formatter(tick_formatter)
    ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=10))
    ax1.set_ylabel('Number of Documents')
    ax1.set_ylim(0, 1000)

    # Topic Distribution by Topic Weights
    ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
    ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
    ax2.xaxis.set_major_formatter(tick_formatter)
    ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=10))

    plt.show()



if __name__ == '__main__':
    args = parse_args()

    # load data for each corpus (each csv file)
    df = pd.read_csv(args['input'])
    df["words"] = df['text'] + df['title']
    words_df = df[df.lang == 'en']
    words_df = words_df[words_df.words.notnull()]
    docs = words_df.words.to_list()  # => data
    # process the data words
    data_words = list(sent_to_words(docs))

    data_ready = process_data_words(data_words)
    # build lda model
    corpus, lda_model = build_model(data_ready)

    # format topics sentences

    df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    print(df_dominant_topic.head(10))

    # the most representative sentence for each topic
    ###################################################################
    representative_sentence_for_each_topic(df_topic_sents_keywords)
    ###################################################################

    # frequency_distribution_word_counts_in_documents(df_dominant_topic)
    #
    # distribution_of_document_word_counts_by_dominant_topic(df_dominant_topic)
    #
    # word_clouds_top_n_keywords_by_topic(lda_model)

    # word_counts_of_topic_keywords(lda_model)
    #
    # sentence_chart_colored_by_topic(lda_model,corpus)

    # plots_number_of_documents_topic_weightage(lda_model)

    # pyldavis_plot(lda_model)