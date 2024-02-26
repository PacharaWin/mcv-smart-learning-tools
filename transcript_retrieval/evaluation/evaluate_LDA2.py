import time
import torch
from tqdm.auto import tqdm
from sentence_transformers import util
import pandas as pd
import numpy as np

from .metrics import hit_k, mean_hit_k, average_precision_k, mean_average_precision_k
from utils.constants import TRANSCRIPT_COL, EDITED_COL
from utils.helpers import extract_query_from_df


class Evaluation:
    def hit_k(self, predicted, actual, k):
        return hit_k(predicted, actual, k)

    def mean_hit_k(self, predicted_list, labeled_list, k=10):
        return mean_hit_k(predicted_list, labeled_list, k)

    def average_precision_k(self, predicted, actual, k):
        return average_precision_k(predicted, actual, k)

    def mean_average_precision_k(self, predicted_list, labeled_list, k=10):
        return mean_average_precision_k(predicted_list, labeled_list, k)

    def get_evaluate_report(self, predicted_list, labeled_list, k_list=[10, 5, 3, 1]):
        report_dict = {}
        for k in k_list:
            report_dict[f"mean_hit@{k}"] = self.mean_hit_k(
                predicted_list, labeled_list, k
            )
            report_dict[f"map@{k}"] = self.mean_average_precision_k(
                predicted_list, labeled_list, k
            )

        return report_dict
    
    import numpy as np
    import pandas as pd 

    def get_topic_distribution(self, lda_model, dataframe, dictionary, text_column):
            """
            Get topic distribution scores for each document in the DataFrame using the given LDA model.

            Parameters:
                lda_model (gensim.models.ldamodel.LdaModel): Gensim LDA model.
                dataframe (pandas.DataFrame): DataFrame containing the text column.
                text_column (str): Name of the text column in the DataFrame.

            Returns:
                pandas.DataFrame: DataFrame with an additional column containing topic distribution scores.
            """
            topic_scores = []

            # Initialize an empty list to store topic distribution scores for each document
            for index, row in dataframe.iterrows():
            # Convert the document text to bag-of-words representation using the dictionary
                bow = dictionary.doc2bow(row[text_column].split())
                
                # Infer the topic distribution for the document using the LDA model
                doc_topics = lda_model.get_document_topics(bow)
                
                # Extract the topic distribution scores from the result
                scores = np.zeros(lda_model.num_topics)  # Initialize scores array
                for topic, score in doc_topics:
                    scores[topic] = score
                
                # Append the topic scores to the list
                topic_scores.append(scores)
            
            # Add the topic distribution scores as a new column in the DataFrame
            dataframe['topic_distribution'] = topic_scores
            
            return dataframe



    def get_st_report(
        self,
        labeled_df,
        model,
        lda_model,
        dictionary,
        tokenizer_func,
        edited_text = False,
        k_list=[10, 5, 3, 1],
    ):
        """
        Generate evaluation report for SentenceTransformer model.
        """
        report_dict = {}
        query_dict = extract_query_from_df(labeled_df)

        print("Embedding context")
        if edited_text:
            corpus_embeddings = labeled_df[EDITED_COL]
        else:
            corpus_embeddings = labeled_df[TRANSCRIPT_COL]

        start_time = time.time()
        corpus_embeddings = model.encode(
            corpus_embeddings, convert_to_tensor=True, show_progress_bar=True
        )
        # print(corpus_embeddings.shape)
        end_time = time.time()
        
        report_dict["number_of_embedding_entry"] = len(labeled_df)
        report_dict["embedding_time"] = end_time - start_time
        report_dict["embedding_time_per_entry"] = (
            report_dict["embedding_time"] / report_dict["number_of_embedding_entry"]
        )
        #corpus add feature
        lda_feature = np.stack(labeled_df.topic_distribution.to_numpy())
        lda_feature = torch.tensor(lda_feature)
        corpus_embeddings = torch.cat((corpus_embeddings, lda_feature), dim = 1)
        print(corpus_embeddings)
        print("Predicting")
        for k in tqdm(k_list):
            predicted_list = []
            labeled_list = []
            for query_type_key, query_type_value in query_dict.items():
                for query in query_type_value:
                    column_name = query_type_key + "_" + query
                    query_embedding = model.encode(query, convert_to_tensor=True)
                    bow = dictionary.doc2bow(tokenizer_func(query))
                    doc_topics = lda_model.get_document_topics(bow)
                    
                    # Extract the topic distribution scores from the result
                    scores = np.zeros(lda_model.num_topics)  # Initialize scores array
                    for topic, score in doc_topics:
                        scores[topic] = score
                    doc_topics = scores
                    # print(query_embedding.shape)
                    # print( torch.tensor(doc_topics).shape)
                    query_embedding = torch.cat((query_embedding, torch.tensor(doc_topics)), dim = 0)


                    # print(query_embedding)
                    labeled = labeled_df[labeled_df[column_name] == 1].index.tolist()
                    labeled_list.append(labeled)

                    

                    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                    top_results = torch.topk(scores, k=k)

                    predicted = top_results.indices.tolist()
                    predicted_list.append(predicted)
                    
            print(predicted_list)
            print(labeled_list)

            report_dict[f"mean_hit@{k}"] = self.mean_hit_k(
                predicted_list, labeled_list, k
            )
            report_dict[f"map@{k}"] = self.mean_average_precision_k(
                predicted_list, labeled_list, k
            )
        return report_dict
