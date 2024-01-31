import time
import torch
from tqdm.auto import tqdm
from sentence_transformers import util

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

    def get_st_report(
        self,
        labeled_df,
        model,
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
        end_time = time.time()
        
        report_dict["number_of_embedding_entry"] = len(labeled_df)
        report_dict["embedding_time"] = end_time - start_time
        report_dict["embedding_time_per_entry"] = (
            report_dict["embedding_time"] / report_dict["number_of_embedding_entry"]
        )

        print("Predicting")
        for k in tqdm(k_list):
            predicted_list = []
            labeled_list = []
            for query_type_key, query_type_value in query_dict.items():
                for query in query_type_value:
                    column_name = query_type_key + "_" + query
                    query_embedding = model.encode(query, convert_to_tensor=True)
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
