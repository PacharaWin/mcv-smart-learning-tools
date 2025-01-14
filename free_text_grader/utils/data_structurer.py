from .constants import (
    ASSIGNMENT_ID_COL,
    QUESTION_NUMBER_COL,
    QUESTION_COL,
    PATTERN_COL,
    PATTERN_NUMBER_COL,
)


class DataStructurer:
    def __init__(self, score_df, pattern_map_df):
        self.score_df = score_df
        self.pattern_map_df = pattern_map_df

    def __len__(self):
        return len(self.score_df)

    def get_question(self, assignment_id, question_number):
        if isinstance(assignment_id, int):
            assignment_id = str(assignment_id)

        filtered_df = self.pattern_map_df[
            (self.pattern_map_df[ASSIGNMENT_ID_COL] == assignment_id)
            & (self.pattern_map_df[QUESTION_NUMBER_COL] == question_number)
        ]

        if not filtered_df.empty:
            return filtered_df[QUESTION_COL].iloc[0]
        else:
            return None

    def get_pattern(self, assignment_id, question_number, pattern_number=-1):
        if isinstance(assignment_id, int):
            assignment_id = str(assignment_id)

        filtered_df = self.pattern_map_df[
            (self.pattern_map_df[ASSIGNMENT_ID_COL] == assignment_id)
            & (self.pattern_map_df[QUESTION_NUMBER_COL] == question_number)
        ]

        if not filtered_df.empty:
            if pattern_number != -1:
                return filtered_df[filtered_df[PATTERN_NUMBER_COL] == pattern_number][
                    PATTERN_COL
                ].iloc[0]
            else:
                return filtered_df[PATTERN_COL]
        else:
            return None

    def __getitem__(self, idx):
        return self.score_df.iloc[idx]
