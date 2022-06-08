import pandas as pd
from argparse import ArgumentParser
import emoji


def get_ema_columns():
    Q17_COLUMNS = ['loudness', 'cumberness', 'jawbone', 'neck', 'tin_day', 'tin_cumber', 'tin_max', 'movement',
                   'stress', 'emotion', 'diary_q11']
    return Q17_COLUMNS


def get_questionnaire_df(qestionnaire_json_df: pd.DataFrame, questionnaire_columns: list,
                         df_cols=['user_id', 'created_at']):
    '''
    Returns a dataframe with the columns specified in questionnaire_columns and the columns specified in questionnaire_columns
    :param qestionnaire_json_df:
    :param questionnaire_columns:
    :param df_cols:
    :return:
    '''

    def init_data_dict(col_list):
        cols_dict = dict()
        for col in col_list:
            cols_dict[col] = list()
        return cols_dict

    def get_complete_row_data_for_questionnaire(row, questionnaire_labels):
        row_data = dict()
        for answer in row['answers']:
            row_data[answer['label']] = answer['value']

        for key in questionnaire_labels:
            if key not in row_data:
                row_data[key] = None
        return row_data

    # all_cols = df_cols + questionnaire_columns

    all_data_dict = init_data_dict(df_cols + questionnaire_columns)

    for i in range(len(qestionnaire_json_df)):
        for col in df_cols:
            all_data_dict[col].append(qestionnaire_json_df.iloc[i][col])
        row_questionnaire_data = get_complete_row_data_for_questionnaire(qestionnaire_json_df.iloc[i],
                                                                         questionnaire_columns)
        for key in row_questionnaire_data:
            if key in questionnaire_columns:
                all_data_dict[key].append(row_questionnaire_data[key])

    df = pd.DataFrame.from_dict(all_data_dict)
    return df


def get_questionnaire_df(qestionnaire_json_df: pd.DataFrame, questionnaire_columns: list,
                         df_cols=['user_id', 'created_at']):
    def init_data_dict(col_list):
        cols_dict = dict()
        for col in col_list:
            cols_dict[col] = list()
        return cols_dict

    def get_complete_row_data_for_questionnaire(row, questionnaire_labels):
        row_data = dict()
        for answer in row['answers']:
            row_data[answer['label']] = answer['value']

        for key in questionnaire_labels:
            if key not in row_data:
                row_data[key] = None
        return row_data

    if isinstance(qestionnaire_json_df['answers'][0], str):
        qestionnaire_json_df['answers'] = qestionnaire_json_df['answers'].map(eval)

    all_data_dict = init_data_dict(df_cols + questionnaire_columns)

    for i in range(len(qestionnaire_json_df)):
        for col in df_cols:
            all_data_dict[col].append(qestionnaire_json_df.iloc[i][col])
        row_questionnaire_data = get_complete_row_data_for_questionnaire(qestionnaire_json_df.iloc[i],
                                                                         questionnaire_columns)
        for key in row_questionnaire_data:
            if key in questionnaire_columns:
                all_data_dict[key].append(row_questionnaire_data[key])

    df = pd.DataFrame.from_dict(all_data_dict)
    return df


def get_ema_df(raw_filepath: str, keep_test_users=False, keep_q11=False, drop_anonymous_users=True):
    '''
    Returns a data frame with the EMA data.
    :param raw_filepath: str
        The raw answersheets file downloaded from the database
    :param keep_test_users: bool, default False
        If True, then do not drop test users
    :param keep_q11: bool, default False
        If True, then do not drop the freetext EMA column
    :param drop_anonymous_users: bool, default True
        If True, then drop users that participated anonymously (user id > 42101)

    :return: pd.DataFrame
        Data frame with the EMA data
    '''
    raw = pd.read_csv(raw_filepath)
    raw = raw[raw.questionnaire_id == 17]
    q17_df = get_questionnaire_df(raw, get_ema_columns())
    q17_df.diary_q11 = q17_df.diary_q11.apply(lambda x: emoji.replace_emoji(x, "") if x is not None else None)

    if not keep_q11:
        q17_df.drop('diary_q11', axis=1, inplace=True)
    else:
        print(
            "This feature will not work because the column has some bad characters on row 423. it will not work until this is taken care of.")
    if not keep_test_users:
        q17_df = q17_df[q17_df.user_id >= 2101]
    if drop_anonymous_users:
        q17_df = q17_df[q17_df.user_id >= 42101]

    return q17_df


parser = ArgumentParser()
parser.add_argument("-r", "--raw-file-path", dest='raw_file_path', help='Path to the raw answersheets.csv from the DB')
parser.add_argument('--keep-test-users', action="store_true",
                    help="Test users will not be dropped from the output if this argument is provided")
parser.add_argument('--keep-q11', action="store_true",
                    help="Test users will not be dropped from the output if this argument is provided")
parser.add_argument('--drop-anonymous-users', action="store_true",
                    help="Users who participate in the app anonymously will be excluded if this argument is provided")
parser.add_argument('-o', '--output-file-path', dest='output_file_path', help='Path to the output file')

if __name__ == "__main__":
    command_line_args = parser.parse_args()

    raw = pd.read_csv(command_line_args.raw_file_path)
    raw = raw[raw.questionnaire_id == 17]
    q17_df = get_questionnaire_df(raw, get_ema_columns())
    q17_df.diary_q11 = q17_df.diary_q11.apply(lambda x: emoji.replace_emoji(x, "") if x is not None else None)
    if not command_line_args.keep_q11:
        q17_df.drop('diary_q11', axis=1, inplace=True)
    else:
        print(
            "This feature will not work because the column has some bad characters on row 423. it will not work until this is taken care of.")
    if not command_line_args.keep_test_users:
        q17_df = q17_df[q17_df.user_id >= 2101]
    if command_line_args.drop_anonymous_users:
        q17_df = q17_df[q17_df.user_id >= 42101]

    q17_df.to_csv(command_line_args.output_file_path, index=False)
