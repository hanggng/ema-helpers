import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from sklearn.preprocessing import StandardScaler

class DataEMAS(object):

    #Data variables
    user_list = []
    question_list = []

    # Load raw data
    def __init__(self, emas_file):
        """

        :param emas_file: a string which represents the file path
        """
        self.df = pd.read_csv(emas_file, index_col=0)
        self.df.ts = pd.to_datetime(self.df.ts)

    # Load daily data (data type changed to float because of NaN)
    def get_daily(self):
        """

        :return: daily data (data grouped by date)
        """
        self.df['date'] = self.df.ts.dt.date
        daily = self.df.reset_index().set_index('date').sort_index()
        idx = pd.Series(np.nan, index=pd.date_range(daily.index.min(), daily.index.max(), freq='D'))
        daily = pd.concat([daily, idx[~idx.index.isin(daily.index)]]).sort_index()
        daily.index.name = 'date'
        daily = daily.set_index([daily.index, 'ts']).sort_index().drop(0, axis=1)
        return daily

    # Get_question_data
    def get_ques(self, question_list: list):
        """

        :param question_list: list of questions
        :return: data of some specific questions in question list
        """
        return self.df[self.df.columns[self.df.columns.isin(question_list)]]


    # Get all data for users (EMA) or users in lists
    def get_users_in_list(self, user_list: list):
        """

        :param user_list: list of users
        :return: data of some specific users in user list
        """
        return self.df[self.df.index.isin(user_list)]

    # Get users with sufficient data at least n_days
    def user_n_days(self, n: int):
        """

        :param n: number of days
        :return: data of users with sufficient data at least n_days
        """
        self.df = self.df.dropna().drop_duplicates()
        self.df['date'] = self.df.ts.dt.date
        return self.df[self.df.groupby(['entity'])['date'].nunique() >= n]

    # Get users with sufficient data at least n_sessions
    def user_n_sessions(self, n: int):
        """

        :param n: number of days
        :return: data of users with sufficient data at least n_sessions
        """
        self.df = self.df.dropna().drop_duplicates()
        return self.df[self.df.groupby(['entity'])['ts'].nunique() >= n]

    # Get users with sufficient data at least n_sessions per day
    def user_n_sessions_per_day(self, n: int):
        """

        :param n: number of days
        :return: data of users with sufficient data at least n_sessions per day
        """
        self.df = self.df.dropna().drop_duplicates()
        return self.df[self.df.groupby(['entity','date'])['ts'].nunique() >= n]

    # Get longest uninterrupted sequence all_users_in_set
    def max_sequence(self, user_list: list):
        """

        :param user_list: list of users
        :return: longest uninterrupted sequence of each users in user list
        """
        sequence_list = []
        for x in user_list:
            user_df = self.df.loc[[x], ['ts']]
            user_df['date'] = user_df.ts.dt.date
            user_df['mask'] = 1
            user_df.loc[user_df['date'] - timedelta(days=1) == user_df['date'].shift(), 'mask'] = 0
            user_df['mask'] = user_df['mask'].cumsum()
            sequence = user_df.loc[user_df['mask'] == user_df['mask'].value_counts().idxmax(), ['date']]
            sequence = sequence.groupby(sequence.index)['date'].apply(list).to_dict()
            sequence_list.append(sequence)
        return sequence_list

    # Get longest sequence with max_gap_size <= G
    def max_sequence_gap(self, user_list: list, n: int):
        """

        :param user_list: list of users
        :param n: gap size counted in day
        :return: longest sequence of each user in user list with max_gap_size <= n
        """
        sequence_list = []
        for x in user_list:
            user_df = self.df.loc[[x], ['ts']]
            user_df['date'] = user_df.ts.dt.date
            user_df['mask'] = 1
            user_df.loc[user_df['date'] - user_df['date'].shift() <= timedelta(days=n), 'mask'] = 0
            user_df['mask'] = user_df['mask'].cumsum()
            sequence = user_df.loc[user_df['mask'] == user_df['mask'].value_counts().idxmax(), ['date']]
            sequence = sequence.groupby(sequence.index)['date'].apply(list).to_dict()
            sequence_list.append(sequence)
        return sequence_list

    # Get all uninterrupted sequences
    def all_uni_seq(self, user_list):
        """

        :param user_list: list of users
        :return: all uninterrupted sequences of each user in user list
        """
        all_uni_seq = []
        for x in user_list:
            user_df = self.df.loc[[x], ['ts']]
            user_df['date'] = user_df.ts.dt.date
            user_df['mask'] = 1
            user_df.loc[user_df['date'] - user_df['date'].shift() <= timedelta(days=1), 'mask'] = 0
            user_df['mask'] = user_df['mask'].cumsum()
            all_seq = user_df.groupby(['mask', 'entity'])['date'].unique().to_frame()  # .tolist()
            all_seq = all_seq.reset_index().drop('mask', axis=1).set_index('entity')
            all_seq = all_seq[all_seq.date.map(len) > 1]
            all_seq = all_seq.groupby(all_seq.index)['date'].apply(list).to_dict()
            all_uni_seq.append(all_seq)
        return all_uni_seq

    # Get all sequences with max_gap_size <= G
    def all_uni_seq_gap(self, user_list: list, n: int):
        """

        :param user_list: list of user
        :param n: gap size counted in day
        :return: all sequences of each users in user list with max_gap_size <= n
        """
        all_uni_seq = []
        for x in user_list:
            user_df = self.df.loc[[x], ['ts']]
            user_df['date'] = user_df.ts.dt.date
            user_df['mask'] = 1
            user_df.loc[user_df['date'] - user_df['date'].shift() <= timedelta(days=n), 'mask'] = 0
            user_df['mask'] = user_df['mask'].cumsum()
            all_seq = user_df.groupby(['mask', 'entity'])['date'].unique().to_frame()  # .tolist()
            all_seq = all_seq.reset_index().drop('mask', axis=1).set_index('entity')
            all_seq = all_seq[all_seq.date.map(len) > 1]
            all_seq = all_seq.groupby(all_seq.index)['date'].apply(list).to_dict()
            all_uni_seq.append(all_seq)
        return all_uni_seq

    # Standardise user_level_emas (sklearn.preprocessing.StandardScaler) scaler on each user in all data
    def standard_scaler_all(self):
        """

        :return: standardized data of all users
        """
        scaler = StandardScaler()
        self.df = self.df.drop(['ts'], axis=1)
        scaler.fit(self.df)
        scaled_features = scaler.transform(self.df)
        df_feat = pd.DataFrame(scaled_features, index=self.df.index, columns=self.df.columns)
        return df_feat

    # Standardise user_level_emas (sklearn.preprocessing.StandardScaler) scaler on each user in user list
    def standard_scaler_list(self, user_list: list):
        """

        :param user_list: list of users
        :return: standardised data of some specific users in user list
        """
        scaler = StandardScaler()
        user_df = self.df[self.df.index.isin(user_list)].drop(['ts'], axis=1).sort_index()
        scaler.fit(user_df)
        scaled_features = scaler.transform(user_df)
        df_feat = pd.DataFrame(scaled_features, index=user_df.index, columns=user_df.columns)
        return df_feat

    # Get user EMA with time-since-last-ema as new column
    def time_since_last_ema(self, user_list: list):
        """

        :param user_list: list of users
        :return: data with time-since-last-ema as new column
        """
        for x in user_list:
            user_df = self.df.loc[[x]].sort_index()
            user_df['date'] = user_df.ts.dt.date
            user_df['time_since_last_ema'] = user_df['date'] - user_df['date'].shift()
            return user_df

    # Get time series data in a way that is compatible with Kats:
    def get_Kats(self):
        """

        :return: time series data in a way that is compatible with Kats
        """
        return self.df.groupby(self.df.index)['ts'].apply(list).to_dict()


    # Relative time
    def rel_time(self, user_list: list):
        """

        :param user_list: list of users
        :return: relative time for each user in user list in a new column
        """
        self.df['date'] = self.df.ts.dt.date
        for x in user_list:
            user_df = self.df.loc[[x]]
            user_df['user_rel_time'] = user_df['date'] - user_df['date'].iloc[0]
            return user_df

    # Get user data with gaps encoded using some given value (input: Encode gaps as 1000, for eg)
    def gap_encoded(self, user_list: list, value: int):
        """

        :param user_list: list of users
        :param value: value for encoding
        :return: user data with gaps encoded using a given value
        """
        self.df['date'] = self.df.ts.dt.date
        for x in user_list:
            user_df = self.df.loc[[x]]
            user_df = user_df.reset_index().set_index('date').sort_index()
            idx = pd.Series(np.nan, index=pd.date_range(user_df.index.min(), user_df.index.max(), freq='D').date)
            user_df = pd.concat([user_df, idx[~idx.index.isin(user_df.index)]]).sort_index()
            user_df.index.name = 'date'
            user_df.entity = x
            user_df = user_df.set_index(['entity', user_df.index]).sort_index().drop(0, axis=1)
            user_df = user_df.fillna(value)
            return user_df

    # Get N-most-similar users with Input user-distance matrix
    def kNN_user(self, dist_matrix: pd.DataFrame, k: int):
        """

        :param dist_matrix: distance matrix
        :param k: number of similar users
        """
        dist_matrix = dist_matrix.T
        for i in dist_matrix.index:
            k_matrix = dist_matrix.nsmallest(k, i)
            k_matrix = k_matrix.T
            k_matrix = k_matrix[k_matrix.index == i]
            k_matrix = k_matrix.loc[:, k_matrix.any()]
            for j in k_matrix.index:
                print(i , ": ", list(k_matrix.columns))

class DataLoaderEMAS(DataEMAS):

    #Data variables
    user_list = []
    question_list = []

    # Load raw data
    def __init__(self, emas_file, user):
        super().__init__(emas_file)
        self.df = self.df.loc[[user]]
        print(self.df)

class DataStatic(object):

    #Data variables
    user_list = []
    question_list = []

    # Load raw data
    def __init__(self, static_file):
        self.df = pd.read_csv(static_file, index_col=0)
        self.df = self.df.drop_duplicates()
        self.df = self.df.reset_index().set_index(['user_id', 'save_date']).sort_values(['user_id', 'save_date']).drop_duplicates(keep='last')
        #print(self.df)

    def get_users_in_list(self, user_list: list):
        return self.df[self.df.index.isin(user_list)]

class DataLoaderStatic(DataStatic):

    #Data variables
    user_list = []
    question_list = []

    # Load raw data
    def __init__(self, static_file, user):
        super().__init__(static_file)
        self.df = self.df.loc[[user]]
        print(self.df)

class Patient(DataLoaderEMAS, DataLoaderStatic):
    def __init__(self,emas_file,static_file,user_list):
        for user in user_list:
            DataLoaderStatic.__init__(self, static_file, user)
            DataLoaderEMAS.__init__(self, emas_file, user)



#user_list = [66,73]
#question_list = ['ts','question1','question2']
#patient = Patient('/Users/hang/Desktop/mHealth - Teamprojects/TrackYourTinnitus/Old format/tyt_emas.csv','/Users/hang/Desktop/mHealth - Teamprojects/TrackYourTinnitus/Old format/tyt_static.csv',user_list)
#data = DataEMAS('/Users/hang/Desktop/mHealth - Teamprojects/TrackYourTinnitus/Old format/tyt_emas.csv')
#print(data.get_daily())
#print(patient)
