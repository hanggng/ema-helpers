import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

class DataLoaderEMA:

    #Data variables
    user_list = []
    question_list = []

    # Load raw data
    def __init__(self, file):
        """
        Load raw data
        :param file: file path
        :type file
        """
        self.df = pd.read_csv(file, index_col=0)
        self.df.ts = pd.to_datetime(self.df.ts)
        print(self.df)

    # Get_question_data
    def get_ques(self, question_list: list):
        """
        get questioned data
        :param question_list: a list of desired questions
        @:type question_list: list
        """
        return self.df[self.df.columns[self.df.columns.isin(question_list)]]


    # Get all data for users (EMA) or users in lists
    def get_users_in_list(self, user_list: list):
        """
        get all data for users (EMA) or users in lists
        :param user_list: list of questioned users
        @:type user_list: list
        """
        return self.df[self.df.index.isin(user_list)]

    # Get users with sufficient data at least n_days
    def user_n_days(self, n: int):
        """
        Get users with sufficient data at least n_days
        :param n: number of days
        """
        self.df = self.df.dropna().drop_duplicates()
        self.df['date'] = self.df.ts.dt.date
        return self.df[self.df.groupby(['entity'])['date'].nunique() >= n]

    # Get users with sufficient data at least n_sessions
    def user_n_sessions(self, n: int):
        """
        Get users with sufficient data at least n_sessions
        :param n: number of sessions
        """
        self.df = self.df.dropna().drop_duplicates()
        return self.df[self.df.groupby(['entity'])['ts'].nunique() >= n]

    # Get users with sufficient data at least n_sessions per day
    def user_n_sessions_per_day(self, n: int):
        """
        Get users with sufficient data at least n_sessions per day
        :param n: number of sessions per day
        @:type n: int
        """
        self.df = self.df.dropna().drop_duplicates()
        return self.df[self.df.groupby(['entity','date'])['ts'].nunique() >= n]

    # Get longest uninterrupted sequence all_users_in_set
    def max_sequence(self, user_list: list):
        """
        Get longest uninterrupted sequence all_users_in_set
        :param user_list: list of questioned users
        @type user_list: list
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
        Get longest sequence with max_gap_size <= G
        :param user_list: list of questioned users
        :param n: maximal gap size in days
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
        Get all uninterrupted sequences
        :param user_list: list of questioned users
        @type user_list: list
        """
        all_uni_seq = []
        for x in user_list:
            user_df = self.loc[[x], ['ts']]
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
        '''
        Get all sequences with max_gap_size <= G
        :param user_list: list of questioned users
        :param n:gap size in days
        :return: all sequences with max_gap_size <= G of all users in list
        '''
        all_uni_seq = []
        for x in user_list:
            user_df = self.loc[[x], ['ts']]
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
        Standardise user_level_emas (sklearn.preprocessing.StandardScaler) scaler on each user in all data
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
        Standardise user_level_emas (sklearn.preprocessing.StandardScaler) scaler on each user in user list
        :param user_list: list of questioned users
        @:type user_list: list
        """
        scaler = StandardScaler()
        user_df = self.df[self.df.index.isin(user_list)].drop(['ts'], axis=1).sort_index()
        scaler.fit(user_df)
        scaled_features = scaler.transform(user_df)
        df_feat = pd.DataFrame(scaled_features, index=user_df.index, columns=user_df.columns)
        return df_feat

    # Get user EMA with time-since-last-ema as new column
    def time_since_last_ema(self, user_list: list):
        '''
        :param user_list: list of asked users
        @:type user_list: list
        :return: user EMA with time-since-last-ema as new column
        '''
        for x in user_list:
            user_df = self.df.loc[[x]].sort_index()
            user_df['date'] = user_df.ts.dt.date
            user_df['time_since_last_ema'] = user_df['date'] - user_df['date'].shift()
            return user_df

    # Get time series data in a way that is compatible with Kats:
    def get_Kats(self):
        """
        Get time series data in a way that is compatible with Kats
        """
        return self.df.groupby(self.df.index)['ts'].apply(list).to_dict()

    # Load daily data (data type changed to float because of NaN)
    def get_daily(self):
        """
        Load daily data (data type changed to float because of NaN)
        """
        self.df['date'] = self.df.ts.dt.date
        daily = self.df.reset_index().set_index('date').sort_index()
        idx = pd.Series(np.nan, index=pd.date_range(daily.index.min(), daily.index.max(), freq='D'))
        daily = pd.concat([daily, idx[~idx.index.isin(daily.index)]]).sort_index()
        daily.index.name = 'date'
        daily = daily.set_index([daily.index, 'ts']).sort_index().drop(0, axis=1)
        return daily

    # Relative time
    def rel_time(self, user_list: list):
        """
        Add relative time as a new column
        :param user_list: list of questioned users
        @type user_list: list
        """
        self.df['date'] = self.df.ts.dt.date
        for x in user_list:
            user_df = self.df.loc[[x]]
            user_df['user_rel_time'] = user_df['date'] - user_df['date'].iloc[0]
            return user_df

    # Get user data with gaps encoded using some given value (input: Encode gaps as 1000, for eg)
    def gap_encoded(self, user_list: list, value: int):
        """
        Get user data with gaps encoded using some given value (input: Encode gaps as 1000, for eg)
        :param user_list: list of questioned users
        :param value: value to fill in
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
        Get N-most-similar users with Input user-distance matrix
        :param dist_matrix: diagonal matrix
        :param k: top k most similar users for each user
        """
        dist_matrix = dist_matrix.T
        for i in dist_matrix.index:
            k_matrix = dist_matrix.nsmallest(k, i)
            k_matrix = k_matrix.T
            k_matrix = k_matrix[k_matrix.index == i]
            k_matrix = k_matrix.loc[:, k_matrix.any()]
            for j in k_matrix.index:
                print(i , ": ", list(k_matrix.columns))









data = DataLoaderEMA('/Users/hang/Desktop/mHealth - Teamprojects/TrackYourTinnitus/Old format/tyt_emas.csv')
#question_list = ['ts','question1','question2']
#print(data.get_ques(question_list))
#user_list = [73]
#data.get_users_in_list(user_list)
#data.user_n_day(7)
#print(data.max_sequence(user_list))
#data.max_sequence_gap(user_list,5)
#data.standard_scaler_all()
#data.standard_scaler_list(user_list)
#print(data.time_since_last_ema(user_list))
#data.get_Kats()
#data.get_daily()
#data.gap_encoded(user_list,1000)
#dist_matrix = pd.DataFrame(np.diag(np.full(5,0)))
#data.kNN_user(dist_matrix,2)
#print(data.rel_time(user_list))


