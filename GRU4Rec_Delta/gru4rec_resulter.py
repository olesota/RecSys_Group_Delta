import gru4rec
import pandas as pd

def get_submission_target(df):
    """Identify target rows with missing click outs."""

    mask = df["reference"].isnull() & (df["action_type"] == "clickout item")
    df_out = df[mask]

    return df_out

def gru_test_19(gru, test_file, batch):
    test = pd.read_csv(test_file, sep=',')
    target = get_submission_target(test)

def string_to_array(s):
    """Convert pipe separated string to array."""

    if isinstance(s, str):
        out = s.split("|")
    elif math.isnan(s):
        out = []
    else:
        raise ValueError("Value must be either string of nan")
    return out
# посколько сразу проссать как хорошо оптимизировать сессии в ровный поток я не смог, делаем кпрощенный алгоритм.
# суть: запихивать в эту адскую машинку сначала сессии одинакового размера пачками по n=200 штук.
# пусть у нас 212 сессий длины 3. Тогда в первый заход мы обработаем 200 сессий, выдадим результат, потом 
# обработаем 12 оставшихся сессий, выдадим результат. Потом переходим к сессиям длины 4.
# в качестве ALL_IMPRESSIONS подаются слитые impressions всех сессий, подаваемых на вход. Это все равно граздо меньше
# чем все айтемы! ОБА-НА!

def process_batch(gru, df, batch_size, all_impressions):
    # примерный вид df:
    # session_id1  session_id2  session_id3  ...  session_id<BATCH_SIZE>
    #   item_id*     item_id*     item_id*   ...        item_id*
    #     ...          ...          ...      ...          ...
    session_ids = list(df.columns.values)
    fake_session_ids = list(range(len(session_ids))) # чтобы не пугать ГРУшный метод... хаха груша.
    for index, step_items in df.iterrows():
        preds = gru.predict_next_batch(fake_session_ids, list(step_items) ,all_impressions, batch_size)
    # прокрутили все сессии теперь в preds лежит то что нужно !
    preds.columns = session_ids
    return preds

###
def process_batch_test_dummy(df, batch_size, all_impressions):
    # примерный вид df:
    # session_id1  session_id2  session_id3  ...  session_id<BATCH_SIZE>
    #   item_id*     item_id*     item_id*   ...        item_id*
    #     ...          ...          ...      ...          ...
    session_ids = list(df.columns.values)
    print(session_ids)
    fake_session_ids = list(range(len(session_ids))) # чтобы не пугать ГРУшный метод... хаха груша.
    print(fake_session_ids)
    for index, step_items in df.iterrows():
        #preds = gru.predict_next_batch(fake_session_ids, list(step_items) ,all_impressions, batch_size)
        print(list(step_items))
    # прокрутили все сессии теперь в preds лежит то что нужно !
    #preds.columns = session_ids
    #return preds
####

def form_results(target_df, preds):
    #target_df dataframes with empty click-outs
    
    #accumulating result for this batch in to_print
    to_print = target_df[['user_id','session_id','timestamp','step']]
    to_print['item_recommendations'] = ''
    for session in preds.columns.values:
        print(str(session))
        tmp = list(preds[session].sort_values(0, False).index.values) # sorted all impressions for a single session:
        #print(tmp)
        given_impressions = frozenset(string_to_array(target_df[target_df['session_id'] == session].iloc[0]['impressions']))
        #print(given_impressions)
        cur_prediction = [x for x in tmp if x in given_impressions]
        #print(cur_prediction)
        indx = to_print.index[to_print['session_id'] == session].tolist()[0]
        to_print.set_value(indx, 'item_recommendations', ' '.join(str(x) for x in cur_prediction))
        #to_print[to_print['session_id'] == session].loc[0,'item_recommendations'] = ' '.join(str(x) for x in intersection)
    return to_print

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def process_sessions_of_length(gru, in_df, max_batch, file):
    # in_df - log containing sessions of needed length + EMPTY CLICKOUT (all equal)
    # max_batch
    
    mask = in_df["reference"].isnull() & (in_df["action_type"] == "clickout item")
    target = in_df[mask] # click_outs with no references
    actions = in_df[-mask] # only actions with references
    
    # batching session_ids 
    ids = list(target['session_id'].values)
    queue = list(chunks(ids, max_batch)) # list of lists of session ids
    
    # forming input for batch_predict:
    #
    # session_id1  session_id2  session_id3  ...  session_id<BATCH_SIZE>
    #   item_id*     item_id*     item_id*   ...        item_id*
    #     ...          ...          ...      ...          ...    
    formed_sessions = pd.DataFrame()
    for s in set(ids): # go over all sessions
        z = list(actions[actions['session_id'] == s]['reference']) # for each session create a column with list of references
        formed_sessions[s] = z # formed sessions!
    
    print(formed_sessions)
    # processing itself
    for step in queue:
        # load session ids for current step
        cur_sessions = formed_sessions[step] # load session ids for current step
        # generate all impressions for the batch
        batch_target = target[target['session_id'].isin(step)]
        print(batch_target)
        batch_impressions_str = '|'.join(str(x) for x in list(batch_target['impressions'])) # all impressions as pipe-sep list
        batch_impressions = set(string_to_array(batch_impressions_str)) # all impressions as set
        print(batch_impressions)
        batch_preds = process_batch(gru, cur_sessions, len(step), batch_impressions)
        batch_res = form_results(batch_target, batch_preds)
        # write batch res to a file
        with open(file, 'a') as f:
            batch_res.to_csv(f, header=False)

def go_predict_gru(gru, test, file):
    k = test['session_id'].value_counts
    for i in range(2,k.max()):
        print("processing sessions of length " + str(i))
        if i<4: 
            batch_size = 500
        else:
            batch_size = 250
        print("batch size: "+str(batch_size))
        cur_sessions = list(k.index[k == i])
        iter_sessions = test[test['session_id'].isin(cur_sessions)]
        print("sessions in the iteration: "+str(len(cur_sessions)))
        process_sessions_of_length(gru=gru,file=file,in_df=iter_sessions,max_batch=batch_size)
        print("--- done ---")