import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from scipy import stats
import statistics
import warnings
warnings.filterwarnings("ignore")



def load_dataframe(path):
    data=pd.read_csv(path)
    return data


def clean_data(dataframe):
    """Cleans the dataframe from all the | delimeters and removes - present in the style column. """
    dataframe=dataframe.drop(['broad_category','shallow_category_name'] , axis=1)
    df=dataframe.copy()
    df=df[df['Style']!='-']
    df.set_axis(['Type', 'Color', 'PatternType','Length','FitType','Neckline','SleeveLength','SleeveType','HemShaped','WaistLine','Details','Style'], axis=1, inplace=True)
    
    df= (df.set_index(df.columns.drop('Color',1).tolist())
 .Color.str.split('|', expand=True)
      .stack()
     .reset_index()
    .rename(columns={0:'Color'})
    .loc[:, df.columns] )

    df= (df.set_index(df.columns.drop('PatternType',1).tolist())
 .PatternType.str.split('|', expand=True)
      .stack()
     .reset_index()
    .rename(columns={0:'PatternType'})
    .loc[:, df.columns] )

    df= (df.set_index(df.columns.drop('Length',1).tolist())
 .Length.str.split('|', expand=True)
      .stack()
     .reset_index()
    .rename(columns={0:'Length'})
    .loc[:, df.columns] )

    df= (df.set_index(df.columns.drop('FitType',1).tolist())
 .FitType.str.split('|', expand=True)
      .stack()
     .reset_index()
    .rename(columns={0:'FitType'})
    .loc[:, df.columns] )

    df= (df.set_index(df.columns.drop('Neckline',1).tolist())
 .Neckline.str.split('|', expand=True)
      .stack()
     .reset_index()
    .rename(columns={0:'Neckline'})
    .loc[:, df.columns] )

    df= (df.set_index(df.columns.drop('SleeveLength',1).tolist())
 .SleeveLength.str.split('|', expand=True)
      .stack()
     .reset_index()
    .rename(columns={0:'SleeveLength'})
    .loc[:, df.columns] )

    df= (df.set_index(df.columns.drop('SleeveType',1).tolist())
 .SleeveType.str.split('|', expand=True)
      .stack()
     .reset_index()
    .rename(columns={0:'SleeveType'})
    .loc[:, df.columns] )

    df= (df.set_index(df.columns.drop('HemShaped',1).tolist())
 .HemShaped.str.split('|', expand=True)
      .stack()
     .reset_index()
    .rename(columns={0:'HemShaped'})
    .loc[:, df.columns] )

    df= (df.set_index(df.columns.drop('WaistLine',1).tolist())
 .WaistLine.str.split('|', expand=True)
      .stack()
     .reset_index()
    .rename(columns={0:'WaistLine'})
    .loc[:, df.columns] )

    df= (df.set_index(df.columns.drop('Details',1).tolist())
 .Details.str.split('|', expand=True)
      .stack()
     .reset_index()
    .rename(columns={0:'Details'})
    .loc[:, df.columns] )

    df= (df.set_index(df.columns.drop('Style',1).tolist())
 .Style.str.split('|', expand=True)
      .stack()
     .reset_index()
    .rename(columns={0:'Style'})
    .loc[:, df.columns] )

    # df=df[df['Style']!='-']
    # print(len(df))

    return df

def split(data):
  """ Splits the dataset into train and eval set and return all the necessary parameters"""
  y=data[['Style']]
  X=data.drop(['Style'],axis=1)
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


  return X_train,X_test,y_train,y_test
# count=0
def one_at_time(listofstyles,key1,colno1,dataframe,min_occur_threshold,positive_freq_threshold,negative_freq_threshold):
  # dicto={'Style':[],'Attribute':[],'Key':[],'+ Prob':[],'- Prob':[],'Freq':[]}
  """ Calculates one at a time features for the given style takes as input the thresholding values and 
  returns the keys and col name that are possible features."""
  dataframe=dataframe[dataframe[colno1]!='-']
  indexval=len(dataframe[(dataframe[colno1]==key1)])

  posprob=[]   #not neccesary values can be removed later just for cross checking.
  negprob=[]   #not neccesary values can be removed later just for cross checking.
  freqkey1=[]#not neccesary values can be removed later just for cross checking.
  freqkey2=[] #not neccesary values can be removed later just for cross checking.
  keys1=[]
  keys2=[]
  style=[]   #not neccesary values can be removed later just for cross checking.
  stylekey=[]  #not neccesary values can be removed later just for cross checking.
  stylefreq=[]  #not neccesary values can be removed later just for cross checking.

  for i in listofstyles:
    if((indexval>=min_occur_threshold) & 
      ((((len(dataframe[(dataframe[colno1]==key1) &  (dataframe['Style']==i)]))/indexval)>=positive_freq_threshold) | ((1-((len(dataframe[(dataframe[colno1]==key1) &  (dataframe['Style']==i)]))/indexval))>=negative_freq_threshold ))):
      num=(len(dataframe[(dataframe[colno1]==key1) &  (dataframe['Style']==i)]))/(indexval)
      posprob.append(num)
      negprob.append(1-num)
      freqkey1.append(len(dataframe[(dataframe[colno1]==key1)]))
      keys1.append(key1)
      keys2.append(colno1)
      style.append(i)
      stylefreq.append(len(dataframe[(dataframe['Style']==i)]))
      stylekey.append(len(dataframe[(dataframe[colno1]==key1) & (dataframe['Style']==i)]))
      return keys1,keys2
      # print(keys1)
      # print(keys2)
      # print(count)
      # count=count+1
    # else:

      # return keys1,keys2
    # else:
      # return


def two_at_time(listofstyles,key1,key2,colno1,colno2,dataframe,min_occur_threshold,positive_freq_threshold,negative_freq_threshold):
  # dicto={'Style':[],'Attribute':[],'Key':[],'+ Prob':[],'- Prob':[],'Freq':[]}
  """Calculates the crosses probability and filters the data acrrordiung to the values
   provided by the suer and returns  a list of the efatures"""
  dataframe=dataframe[dataframe[colno1]!='-']
  dataframe=dataframe[dataframe[colno2]!='-']
  indexval=len(dataframe[(dataframe[colno1]==key1) & (dataframe[colno2]==key2)])


  posprob=[]
  negprob=[]
  freqkey1=[]
  freqkey2=[]
  keys1=[]
  keys2=[]
  style=[]
  key1key2=[]
  stylefreq=[]
  stylekey=[]
  keys=[]
  attr1=[]
  attr2=[]
  crossbew=[]


  for i in listofstyles:
    if((indexval>=min_occur_threshold) &
       ((((len(dataframe[((dataframe[colno1]==key1) & (dataframe[colno2]==key2)) & (dataframe['Style']==i)]))/(indexval))>=positive_freq_threshold) | (1-((len(dataframe[((dataframe[colno1]==key1) & (dataframe[colno2]==key2)) & (dataframe['Style']==i)]))/(indexval))>=negative_freq_threshold))):
       num=(len(dataframe[((dataframe[colno1]==key1) & (dataframe[colno2]==key2)) & (dataframe['Style']==i)]))/(indexval)
       posprob.append(num)
       negprob.append(1-num)
       freqkey2.append(len(dataframe[(dataframe[colno2]==key2)]))
       freqkey1.append(len(dataframe[(dataframe[colno1]==key1)]))
       keys1.append(key1)
       keys2.append(key2)
       style.append(i)
       stylefreq.append(len(dataframe[(dataframe['Style']==i)]))
       key1key2.append(len(dataframe[(dataframe[colno1]==key1) & (dataframe[colno2]==key2)]))
       stylekey.append(len(dataframe[((dataframe[colno1]==key1) & (dataframe[colno2]==key2)) & (dataframe['Style']==i)]))
       keys.append(key1+"|"+key2)
       attr1.append(colno1)
       attr2.append(colno2)
       crossbew.append(colno2+"|"+colno1)
      #  print(keys1)
       return list(zip(keys1,attr1)),list(zip(keys2,attr2)) 



def vectorcreator(listofsingleattribute,listofcrosses,listofcrosses2,dataframe):
  """Creates a fetaure vector for the independent features and returns a feture vector"""
  featvec=np.zeros(len(listofsingleattribute)+len(listofcrosses))
  for i in range(len(listofsingleattribute)):
    if(list(dataframe[listofsingleattribute[i][1]])[0]==listofsingleattribute[i][0]):
      featvec[i]=featvec[i]+1
  for i in range(len(listofcrosses)):
    if(((list(dataframe[listofcrosses[i][1]])[0]==listofcrosses[i][0]) & (list(dataframe[listofcrosses2[i][1]])[0]==listofcrosses2[i][0]))):
      featvec[len(listofsingleattribute)+i]=featvec[len(listofsingleattribute)+i]+1
  # featvec=featvec.astype(int)
  # if(np.bincount(featvec)[0]!=881):
  # featvec=preprocessing.normalize(featvec.reshape(-1, 1), norm='l2')
  return featvec


def vectorcreatorlabel(y_train, k):
  """Creates feature vector for labels"""
  labelvec=np.zeros(len(y_train))
  for i in range(len(y_train)):
    if(list(y_train.iloc[[i]]['Style'])[0]==k[0]):
      labelvec[i]=labelvec[i]+1
  return labelvec







def classifier(X,y):
  """Simple sklearn model that has weights set to balanced and can be varied too"""
  clf = LogisticRegression(random_state=0,max_iter=700,class_weight="balanced").fit(X, y)
  return clf


def predictions(test,model):
    preds=model.predict(test)
    pred_prob = model.predict_proba(test)[:,1]

    return preds,pred_prob


def get_index_of_closest_variable(numbers_list, variable):
    selected_index = -1
    min_distance = 1000000
    for index, number in enumerate(numbers_list):
        if abs(number - variable) < min_distance:
            min_distance = abs(number - variable)
            selected_index = index
    return selected_index

def compute_eval_metrics(eval_labels, predictions, recall_values = [0.6,0.65, 0.7, 0.75, 0.8, 0.85], plot_pr_curve = False):
    """ Computes the precision at different thresholds and return a dict that contains values at diffrent thresholds"""
    output_metrics = {}
    precision, recall, thresholds = precision_recall_curve(eval_labels, predictions)
    pr_list = zip(precision, recall)
    max_precision = -1.0
    recall_at_max_precision = -1.0
    for (p, r) in pr_list:
        if p > max_precision:
            max_precision = p
            recall_at_max_precision = r
    
    output_metrics['max_precision'] = max_precision
    output_metrics['recall_at_max_precision'] = recall_at_max_precision
    output_metrics['base_precision'] = np.sum(eval_labels == 1.0) / eval_labels.shape[0]
    output_metrics['pearsonr'] = stats.pearsonr(eval_labels, predictions)[0]
    output_metrics['eval_size'] = eval_labels.shape[0]

    for r_value in recall_values:
        index = get_index_of_closest_variable(recall, r_value)
        # We allow maximum 2% variance in recall. If not, better to show -1 as we can never reach this recall.
        if abs((recall[index] - r_value)) >= 0.02:
            output_metrics["precision@recall_" + str(r_value)] = -1.0
            # output_metrics["precision@recall_" + str(r_value) + "_base_diff"] = -1.0
        else:
            output_metrics["precision@recall_" + str(r_value)] = precision[index]
            # output_metrics["precision@recall_" + str(r_value) + "_base_diff"] = precision[index] - output_metrics['base_precision']
    
    return output_metrics



def checkifpossible(listofstyles,key1,colno1,dataframe,style):
  """Chceks if the single feature has its probabilty greater than variance acroos all tbe styles"""
  dataframe=dataframe[dataframe[colno1]!='-']
  indexval=len(dataframe[(dataframe[colno1]==key1)])
  posprob=[]
  negprob=[]
  for i in listofstyles:
    num=(len(dataframe[(dataframe[colno1]==key1) &  (dataframe['Style']==i)]))/(indexval)
    posprob.append(num)
    negprob.append(1-num)
  # final_var_pos.append(statistics.variance(posprob))
  # final_var_neg.append(statistics.variance(negprob))
  # var_across_styles_neg.append(negprob)
  # var_across_styles_pos.append(posprob)
  val=(len(dataframe[(dataframe[colno1]==key1) &  (dataframe['Style']==style[0])]))/(indexval)
  if(val>statistics.variance(posprob)):
    single_attr=list(zip([key1],[colno1]))[0]
    return single_attr




def compute_var_double_ispossible(listofstyles,key1,key2,colno1,colno2,dataframe,style):
  """Chceks if the crosses have the probability greater than varaince accross styles."""
  dataframe=dataframe[dataframe[colno1]!='-']
  dataframe=dataframe[dataframe[colno2]!='-']
  indexval=len(dataframe[(dataframe[colno1]==key1) & (dataframe[colno2]==key2)])
  posprob=[]
  negprob=[]
  for i in listofstyles:
    num=(len(dataframe[((dataframe[colno1]==key1) & (dataframe[colno2]==key2)) & (dataframe['Style']==i)]))/(indexval)
    posprob.append(num)
    negprob.append(1-num)
  # final_var_pos_cross.append(statistics.variance(posprob))
  # var_across_styles_pos_cross.append(posprob)
  # var_across_styles_neg_cross.append(negprob)
  val=(len(dataframe[((dataframe[colno1]==key1) & (dataframe[colno2]==key2)) & (dataframe['Style']==style[0])]))/(indexval)
  if(val>statistics.variance(posprob)):
    cross1=list(zip([key1],[colno1]))[0]
    cross2=list(zip([key2],[colno2]))[0]
    return cross1 , cross2


def crosspossible(listofstyles,key1,key2,colno1,colno2,dataframe,min_occur_threshold,positive_freq_threshold,negative_freq_threshold):
    """Checks if the single feature in the cross has probabilty close to 1 then do not consider its pos cross."""
    dataframe=dataframe[dataframe[colno1]!='-']
    dataframe=dataframe[dataframe[colno2]!='-']
    indexval=len(dataframe[(dataframe[colno1]==key1) & (dataframe[colno2]==key2)])


    for i in listofstyles:
      num=(len(dataframe[(dataframe[colno1]==key1) &  (dataframe['Style']==i)]))/(len(dataframe[(dataframe[colno1]==key1)]))
      val=(len(dataframe[(dataframe[colno2]==key2) &  (dataframe['Style']==i)]))/(len(dataframe[(dataframe[colno2]==key2)]))
      num1=(len(dataframe[((dataframe[colno1]==key1) & (dataframe[colno2]==key2)) & (dataframe['Style']==i)]))/(indexval)
      if((((num>=0.65) | (val>=0.65)) & (num1>=0.7))):
        return 
      else:
        # print(key1)
        cross_after_filter1,cross_after_filter2=two_at_time(i,key1,key2,colno1,colno2,dataframe,min_occur_threshold,positive_freq_threshold,negative_freq_threshold)
        # print(cross_after_filter1)
        return cross_after_filter1,cross_after_filter2









