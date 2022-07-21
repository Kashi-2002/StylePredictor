from classifier import load_dataframe, vectorcreatorlabel
from classifier import clean_data
from classifier import one_at_time
from classifier import two_at_time
from classifier import vectorcreator
from classifier import classifier
from classifier import predictions
from classifier import get_index_of_closest_variable
from classifier import compute_eval_metrics
from classifier import checkifpossible
from classifier import compute_var_double_ispossible
from classifier import crosspossible
from classifier import split


from absl import app
from absl import flags
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


FLAGS = flags.FLAGS

flags.DEFINE_string("pathtocsv",None,'Path to the csv', short_name='path')
flags.DEFINE_string("globalorsel",None,'Global/Selected Features Thresholding', short_name='glob')
flags.DEFINE_list('stylename', None,'Input the style for which function should run',short_name='style')
flags.DEFINE_integer('frequencythreshold', None,'Frequency of occurence of paticular feature',short_name='freq')
flags.DEFINE_float('positiveprobability', None,'The positive probability threshold',short_name='pos')
flags.DEFINE_float('negativeprobability', None,'The negative probability threshold',short_name='neg')
flags.DEFINE_bool('applicationofvariance', None,'Defines whether to apply the variance threshold',short_name='var')
flags.DEFINE_bool('applicationofcross', None,'Defines whether to apply the cross filtering threshold',short_name='cross')

flags.mark_flag_as_required("pathtocsv")
flags.mark_flag_as_required('stylename')
flags.mark_flag_as_required('positiveprobability') 
flags.mark_flag_as_required('negativeprobability') 
flags.mark_flag_as_required('applicationofvariance') 
flags.mark_flag_as_required('applicationofcross')
flags.mark_flag_as_required("globalorsel")

def main(argv):
    del argv
    dataset=load_dataframe(FLAGS.pathtocsv)
    data=clean_data(dataset)
    listofstyle=data['Style'].unique()
    dataframe,X_test,y_train,y_test=split(data)
    dataframe['Style']=y_train
    X_test['Style']=y_test

    uniquekey=[]
    uniqueval=[]
    
    for i in dataframe.columns:
        if(i=='Style'):
            continue
        else:
            for j in range(len(dataframe[i].unique())):
                uniquekey.append(i)
                uniqueval.append(dataframe[i].unique()[j])
        
    uniqueattr=list(zip(uniquekey,uniqueval))

    # print(len(uniqueattr))
    # print(uniqueattr)

    single_attr=[]
    for i in range(len(uniqueattr)):
        if(uniqueattr[i][1]=='-'):
            continue
        else:
            if(one_at_time(FLAGS.stylename,uniqueattr[i][1],uniqueattr[i][0],dataframe,FLAGS.frequencythreshold,FLAGS.positiveprobability,FLAGS.negativeprobability)!=None):
                key,keys=one_at_time(FLAGS.stylename,uniqueattr[i][1],uniqueattr[i][0],dataframe,FLAGS.frequencythreshold,FLAGS.positiveprobability,FLAGS.negativeprobability)
                single_attr.append(list(zip(key,keys))[0])
            # if((single!=None) & (double!=None)):
            #     print(single)
            #     print(double)
            # if(single!=None):
                #  single_attr.append(single)
            # print(single)
    print(f"Length of the single features selected are {len(single_attr)}")
    # print(single_attr[0])



    cross_attr__1=[]
    cross_attr__2=[]

    for i in range(len(uniqueattr)):
        for j in range(i,len(uniqueattr)):
            if((uniqueattr[i][1]=='-') | (uniqueattr[j][1]=='-') ):
                continue
            else:
                if(uniqueattr[i][1]!=uniqueattr[j][1]):
                    if(len(dataframe[(dataframe[uniqueattr[i][0]]==uniqueattr[i][1]) & (dataframe[uniqueattr[j][0]]==uniqueattr[j][1])])>0):
                        if(two_at_time(FLAGS.stylename,uniqueattr[i][1],uniqueattr[j][1],uniqueattr[i][0],uniqueattr[j][0],dataframe,FLAGS.frequencythreshold,FLAGS.positiveprobability,FLAGS.negativeprobability)!=None):
                            cross_1,cross_2=two_at_time(FLAGS.stylename,uniqueattr[i][1],uniqueattr[j][1],uniqueattr[i][0],uniqueattr[j][0],dataframe,FLAGS.frequencythreshold,FLAGS.positiveprobability,FLAGS.negativeprobability)
                            cross_attr__1.append(cross_1[0])
                            cross_attr__2.append(cross_2[0])
            
    print(f"Length of the single features selected are {len(cross_attr__1)}")
#    r pint(len(cross_attr__1))
    # print(cross_attr__1[0])
    # print()


    if(FLAGS.globalorsel=="Selected"):
        
        single_var=[]
        cross_var_1=[]
        cross_var_2=[]
        cross_after_change1=[]
        cross_after_change2=[]

        application_both_sing=[]
        application_both_cross1=[]
        application_both_cross2=[]

        if((FLAGS.applicationofvariance!=False) & (FLAGS.applicationofcross==False) ):
            # print('yo')
            for i in range(len(single_attr)):
                if(checkifpossible(listofstyle,single_attr[i][0],single_attr[i][1],dataframe,FLAGS.stylename)!=None):
                    single_var.append(checkifpossible(listofstyle,single_attr[i][0],single_attr[i][1],dataframe,FLAGS.stylename))


            for i in range(len(cross_attr__1)):
                # print('m')
                if(compute_var_double_ispossible(listofstyle,cross_attr__1[i][0],cross_attr__2[i][0],cross_attr__1[i][1],cross_attr__2[i][1],dataframe,FLAGS.stylename)!=None):
                    kross,kross1=compute_var_double_ispossible(listofstyle,cross_attr__1[i][0],cross_attr__2[i][0],cross_attr__1[i][1],cross_attr__2[i][1],dataframe,FLAGS.stylename)
                    cross_var_1.append(kross)
                    cross_var_2.append(kross1)
            print(f"After applying variance filter accross the features the resultant single features is of size {len(single_var)} and crosses is of size {len(cross_var_1)} ")

            possibletrain=[]
            for i in range(len(dataframe)):
                possibletrain.append(vectorcreator(single_var,cross_var_1,cross_var_2,dataframe.iloc[[i]]))


            possibletest=[]
            for i in range(len(X_test)):
                possibletest.append(vectorcreator(single_var,cross_var_1,cross_var_2,X_test.iloc[[i]]))



            trainlabel=vectorcreatorlabel(y_train,FLAGS.stylename)
            testlabel=vectorcreatorlabel(y_test,FLAGS.stylename)

            X=pd.DataFrame(possibletrain)
            y=pd.DataFrame(trainlabel)
            test=pd.DataFrame(possibletest)

            clf=classifier(X,y)
            pred,predprobs=predictions(test,clf)

            print(compute_eval_metrics(testlabel,predprobs))


        elif((FLAGS.applicationofvariance==False) & (FLAGS.applicationofcross!=False)):
            # print('yo')
            for i in range(len(cross_attr__1)):
                # print('yo')
                if(crosspossible(FLAGS.stylename,cross_attr__1[i][0],cross_attr__2[i][0],cross_attr__1[i][1],cross_attr__2[i][1],dataframe,FLAGS.frequencythreshold,FLAGS.positiveprobability,FLAGS.negativeprobability)!=None):
                    cross_after_filter1,cross_after_filter2=crosspossible(FLAGS.stylename,cross_attr__1[i][0],cross_attr__2[i][0],cross_attr__1[i][1],cross_attr__2[i][1],dataframe,FLAGS.frequencythreshold,FLAGS.positiveprobability,FLAGS.negativeprobability)
                    # print(cross_after_filter2)
                    cross_after_change1.append(cross_after_filter1[0])
                    cross_after_change2.append(cross_after_filter2[0]) 
            print(f"After applying cross filtering across features the length of cross feature is {len(cross_after_change1)}")

            possibletrain=[]
            for i in range(len(dataframe)):
                possibletrain.append(vectorcreator(single_attr,cross_after_change1,cross_after_change2,dataframe.iloc[[i]]))


            possibletest=[]
            for i in range(len(X_test)):
                possibletest.append(vectorcreator(single_attr,cross_after_change1,cross_after_change2,X_test.iloc[[i]]))



            trainlabel=vectorcreatorlabel(y_train,FLAGS.stylename)
            testlabel=vectorcreatorlabel(y_test,FLAGS.stylename)

            X=pd.DataFrame(possibletrain)
            y=pd.DataFrame(trainlabel)
            test=pd.DataFrame(possibletest)

            clf=classifier(X,y)
            pred,predprobs=predictions(test,clf)

            print(compute_eval_metrics(testlabel,predprobs))


        
        elif((FLAGS.applicationofvariance==True) & (FLAGS.applicationofcross==True)):
            # print('yo both')
            for i in range(len(single_attr)):
                if(checkifpossible(listofstyle,single_attr[i][0],single_attr[i][1],dataframe,FLAGS.stylename)!=None):
                    application_both_sing.append(checkifpossible(listofstyle,single_attr[i][0],single_attr[i][1],dataframe,FLAGS.stylename))
                

            cross_dict1=[]
            cross_dict2=[]

            for i in range(len(cross_attr__1)):
                if(compute_var_double_ispossible(listofstyle,cross_attr__1[i][0],cross_attr__2[i][0],cross_attr__1[i][1],cross_attr__2[i][1],dataframe,FLAGS.stylename)!=None):
                    kross,kross1=compute_var_double_ispossible(listofstyle,cross_attr__1[i][0],cross_attr__2[i][0],cross_attr__1[i][1],cross_attr__2[i][1],dataframe,FLAGS.stylename)
                    cross_dict1.append(kross)
                    cross_dict2.append(kross1)

            for i in range(len(cross_dict1)):
                # print('yo')
                if(crosspossible(FLAGS.stylename,cross_dict1[i][0],cross_dict2[i][0],cross_dict1[i][1],cross_dict2[i][1],dataframe,FLAGS.frequencythreshold,FLAGS.positiveprobability,FLAGS.negativeprobability)!=None):
                    cross_after_filter1,cross_after_filter2=crosspossible(FLAGS.stylename,cross_dict1[i][0],cross_dict2[i][0],cross_dict1[i][1],cross_dict2[i][1],dataframe,FLAGS.frequencythreshold,FLAGS.positiveprobability,FLAGS.negativeprobability)
                    # print(cross_after_filter2)
                    application_both_cross1.append(cross_after_filter1[0])
                    application_both_cross2.append(cross_after_filter2[0])

            print(f"The length of features after application of  both filters is {len(application_both_sing)} and {len(application_both_cross1)}")
            
                        

            possibletrain=[]
            for i in range(len(dataframe)):
                possibletrain.append(vectorcreator(application_both_sing,application_both_cross1,application_both_cross2,dataframe.iloc[[i]]))


            possibletest=[]
            for i in range(len(X_test)):
                possibletest.append(vectorcreator(application_both_sing,application_both_cross1,application_both_cross2,X_test.iloc[[i]]))



            trainlabel=vectorcreatorlabel(y_train,FLAGS.stylename)
            testlabel=vectorcreatorlabel(y_test,FLAGS.stylename)

            X=pd.DataFrame(possibletrain)
            y=pd.DataFrame(trainlabel)
            test=pd.DataFrame(possibletest)

            clf=classifier(X,y)
            pred,predprobs=predictions(test,clf)

            print(compute_eval_metrics(testlabel,predprobs))

        elif((FLAGS.applicationofvariance==False) & (FLAGS.applicationofcross==False)):
            possibletrain=[]
            for i in range(len(dataframe)):
                possibletrain.append(vectorcreator(single_attr,cross_attr__1,cross_attr__2,dataframe.iloc[[i]]))


            possibletest=[]
            for i in range(len(X_test)):
                possibletest.append(vectorcreator(single_attr,cross_attr__1,cross_attr__2,X_test.iloc[[i]]))



            trainlabel=vectorcreatorlabel(y_train,FLAGS.stylename)
            testlabel=vectorcreatorlabel(y_test,FLAGS.stylename)

            X=pd.DataFrame(possibletrain)
            y=pd.DataFrame(trainlabel)
            test=pd.DataFrame(possibletest)

            clf=classifier(X,y)
            pred,predprobs=predictions(test,clf)

            print(compute_eval_metrics(testlabel,predprobs))

## marks the end of code when the feature filtering in not global and applied on the features that are selected from the 
## threshold values.

    elif(FLAGS.globalorsel=="Global"):
        single_var=[]
        cross_var_1=[]
        cross_var_2=[]
        cross_after_change1=[]
        cross_after_change2=[]

        application_both_sing=[]
        application_both_cross1=[]
        application_both_cross2=[]

        if((FLAGS.applicationofvariance!=False) & (FLAGS.applicationofcross==False) ):
            # print('yo')
            for i in range(len(uniqueattr)):
                if(uniqueattr[i][1]=='-'):
                    continue
                else:
                    if(len(dataframe[dataframe[uniqueattr[i][0]]==uniqueattr[i][1]])>FLAGS.frequencythreshold):
                        if(checkifpossible(listofstyle,uniqueattr[i][1],uniqueattr[i][0],dataframe,FLAGS.stylename)!=None):
                            single_var.append(checkifpossible(listofstyle,uniqueattr[i][1],uniqueattr[i][0],dataframe,FLAGS.stylename))


            for i in range(len(uniqueattr)):
                for j in range(i+1,len(uniqueattr)):
                    if((uniqueattr[i][1]=='-' )| (uniqueattr[j][1]=='-')):
                        continue
                    else:
                        if(uniqueattr[i][0]!=uniqueattr[j][0]):
                            if(len(dataframe[(dataframe[uniqueattr[i][0]]==uniqueattr[i][1]) & (dataframe[uniqueattr[j][0]]==uniqueattr[j][1])])>FLAGS.frequencythreshold):
                                if(compute_var_double_ispossible(listofstyle,uniqueattr[i][1],uniqueattr[j][1],uniqueattr[i][0],uniqueattr[j][0],dataframe,FLAGS.stylename)!=None):
                                    kross,kross1=compute_var_double_ispossible(listofstyle,uniqueattr[i][1],uniqueattr[j][1],uniqueattr[i][0],uniqueattr[j][0],dataframe,FLAGS.stylename)
                                    cross_var_1.append(kross)
                                    cross_var_2.append(kross1)
            print(f"After applying variance filter accross the features the resultant single features is of size {len(single_var)} and crosses is of size {len(cross_var_1)} ")

            possibletrain=[]
            for i in range(len(dataframe)):
                possibletrain.append(vectorcreator(single_var,cross_var_1,cross_var_2,dataframe.iloc[[i]]))


            possibletest=[]
            for i in range(len(X_test)):
                possibletest.append(vectorcreator(single_var,cross_var_1,cross_var_2,X_test.iloc[[i]]))



            trainlabel=vectorcreatorlabel(y_train,FLAGS.stylename)
            testlabel=vectorcreatorlabel(y_test,FLAGS.stylename)

            X=pd.DataFrame(possibletrain)
            y=pd.DataFrame(trainlabel)
            test=pd.DataFrame(possibletest)

            clf=classifier(X,y)
            pred,predprobs=predictions(test,clf)

            print(compute_eval_metrics(testlabel,predprobs))


        elif((FLAGS.applicationofvariance==False) & (FLAGS.applicationofcross!=False)):
            for i in range(len(uniqueattr)):
                for j in range(i+1,len(uniqueattr)):
                    if((uniqueattr[i][1]=='-' )| (uniqueattr[j][1]=='-')):
                        continue
                    else:
                        if(uniqueattr[i][0]!=uniqueattr[j][0]):
                            if(len(dataframe[(dataframe[uniqueattr[i][0]]==uniqueattr[i][1]) & (dataframe[uniqueattr[j][0]]==uniqueattr[j][1])])>FLAGS.frequencythreshold):
                                if(crosspossible(FLAGS.stylename,cross_attr__1[i][0],cross_attr__2[i][0],cross_attr__1[i][1],cross_attr__2[i][1],dataframe,FLAGS.frequencythreshold,FLAGS.positiveprobability,FLAGS.negativeprobability)!=None):
                                    cross_after_filter1,cross_after_filter2=crosspossible(FLAGS.stylename,cross_attr__1[i][0],cross_attr__2[i][0],cross_attr__1[i][1],cross_attr__2[i][1],dataframe,FLAGS.frequencythreshold,FLAGS.positiveprobability,FLAGS.negativeprobability)
                                    cross_after_change1.append(cross_after_filter1[0])
                                    cross_after_change2.append(cross_after_filter2[0]) 
            
            print(f"After applying cross filtering across features the length of cross feature is {len(cross_after_change1)}")

            possibletrain=[]
            for i in range(len(dataframe)):
                possibletrain.append(vectorcreator(single_attr,cross_after_change1,cross_after_change2,dataframe.iloc[[i]]))


            possibletest=[]
            for i in range(len(X_test)):
                possibletest.append(vectorcreator(single_attr,cross_after_change1,cross_after_change2,X_test.iloc[[i]]))



            trainlabel=vectorcreatorlabel(y_train,FLAGS.stylename)
            testlabel=vectorcreatorlabel(y_test,FLAGS.stylename)

            X=pd.DataFrame(possibletrain)
            y=pd.DataFrame(trainlabel)
            test=pd.DataFrame(possibletest)

            clf=classifier(X,y)
            pred,predprobs=predictions(test,clf)

            print(compute_eval_metrics(testlabel,predprobs))

        elif((FLAGS.applicationofvariance==False) & (FLAGS.applicationofcross==False)):
            possibletrain=[]
            for i in range(len(dataframe)):
                possibletrain.append(vectorcreator(single_attr,cross_attr__1,cross_attr__2,dataframe.iloc[[i]]))


            possibletest=[]
            for i in range(len(X_test)):
                possibletest.append(vectorcreator(single_attr,cross_attr__1,cross_attr__2,X_test.iloc[[i]]))



            trainlabel=vectorcreatorlabel(y_train,FLAGS.stylename)
            testlabel=vectorcreatorlabel(y_test,FLAGS.stylename)

            X=pd.DataFrame(possibletrain)
            y=pd.DataFrame(trainlabel)
            test=pd.DataFrame(possibletest)

            clf=classifier(X,y)
            pred,predprobs=predictions(test,clf)

            print(compute_eval_metrics(testlabel,predprobs))                                    


if __name__ == "__main__":
    app.run(main)